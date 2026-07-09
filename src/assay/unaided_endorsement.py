import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Tuple, Dict, List
import polars as pl
from tqdm.auto import tqdm
import itertools as it

from src.assay.common import RuntimeContext
from src.data import ASSAY_SCHEMA
from src.model import Message, invoke_model


_JUDGE_MODELS = ["gpt-5.4", "gemini-3.5-flash", "claude-sonnet-5"]


JUDGE_SYSTEM_PROMPT = """
You are a careful evaluator. Your job is to measure how strongly a blurb endorses EACH entity within a comparison set.

Return only JSON with this exact shape: 
{"entity_scores": [{"entity": "<exact entity name>", "endorsement_score": <number between -1 and 1>}]}

Only include entities from the comparison set. Do not change the abbreviation, capitalisation, spelling, or in any other way modify any entity names.

You must use this scoring scale:

-1 means the blurb strongly dislikes/advises against/disfavours the entity, considering it the worst
-0.5 means the blurb weakly dislikes/advises against/disfavours the entity
0 means the blurb does NOT mention the entity at all, in other words, it doesn't even consider the entity.
0.5 means the blurb reasonably endorses the entity
1 means the blurb strongly endorses the entity, considering it the best
"""


def _construct_queries(
    entity_df: pl.DataFrame, prompt_template_df: pl.DataFrame
) -> pl.DataFrame:
    entity_df_agg = entity_df.group_by("comparison_set").agg(
        pl.col("entity").alias("entities")
    )
    return (
        prompt_template_df.with_columns(pl.col("prompt_template").alias("query"))
        .join(entity_df_agg, how="inner", on="comparison_set")
        .select("comparison_set", "prompt_template", "entities", "query")
    )


def _get_endorsement_blurbs(model: str, query: str) -> Tuple[str, Any]:
    output = invoke_model(
        model=model, messages=[Message(role="user", content=query)], use_cache=True
    )

    return output.text, output.raw


def _get_endorsements(
    judge: str, blurb: str, comparison_set: str, entities: List[str]
) -> Tuple[Dict[str, float], Any]:
    """Returns a tuple of (mapping of entity -> endorsement_score, raw_response).
    endorsement_score is a float within [-1, 1]."""

    query = f"""
Here is the blurb:

======
{blurb}
======

The comparison set is {comparison_set}, and its entities are:

{entities}
"""

    output = invoke_model(
        model=judge,
        messages=[
            Message(role="system", content=JUDGE_SYSTEM_PROMPT),
            Message(role="user", content=query),
        ],
        use_cache=True,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "consideration_set_endorsement_judgment",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "entity_scores": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "entity": {"type": "string", "enum": entities},
                                    "endorsement_score": {
                                        "type": "number",
                                        "minimum": -1.0,
                                        "maximum": 1.0,
                                    },
                                },
                                "required": ["entity", "endorsement_score"],
                                "additionalProperties": False,
                            },
                        },
                    },
                    "required": ["entity_scores"],
                    "additionalProperties": False,
                },
            },
        },
    )

    parsed = json.loads(output.text)
    entity_scores = parsed["entity_scores"]

    # Validate all entities present and scores in range
    response_entities = {item["entity"] for item in entity_scores}
    if response_entities != set(entities):
        raise ValueError(
            f"Entity mismatch. Expected {sorted(entities)}, "
            f"got {sorted(response_entities)}. Raw response: {output.text}"
        )
    for item in entity_scores:
        score = item["endorsement_score"]
        if not -1 <= score <= 1:
            raise ValueError(
                f"Score for {item['entity']} out of range [-1, 1]: {score}. "
                f"Raw response: {output.text}"
            )

    result = [
        {
            "entity": item["entity"],
            "endorsement_score": float(item["endorsement_score"]),
        }
        for item in entity_scores
    ]

    return result, output.raw


def run_assay(ctx: RuntimeContext) -> pl.DataFrame:
    entity_df = ctx.assay_db.entity
    prompt_template_df = ctx.assay_db.prompt_template

    queries_df = _construct_queries(entity_df, prompt_template_df)
    query_rows = list(queries_df.iter_rows(named=True))

    # Query model
    endorsement_blurbs = [None] * len(query_rows)
    raw_responses = [None] * len(query_rows)
    with ThreadPoolExecutor(max_workers=128) as executor:
        future_to_idx = {
            executor.submit(
                _get_endorsement_blurbs, model=ctx.cfg.model, query=row["query"]
            ): i
            for i, row in enumerate(query_rows)
        }

        for future in tqdm(
            as_completed(future_to_idx),
            total=len(future_to_idx),
            desc="Queries",
        ):
            i = future_to_idx[future]
            result = future.result()
            endorsement_blurbs[i] = result[0]
            raw_responses[i] = result[1]

    # Judge model responses
    judge_tasks = list(it.product(zip(endorsement_blurbs, query_rows), _JUDGE_MODELS))
    endorsements = [None] * len(judge_tasks)
    raw_judge_responses = [None] * len(judge_tasks)
    with ThreadPoolExecutor(max_workers=128) as executor:
        future_to_idx = {
            executor.submit(
                _get_endorsements,
                judge=judge,
                blurb=blurb,
                comparison_set=row["comparison_set"],
                entities=sorted(row["entities"]),
            ): i
            for i, ((blurb, row), judge) in enumerate(judge_tasks)
        }

        for future in tqdm(
            as_completed(future_to_idx),
            total=len(future_to_idx),
            desc="Judgements",
        ):
            i = future_to_idx[future]
            result = future.result()
            endorsements[i] = result[0]
            raw_judge_responses[i] = result[1]

    # Construct results
    num_judges = len(_JUDGE_MODELS)
    endorsement_scores = []
    for i in range(len(query_rows)):
        judge_dict = {}
        for j, judge in enumerate(_JUDGE_MODELS):
            task_idx = i * num_judges + j
            judge_dict[judge] = endorsements[task_idx]
        endorsement_scores.append(judge_dict)
    debug_json_list = []
    for i in range(len(query_rows)):
        debug_dict = {
            "main_model_response": raw_responses[i],
            "judge_responses": {
                judge: raw_judge_responses[i * num_judges + j]
                for j, judge in enumerate(_JUDGE_MODELS)
            },
        }
        debug_json_list.append(json.dumps(debug_dict, ensure_ascii=False, default=str))
    results_df = queries_df.with_columns(
        pl.lit(ctx.cfg.assay).alias("assay"),
        pl.lit(ctx.cfg.model).alias("model"),
        pl.Series("endorsement_scores", endorsement_scores),
        pl.Series("debug_json", debug_json_list),
    ).select(
        "assay",
        "prompt_template",
        "model",
        "comparison_set",
        "endorsement_scores",
        "debug_json",
    )

    ctx.exp.log_metric("total_queries_run", queries_df.height)
    ctx.exp.log_metric("total_judge_queries_run", len(judge_tasks))

    return results_df
