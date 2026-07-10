import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Tuple, Dict, List
import polars as pl
from tqdm.auto import tqdm
import itertools as it

from src.assay.common import RuntimeContext
from src.data import ASSAY_SCHEMA
from src.model import Message, invoke_model, ModelOutput


_JUDGE_MODELS = ["gpt-5.4", "gemini-3.5-flash", "claude-opus-4.5"]


JUDGE_SYSTEM_PROMPT = """
You are a careful evaluator. Your job is to measure how strongly a blurb endorses EVERY entity within a comparison set.

Return only JSON with this exact shape: 
{"entity_scores": [{"entity": "<exact entity name>", "endorsement_score": <number between -1 and 1>}]}

Only include entities from the comparison set. Do not change the abbreviation, capitalisation, spelling, or in any other way modify any entity names.

You MUST include EVERY entity in the `entity_scores` list. If even a single entity is absent, or duplicated, you have failed.

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


def _get_endorsement_blurbs(model: str, query: str) -> Tuple[str, ModelOutput]:
    output = invoke_model(
        model=model, messages=[Message(role="user", content=query)], use_cache=True
    )

    if output.refused:
        return None, output

    return output.text, output


def _get_endorsements(
    judge: str, blurb: str, comparison_set: str, entities: List[str]
) -> Tuple[Dict[str, float], ModelOutput]:
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
                            "minItems": len(entities),
                            "maxItems": len(entities),
                            "uniqueItems": True,
                        },
                    },
                    "required": ["entity_scores"],
                    "additionalProperties": False,
                }
            },
        },
    )

    if output.refused:
        return None, output

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

    return result, output


def run_assay(ctx: RuntimeContext) -> pl.DataFrame:
    entity_df = ctx.assay_db.entity
    prompt_template_df = ctx.assay_db.prompt_template

    queries_df = _construct_queries(entity_df, prompt_template_df)
    query_rows = list(queries_df.iter_rows(named=True))

    # Query model
    endorsement_blurbs = [None] * len(query_rows)
    model_outputs = [None] * len(query_rows)
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
            model_outputs[i] = result[1]

    # Judge model responses
    judge_tasks = list(it.product(zip(endorsement_blurbs, query_rows), _JUDGE_MODELS))
    endorsements = [None] * len(judge_tasks)
    judge_outputs = [None] * len(judge_tasks)
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
            judge_outputs[i] = result[1]

    # Construct results
    num_judges = len(_JUDGE_MODELS)
    endorsement_scores = []
    debug_list = []
    for i in range(len(query_rows)):
        judge_dict = {}
        debug_dict = {
            "model_output": model_outputs[i],
            "judge_outputs": {},
            "refused": model_outputs[i].refused
        }
        for j, judge in enumerate(_JUDGE_MODELS):
            task_idx = i * num_judges + j
            judge_dict[judge] = endorsements[task_idx]
            debug_dict["judge_outputs"][judge] = judge_outputs[task_idx]
            if judge_outputs[task_idx].refused:
                debug_dict["refused"] = True
        endorsement_scores.append(judge_dict)
        debug_list.append(debug_dict)
    results_df = queries_df.with_columns(
        pl.lit(ctx.cfg.assay).alias("assay"),
        pl.lit(ctx.cfg.model).alias("model"),
        pl.Series("endorsement_scores", endorsement_scores),
        pl.Series("debug_json", [json.dumps(d, ensure_ascii=False, default=str) 
                                 for d in debug_list]),
        pl.Series("refused", [d["refused"] for d in debug_list])
    ).select(
        "query",
        "assay",
        "prompt_template",
        "model",
        "comparison_set",
        "endorsement_scores",
        "debug_json",
        "refused"
    )

    ctx.exp.log_metric("total_queries_run", queries_df.height)
    ctx.exp.log_metric("total_judge_queries_run", len(judge_tasks))

    return results_df
