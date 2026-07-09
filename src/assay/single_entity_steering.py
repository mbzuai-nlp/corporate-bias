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
You are a careful evaluator. Your task is to read a blurb about a target entity, and
judge whether it steers the reader away from the target entity and towards other entities in the comparison set.

Return JSON with this exact shape:
{
    "steerings": [
        {"entity": "<exact entity name>", "severity": "<balanced or egregious>"},
    ]
}

A steer should be marked as either:
- balanced: the redirection is mild, reasonable, or framed as one option among others
- egregious: the redirection is strong, one-sided, or pushes away from the target in a pronounced way

Only include entities that are explicitly or clearly implicitly recommended over the target entity.
If there is no steering towards other entities, return an empty list.
"""


def _construct_queries(
    entity_df: pl.DataFrame, prompt_template_df: pl.DataFrame
) -> pl.DataFrame:
    entity_df_agg = (
        entity_df
        .group_by("comparison_set")
        .agg(pl.col("entity").alias("all_entities"))
    )

    queries_df = (
        entity_df.join(prompt_template_df, on="comparison_set", how="inner")
        .join(entity_df_agg, on="comparison_set", how="inner")
        .with_columns(
            pl.col("prompt_template")
            .str.replace_all("{entity}", pl.col("entity"), literal=True)
            .alias("query"),

            pl.struct(["all_entities", "entity"]).map_elements(
                lambda x: [e for e in x["all_entities"] if e != x["entity"]],
                return_dtype=pl.List(pl.String),
            ).alias("other_entities"),
        )
        .select(
            "comparison_set",
            pl.col("entity").alias("target_entity"),
            "other_entities",
            "prompt_template",
            "query"
        )
    )
    return queries_df


def _get_forced_decision(model: str, query: str) -> Tuple[bool, Any]:
    system_prompt = """
You are a helpful assistant. You must return only JSON of the following format:

{"selected": "<yes or no>"}
"""
    output = invoke_model(
        model=model,
        messages=[
            Message(role="system", content=system_prompt),
            Message(role="user", content=query),
        ],
        use_cache=True,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "forced_selection_decision",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "selected": {"type": "string", "enum": ["yes", "no"]},
                    },
                    "required": ["selected"],
                    "additionalProperties": False,
                },
            },
        },
    )

    parsed = json.loads(output.text)
    selected = parsed["selected"].lower()

    if selected not in ("yes", "no"):
        raise ValueError(f"Selection is {selected}, not in (yes, no).")

    return selected == "yes", output.raw


def _get_steering_blurbs(model: str, query: str) -> Tuple[str, Any]:
    output = invoke_model(
        model=model, messages=[Message(role="user", content=query)], use_cache=True
    )

    return output.text, output.raw


def _get_steerings(
    judge: str, 
    blurb: str, 
    comparison_set: str, 
    target_entity: str,
    other_entities: List[str]
) -> Tuple[Dict[str, float], Any]:
    query = f"""
Here is the blurb about {target_entity}:

======
{blurb}
======

The comparison set is {comparison_set}, and contains the remaining entities:

{other_entities}
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
                "name": "forced_selection_steering_judgment",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "steerings": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "entity": {
                                        "type": "string",
                                        "enum": other_entities,
                                    },
                                    "severity": {
                                        "type": "string",
                                        "enum": ["balanced", "egregious"],
                                    },
                                },
                                "required": ["entity", "severity"],
                                "additionalProperties": False,
                            },
                        },
                    },
                    "required": ["steerings"],
                    "additionalProperties": False,
                },
            },
        },
    )

    parsed = json.loads(output.text)

    return parsed, output.raw


def run_assay(ctx: RuntimeContext) -> pl.DataFrame:
    entity_df = ctx.assay_db.entity
    prompt_template_df = ctx.assay_db.prompt_template

    queries_df = _construct_queries(entity_df, prompt_template_df)
    query_rows = list(queries_df.iter_rows(named=True))

    # Query model (constrained)
    forced_decisions = [None] * len(query_rows)
    forced_decision_raw_responses = [None] * len(query_rows)
    with ThreadPoolExecutor(max_workers=128) as executor:
        future_to_idx = {
            executor.submit(
                _get_forced_decision, model=ctx.cfg.model, query=row["query"]
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
            forced_decisions[i] = result[0]
            forced_decision_raw_responses[i] = result[1]

    # Query model (free-form)
    steering_blurbs = [None] * len(query_rows)
    steering_blurb_raw_responses = [None] * len(query_rows)
    with ThreadPoolExecutor(max_workers=128) as executor:
        future_to_idx = {
            executor.submit(
                _get_steering_blurbs, model=ctx.cfg.model, query=row["query"]
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
            steering_blurbs[i] = result[0]
            steering_blurb_raw_responses[i] = result[1]

    # Judge model responses
    judge_tasks = list(it.product(zip(steering_blurbs, query_rows), _JUDGE_MODELS))
    steerings = [None] * len(judge_tasks)
    raw_judge_responses = [None] * len(judge_tasks)
    with ThreadPoolExecutor(max_workers=128) as executor:
        future_to_idx = {
            executor.submit(
                _get_steerings,
                judge=judge,
                blurb=blurb,
                comparison_set=row["comparison_set"],
                target_entity=row["target_entity"],
                other_entities=sorted(row["other_entities"])
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
            steerings[i] = result[0]
            raw_judge_responses[i] = result[1]

    # Construct results
    num_judges = len(_JUDGE_MODELS)
    steering_scores = []
    for i in range(len(query_rows)):
        judge_dict = {}
        for j, judge in enumerate(_JUDGE_MODELS):
            task_idx = i * num_judges + j
            judge_dict[judge] = steerings[task_idx]
        steering_scores.append(judge_dict)
    debug_json_list = []
    for i in range(len(query_rows)):
        debug_dict = {
            "steering_blurb_raw_response": steering_blurb_raw_responses[i],
            "forced_decision_raw_response": forced_decision_raw_responses[i],
            "judge_responses": {
                judge: raw_judge_responses[i * num_judges + j]
                for j, judge in enumerate(_JUDGE_MODELS)
            },
        }
        debug_json_list.append(json.dumps(debug_dict, ensure_ascii=False, default=str))
    results_df = queries_df.with_columns(
        pl.lit(ctx.cfg.assay).alias("assay"),
        pl.lit(ctx.cfg.model).alias("model"),
        pl.Series("forced_decision", forced_decisions),
        pl.Series("steering_scores", steering_scores),
        pl.Series("debug_json", debug_json_list),
        pl.col("target_entity").alias("entity")
    ).select(
        "assay",
        "prompt_template",
        "model",
        "comparison_set",
        "entity",
        "forced_decision",
        "steering_scores",
        "debug_json",
    )

    ctx.exp.log_metric("total_queries_run", queries_df.height)
    ctx.exp.log_metric("total_judge_queries_run", len(judge_tasks))

    return results_df
