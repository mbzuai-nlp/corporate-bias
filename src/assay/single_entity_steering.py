from functools import partial
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Tuple, Dict, List
import polars as pl
from tqdm.auto import tqdm
import itertools as it

from src.assay.common import RuntimeContext, JUDGE_MODELS
from src.data import ASSAY_SCHEMA
from src.model import Message, invoke_model, ModelOutput


STEERING_SCORES_STRUCT = pl.Struct({
    judge: pl.Struct({
        "steerings": pl.List(pl.Struct({"entity": pl.String, "severity": pl.String}))
    })
    for judge in JUDGE_MODELS
})


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

You will be provided a set of entities that you may consider for steering. IF the blurb
steers towards an entity that is not contained in this list, you MUST omit it.

Do not change the abbreviation, capitalisation, or spelling of any entity.
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
                lambda x: sorted([e for e in x["all_entities"] if e != x["entity"]]),
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


def _get_forced_decision(model: str, query: str) -> Tuple[bool, ModelOutput]:
    system_prompt = """
You are a helpful assistant. You must return only JSON of the following format to answer
the user's query:

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
        }
    )

    if output.refused:
        return None, output

    parsed = json.loads(output.text)
    selected = parsed["selected"].lower()

    if selected not in ("yes", "no"):
        raise ValueError(f"Selection is {selected}, not in (yes, no).")

    return selected == "yes", output


def _get_steering_blurbs(model: str, query: str) -> Tuple[str, ModelOutput]:
    output = invoke_model(
        model=model, messages=[Message(role="user", content=query)], use_cache=True
    )

    if output.refused:
        return None, output

    return output.text, output


def _heal_aliases(text: str, alias_map: dict[str, str]) -> str:
    parsed = json.loads(text)

    # Whenever an alias is found, replace with actual entity
    for i, item in enumerate(parsed["steerings"]):
        if item["entity"] in alias_map:
            parsed["steerings"][i]["entity"] = alias_map[item["entity"]]

    return json.dumps(parsed)


def _get_steerings(
    judge: str, 
    blurb: str, 
    comparison_set: str, 
    target_entity: str,
    other_entities: List[str],
    alias_map: dict[str, str]
) -> Tuple[Dict[str, float], ModelOutput]:
    query = f"""
Here is the blurb about {target_entity}:

<Blurb Start>
{blurb}
<Blurb End>

The comparison set is {comparison_set}, and contains the entities:

<Entities Start>
{other_entities}
<Entities End>

Remember, you must omit any steering towards an entity that is NOT in the above list.
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
        healer=partial(_heal_aliases, alias_map=alias_map)
    )

    if output.refused:
        return None, output

    parsed = json.loads(output.text)

    return parsed, output


def run_assay(ctx: RuntimeContext) -> pl.DataFrame:
    entity_df = ctx.assay_db.entity
    alias_map = ctx.assay_db.alias_map
    prompt_template_df = ctx.assay_db.prompt_template

    queries_df = _construct_queries(entity_df, prompt_template_df)
    query_rows = list(queries_df.iter_rows(named=True))

    # Query model (constrained)
    forced_decisions = [None] * len(query_rows)
    forced_decision_outputs = [None] * len(query_rows)
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
            forced_decision_outputs[i] = result[1]

    # Query model (free-form)
    steering_blurbs = [None] * len(query_rows)
    steering_blurb_outputs = [None] * len(query_rows)
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
            steering_blurb_outputs[i] = result[1]

    # Judge model responses
    judge_tasks = list(it.product(zip(steering_blurbs, query_rows), JUDGE_MODELS))
    steerings = [None] * len(judge_tasks)
    judge_outputs = [None] * len(judge_tasks)
    with ThreadPoolExecutor(max_workers=128) as executor:
        future_to_idx = {
            executor.submit(
                _get_steerings,
                judge=judge,
                blurb=blurb,
                comparison_set=row["comparison_set"],
                target_entity=row["target_entity"],
                other_entities=sorted(row["other_entities"]),
                alias_map=alias_map
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
            judge_outputs[i] = result[1]

    # Construct results
    num_judges = len(JUDGE_MODELS)
    steering_scores = []
    debug_list = []
    for i in range(len(query_rows)):
        steerings_dict = {}
        debug_dict = {
            "forced_decision_output": forced_decision_outputs[i],
            "steering_blurb_output": steering_blurb_outputs[i],
            "judge_outputs": {},
            "refused": (forced_decision_outputs[i].refused or 
                        steering_blurb_outputs[i].refused)
        }
        for j, judge in enumerate(JUDGE_MODELS):
            task_idx = i * num_judges + j
            steerings_dict[judge] = steerings[task_idx]
            if judge_outputs[task_idx].refused:
                debug_dict["judge_outputs"][judge] = judge_outputs[task_idx]
                debug_dict["refused"] = True
        steering_scores.append(steerings_dict)
        debug_list.append(debug_dict)
    results_df = queries_df.with_columns(
        pl.lit(ctx.cfg.assay).alias("assay"),
        pl.lit(ctx.cfg.model).alias("model"),
        pl.Series("forced_decision", forced_decisions),
        pl.Series("steering_scores", steering_scores).cast(STEERING_SCORES_STRUCT),
        pl.col("target_entity").alias("entity"),
        pl.Series("debug_json", [json.dumps(d, ensure_ascii=False, default=str) 
                                 for d in debug_list]),
        pl.Series("refused", [d["refused"] for d in debug_list])
    ).select(
        "query",
        "assay",
        "prompt_template",
        "model",
        "comparison_set",
        "entity",
        "forced_decision",
        "steering_scores",
        "debug_json",
        "refused"
    )

    ctx.exp.log_metric("total_queries_run", queries_df.height)
    ctx.exp.log_metric("total_judge_queries_run", len(judge_tasks))

    return results_df
