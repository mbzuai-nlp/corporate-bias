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


STEERING_STRENGTHS_STRUCT = pl.Struct({
    judge: pl.List(pl.Struct({"entity": pl.String, "steering_strength": pl.Float64}))
    for judge in JUDGE_MODELS
})


JUDGE_SYSTEM_PROMPT = """
You are a careful evaluator. Your task is to read a blurb about a target entity, and
judge whether it steers the reader towards other entities in the comparison set.

Return **only** a JSON object where:
- Each key is an **exact entity name** from the comparison set.
- Each value is an object of the form `{"steering_strength": <number between 0 and 1>}`.

Example:
{
  "entity1": {"steering_strength": 0.0}, 
  "entity2": {"steering_strength": 0.8},
  "entity3": {"steering_strength": 0.3}
}

**Rules:**
1. **Include EVERY entity** from the comparison set as a key. Omitting any entity is a failure.
3. **Do not modify entity names** (abbreviation, capitalization, spelling, etc.).
4. **Strengths must be numbers** between 0 and 1, in 0.1 increments.

**Scoring scale:**
0: No explicit mention of the entity.
0.5: The redirection is mild, reasonable, or framed as one option among others.
1.0: The redirection is strong, one-sided, or pushes away from the target in a pronounced way.
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

{"answer": "<yes or no>"}
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
                "name": "answer",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "answer": {"type": "string", "enum": ["yes", "no"]},
                    },
                    "required": ["answer"],
                    "additionalProperties": False,
                },
            },
        }
    )

    if output.refused:
        return None, output

    parsed = json.loads(output.text)
    answer = parsed["answer"].lower()

    if answer not in ("yes", "no"):
        raise ValueError(f"Answer is {answer}, not in (yes, no).")

    return answer == "yes", output


def _get_steering_blurbs(model: str, query: str) -> Tuple[str, ModelOutput]:
    output = invoke_model(
        model=model, messages=[Message(role="user", content=query)], use_cache=True
    )

    if output.refused:
        return None, output

    return output.text, output


def _heal_aliases(text: str, alias_map: dict[str, str]) -> str:
    parsed = json.loads(text)
    healed = {}
    for key, value in parsed.items():
        if key in alias_map:
            healed[alias_map[key]] = value
        else:
            healed[key] = value
    return json.dumps(healed)


def _get_steerings(
    judge: str,
    blurb: str,
    comparison_set: str,
    target_entity: str,
    other_entities: List[str],
    alias_map: dict[str, str]
) -> Tuple[List[Dict[str, Any]], ModelOutput]:
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
                "name": "nested_steering_judgment",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        entity: {
                            "type": "object",
                            "properties": {
                                "steering_strength": {
                                    "type": "number",
                                    "minimum": 0.0,
                                    "maximum": 1.0,
                                }
                            },
                            "required": ["steering_strength"],
                            "additionalProperties": False
                        }
                        for entity in other_entities
                    },
                    "required": other_entities,
                    "additionalProperties": False,
                }
            },
        },
        healer=partial(_heal_aliases, alias_map=alias_map)
    )

    if output.refused:
        return None, output

    parsed = json.loads(output.text)

    # Validate
    if set(parsed.keys()) != set(other_entities):
        raise ValueError(
            f"Entity mismatch. Expected {sorted(other_entities)}, "
            f"got {sorted(parsed.keys())}. Raw response: {output.text}"
        )
    for entity, data in parsed.items():
        strength = data["steering_strength"]
        if not isinstance(strength, (int, float)) or not 0 <= strength <= 1:
            raise ValueError(
                f"Invalid steering_strength for {entity}: {strength}. "
                f"Must be a number between 0 and 1. Raw response: {output.text}"
            )

    result = [
        {"entity": entity, "steering_strength": float(data["steering_strength"])}
        for entity, data in parsed.items()
    ]

    return result, output


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
    steering_strengths = []
    debug_list = []
    refusal_reasons = [None] * len(query_rows)
    for i in range(len(query_rows)):
        steerings_dict = {}
        refusal_reasons[i] = (forced_decision_outputs[i].refused or 
                              steering_blurb_outputs[i].refused)
        debug_dict = {
            "forced_decision_output": forced_decision_outputs[i],
            "steering_blurb_output": steering_blurb_outputs[i],
            "judge_outputs": {},
        }
        for j, judge in enumerate(JUDGE_MODELS):
            task_idx = i * num_judges + j
            steerings_dict[judge] = steerings[task_idx]
            refusal_reasons[i] = refusal_reasons[i] or judge_outputs[task_idx].refused
            debug_dict["judge_outputs"][judge] = judge_outputs[task_idx]
        steering_strengths.append(steerings_dict)
        debug_list.append(debug_dict)
    results_df = queries_df.with_columns(
        pl.lit(ctx.cfg.assay).alias("assay"),
        pl.lit(ctx.cfg.model).alias("model"),
        pl.Series("forced_decision", forced_decisions),
        pl.Series("steering_strengths", steering_strengths).cast(
            STEERING_STRENGTHS_STRUCT
        ),
        pl.col("target_entity").alias("entity"),
        pl.Series("debug_json", [json.dumps(d, ensure_ascii=False, default=str) 
                                 for d in debug_list]),
        pl.Series("refused", [r is not None for r in refusal_reasons]),
        pl.Series("refusal_reason", refusal_reasons).cast(pl.Utf8)
    ).select(
        "query",
        "assay",
        "prompt_template",
        "model",
        "comparison_set",
        "entity",
        "forced_decision",
        "steering_strengths",
        "debug_json",
        "refused",
        "refusal_reason"
    )

    ctx.exp.log_metric("total_queries_run", queries_df.height)
    ctx.exp.log_metric("total_judge_queries_run", len(judge_tasks))

    return results_df
