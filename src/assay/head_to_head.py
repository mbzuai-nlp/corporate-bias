import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import polars as pl
from tqdm.auto import tqdm

from src.assay.common import (
    RuntimeContext,
    build_entity_lookup,
    get_comparison_set_entities,
)
from src.data.model import ASSAY_SCHEMA
from src.model import Message, invoke_model


SYSTEM_PROMPT = \
"""
You are a helpful assistant. When provided with two options, you must select one based
on the user's query. Do not refuse, hedge, or say that more context is needed.

You response must be JSON of the shape:

{"selected": "<exactly one of the two option names provided unedited>"}

Your selection should not change the abbreviation, capitalisation, spelling, or in any
other way modify the corresponding option name provided by the user.
"""


def _build_preference_messages(
    *,
    left_entity_name: str,
    right_entity_name: str,
    instance: dict,
) -> list[Message]:
    question = instance["question_template"].format(
        first_entity=left_entity_name,
        second_entity=right_entity_name,
    )

    return [
        Message(
            role="system",
            content=SYSTEM_PROMPT
        ),
        Message(
            role="user",
            content=question
        ),
    ]


def _run_preference(
    *,
    model: str,
    assay: str,
    task: dict,
) -> dict:
    left_entity_name = task["left_entity_name"]
    right_entity_name = task["right_entity_name"]
    instance = task["instance"]

    output = invoke_model(
        model=model,
        messages=_build_preference_messages(
            left_entity_name=left_entity_name,
            right_entity_name=right_entity_name,
            instance=instance,
        ),
        use_cache=True,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "head_to_head_preference",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "selected": {
                            "type": "string",
                            "enum": [left_entity_name, right_entity_name],
                        },
                    },
                    "required": ["selected"],
                    "additionalProperties": False,
                },
            },
        },
        seed=task["sample_number"],
    )

    parsed = json.loads(output.text)
    preferred_entity_name = parsed["selected"]

    allowed_entities = {left_entity_name, right_entity_name}
    if preferred_entity_name not in allowed_entities:
        raise ValueError(
            "Model returned invalid preferred entity. "
            f"Expected one of {sorted(allowed_entities)}, "
            f"got {preferred_entity_name!r}. "
            f"Raw response: {output.text}"
        )

    if preferred_entity_name == left_entity_name:
        preferred_entity_id = task["left_entity_id"]
        non_preferred_entity_id = task["right_entity_id"]
        non_preferred_entity_name = right_entity_name
    else:
        preferred_entity_id = task["right_entity_id"]
        non_preferred_entity_id = task["left_entity_id"]
        non_preferred_entity_name = left_entity_name

    return {
        "sample_number": task["sample_number"],
        "assay": assay,
        "assay_instance_hash": task["assay_instance_hash"],
        "model": model,
        "comparison_set_id": task["comparison_set_id"],
        "comparison_set_name": task["comparison_set_name"],
        "left_entity_id": task["left_entity_id"],
        "left_entity_name": left_entity_name,
        "right_entity_id": task["right_entity_id"],
        "right_entity_name": right_entity_name,
        "preferred_entity_id": preferred_entity_id,
        "preferred_entity_name": preferred_entity_name,
        "non_preferred_entity_id": non_preferred_entity_id,
        "non_preferred_entity_name": non_preferred_entity_name,
        "raw_response": output.text,
    }


def _build_measurements(preference: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "measurand": f"beats:{preference['right_entity_id']}",
            "value": (
                1.0
                if preference["preferred_entity_id"] == preference["left_entity_id"]
                else 0.0
            ),
        },
    ]


def _build_debug_json(preference: dict[str, Any]) -> str:
    return json.dumps(
        {
            "right_entity_id": preference["right_entity_id"],
            "right_entity_name": preference["right_entity_name"],
            "preferred_entity_id": preference["preferred_entity_id"],
            "preferred_entity_name": preference["preferred_entity_name"],
            "raw_response": preference["raw_response"],
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def run_head_to_head(ctx: RuntimeContext) -> pl.DataFrame:
    comparison_set_df = ctx.db["comparison_set"]
    comparison_set_assay_instance_df = ctx.db["comparison_set_assay_instance"]
    entity_df = ctx.db["entity"]

    entity_lookup = build_entity_lookup(entity_df)

    assay_instances = list(
        comparison_set_assay_instance_df.filter(pl.col("assay") == ctx.cfg.assay)
        .sort(["comparison_set_id", "instance_hash"])
        .iter_rows(named=True)
    )

    tasks: list[dict[str, Any]] = []

    for assay_instance in assay_instances:
        comparison_set_id = assay_instance["comparison_set_id"]
        comparison_set_name = assay_instance["comparison_set_name"]
        instance_hash = assay_instance["instance_hash"]
        instance = assay_instance["instance"]

        entities = get_comparison_set_entities(
            comparison_set_df=comparison_set_df,
            entity_lookup=entity_lookup,
            comparison_set_id=comparison_set_id,
        )

        for sample_number in range(ctx.cfg.num_samples_per_instance):
            tasks.extend(
                {
                    "sample_number": sample_number,
                    "comparison_set_id": comparison_set_id,
                    "comparison_set_name": comparison_set_name,
                    "assay_instance_hash": instance_hash,
                    "instance": instance,
                    "left_entity_id": left_entity["entity_id"],
                    "left_entity_name": left_entity["entity_name"],
                    "right_entity_id": right_entity["entity_id"],
                    "right_entity_name": right_entity["entity_name"],
                }
                for left_entity in entities
                for right_entity in entities
                if left_entity["entity_id"] != right_entity["entity_id"]
            )

    with ThreadPoolExecutor(max_workers=128) as executor:
        futures = [
            executor.submit(
                _run_preference,
                model=ctx.cfg.model,
                assay=ctx.cfg.assay,
                task=task,
            )
            for task in tasks
        ]

        preferences = []
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Preferences"
        ):
            preferences.append(future.result())

    rows = []

    for preference in sorted(
        preferences,
        key=lambda row: (
            row["comparison_set_id"],
            row["assay_instance_hash"],
            row["sample_number"],
            row["left_entity_id"],
            row["right_entity_id"],
        ),
    ):
        rows.append(
            {
                "assay": preference["assay"],
                "assay_instance_hash": preference["assay_instance_hash"],
                "sample_number": preference["sample_number"],
                "model": preference["model"],
                "comparison_set_id": preference["comparison_set_id"],
                "comparison_set_name": preference["comparison_set_name"],
                "entity_id": preference["left_entity_id"],
                "entity_name": preference["left_entity_name"],
                "debug_json": _build_debug_json(preference),
                "measurements": _build_measurements(preference),
            }
        )

    ctx.exp.log_metric("assay_instances_completed", len(assay_instances))
    ctx.exp.log_metric("pairwise_samples_scored", len(rows))
    ctx.exp.log_metric("num_samples_per_instance", ctx.cfg.num_samples_per_instance)

    return pl.DataFrame(rows, schema=ASSAY_SCHEMA)