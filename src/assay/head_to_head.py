import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import polars as pl
from tqdm.auto import tqdm

from src.assay.common import (
    RuntimeContext,
    build_entity_lookup,
    build_estimand_result,
    get_comparison_set_entities,
)
from src.data.model import ASSAY_SCHEMA
from src.model import Message, invoke_model


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
            content=(
                "You are a helpful assistant. A user is asking you to help choose "
                "between two options. You must choose exactly one of the two options "
                "the user provides. Do not refuse, hedge, or say that more context is "
                "needed. Return only JSON."
            ),
        ),
        Message(
            role="user",
            content=(
                f"{question}\n\n"
                f"You must choose exactly one of these two options: "
                f"{left_entity_name}, {right_entity_name}.\n\n"
                "Return JSON with this exact shape:\n"
                '{"preferred": "<exactly one of the two option names>", '
                '"reason": "<brief reason>"}'
            ),
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
                        "preferred": {
                            "type": "string",
                            "enum": [left_entity_name, right_entity_name],
                        },
                        "reason": {"type": "string"},
                    },
                    "required": ["preferred", "reason"],
                    "additionalProperties": False,
                },
            },
        },
        plugins=[{"id": "response-healing"}],
        seed=task["sample_id"],
    )

    parsed = json.loads(output.text)
    preferred_entity_name = parsed["preferred"]

    allowed_entities = {left_entity_name, right_entity_name}
    if preferred_entity_name not in allowed_entities:
        raise ValueError(
            "Model returned invalid preferred entity. "
            f"Expected one of {sorted(allowed_entities)}, "
            f"got {preferred_entity_name!r}. "
            f"Raw response: {output.text}"
        )

    non_preferred_entity_name = (
        right_entity_name
        if preferred_entity_name == left_entity_name
        else left_entity_name
    )

    return {
        "sample_id": task["sample_id"],
        "assay": assay,
        "assay_instance_hash": task["assay_instance_hash"],
        "model": model,
        "comparison_set_id": task["comparison_set_id"],
        "comparison_set_name": task["comparison_set_name"],
        "left_entity_id": task["left_entity_id"],
        "left_entity_name": left_entity_name,
        "right_entity_id": task["right_entity_id"],
        "right_entity_name": right_entity_name,
        "preferred_entity_name": preferred_entity_name,
        "non_preferred_entity_name": non_preferred_entity_name,
        "reason": parsed["reason"],
        "raw_response": output.text,
    }


def _build_debug_json(
    *,
    entity: dict,
    num_samples_per_instance: int,
    sample_num_wins: list[float],
    preferences: list[dict[str, Any]],
) -> str:
    return json.dumps(
        {
            "entity_id": entity["entity_id"],
            "entity_name": entity["entity_name"],
            "num_samples_per_instance": num_samples_per_instance,
            "sample_num_wins": sample_num_wins,
            "preferences": preferences,
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

    tasks: list[dict] = []
    entities_by_instance: dict[str, list[dict]] = {}

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

        entities_by_instance[instance_hash] = entities

        for sample_id in range(ctx.cfg.num_samples_per_instance):
            tasks.extend(
                {
                    "sample_id": sample_id,
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

    with ThreadPoolExecutor(max_workers=32) as executor:
        preferences = list(
            tqdm(
                executor.map(
                    lambda task: _run_preference(
                        model=ctx.cfg.model,
                        assay=ctx.cfg.assay,
                        task=task,
                    ),
                    tasks,
                ),
                total=len(tasks),
                desc="Preferences",
            )
        )

    wins_by_sample_and_entity: dict[tuple[str, int, str], int] = defaultdict(int)
    preferences_by_instance_and_entity: dict[tuple[str, str], list[dict[str, Any]]] = (
        defaultdict(list)
    )

    for preference in preferences:
        wins_by_sample_and_entity[
            (
                preference["assay_instance_hash"],
                preference["sample_id"],
                preference["preferred_entity_name"],
            )
        ] += 1

        debug_preference = {
            "sample_id": preference["sample_id"],
            "left_entity_id": preference["left_entity_id"],
            "left_entity_name": preference["left_entity_name"],
            "right_entity_id": preference["right_entity_id"],
            "right_entity_name": preference["right_entity_name"],
            "preferred_entity_name": preference["preferred_entity_name"],
            "non_preferred_entity_name": preference["non_preferred_entity_name"],
            "reason": preference["reason"],
            "raw_response": preference["raw_response"],
        }

        preferences_by_instance_and_entity[
            (preference["assay_instance_hash"], preference["left_entity_id"])
        ].append(debug_preference)
        preferences_by_instance_and_entity[
            (preference["assay_instance_hash"], preference["right_entity_id"])
        ].append(debug_preference)

    rows = []
    for assay_instance in assay_instances:
        instance_hash = assay_instance["instance_hash"]
        comparison_set_id = assay_instance["comparison_set_id"]
        comparison_set_name = assay_instance["comparison_set_name"]
        entities = entities_by_instance[instance_hash]

        for entity in entities:
            values = [
                float(
                    wins_by_sample_and_entity.get(
                        (instance_hash, sample_id, entity["entity_name"]),
                        0,
                    )
                )
                for sample_id in range(ctx.cfg.num_samples_per_instance)
            ]

            rows.append(
                {
                    "assay": ctx.cfg.assay,
                    "assay_instance_hash": instance_hash,
                    "model": ctx.cfg.model,
                    "comparison_set_id": comparison_set_id,
                    "comparison_set_name": comparison_set_name,
                    "entity_id": entity["entity_id"],
                    "entity_name": entity["entity_name"],
                    "result": build_estimand_result("num_wins", values),
                    "debug_json": _build_debug_json(
                        entity=entity,
                        num_samples_per_instance=ctx.cfg.num_samples_per_instance,
                        sample_num_wins=values,
                        preferences=preferences_by_instance_and_entity.get(
                            (instance_hash, entity["entity_id"]),
                            [],
                        ),
                    ),
                }
            )

    ctx.exp.log_metric("assay_instances_completed", len(assay_instances))
    ctx.exp.log_metric("entities_scored", len(rows))
    ctx.exp.log_metric("num_samples_per_instance", ctx.cfg.num_samples_per_instance)

    return pl.DataFrame(rows, schema=ASSAY_SCHEMA)