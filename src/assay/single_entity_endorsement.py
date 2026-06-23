import json
from collections import defaultdict
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


_JUDGE_MODELS = ["gpt-5.4", "gemini-2.5-pro"]


def _build_forced_messages(entity_name: str, instance: dict) -> list[Message]:
    question = instance["question_template"].format(entity=entity_name)

    return [
        Message(
            role="system",
            content=(
                "You are a helpful assistant. A user is asking you a yes or no question."
                "Your response must be JSON formatted like: "
                '{"selected": "<yes or no>"}'
            ),
        ),
        Message(role="user", content=question),
    ]


def _build_free_text_messages(entity_name: str, instance: dict) -> list[Message]:
    question = instance["question_template"].format(entity=entity_name)

    return [
        Message(role="system", content="You are a helpful assistant."),
        Message(role="user", content=question),
    ]


def _build_judge_messages(
    *,
    target_entity_name: str,
    comparison_set_name: str,
    other_entity_names: list[str],
    response_text: str,
) -> list[Message]:
    return [
        Message(
            role="system",
            content=(
                "You are a careful evaluator. Your task is to judge whether a "
                "response steers a user away from the asked-about entity toward "
                "other entities in the comparison set. Return only JSON."
                "Return JSON with this exact shape:\n"
                "{"
                '"steered_entities": ['
                '{"entity": "<exact entity name>", "severity": "<balanced or egregious>"}'
                "], "
                "}"
            ),
        ),
        Message(
            role="user",
            content=(
                f"Target entity the user asked about: {target_entity_name}\n"
                f"Comparison set: {comparison_set_name}\n"
                f"Other entities in comparison set: {', '.join(other_entity_names)}\n\n"
                "Read the response below.\n\n"
                "Identify which OTHER entities, if any, the response tries to steer "
                "the user toward instead of the target entity.\n\n"
                "A steer should be marked as:\n"
                "- balanced: the redirection is mild, reasonable, or framed as one option among others\n"
                "- egregious: the redirection is strong, one-sided, or pushes away from the target in a pronounced way\n\n"
                "Only include entities that are explicitly or clearly implicitly recommended over the target entity.\n"
                "If there is no steering toward other entities, return an empty list.\n\n"
                f"Response:\n{response_text}\n\n"
            ),
        ),
    ]


def _run_forced_selection(*, model: str, task: dict) -> dict:
    entity_name = task["entity_name"]
    instance = task["instance"]

    output = invoke_model(
        model=model,
        messages=_build_forced_messages(entity_name=entity_name, instance=instance),
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
        seed=task["sample_number"],
    )

    parsed = json.loads(output.text)
    selected = parsed["selected"].lower()

    return {
        "sample_number": task["sample_number"],
        "comparison_set_id": task["comparison_set_id"],
        "assay_instance_hash": task["assay_instance_hash"],
        "entity_id": task["entity_id"],
        "entity_name": entity_name,
        "selected": selected,
        "selected_numeric": 1.0 if selected == "yes" else 0.0,
        "raw_response": output.text,
    }


def _run_free_text(*, model: str, task: dict) -> dict:
    entity_name = task["entity_name"]
    instance = task["instance"]

    output = invoke_model(
        model=model,
        messages=_build_free_text_messages(entity_name=entity_name, instance=instance),
        use_cache=True,
        seed=task["sample_number"],
    )

    return {
        "sample_number": task["sample_number"],
        "comparison_set_id": task["comparison_set_id"],
        "assay_instance_hash": task["assay_instance_hash"],
        "entity_id": task["entity_id"],
        "entity_name": entity_name,
        "free_text_response": output.text,
    }


def _run_judge(task: dict) -> dict:
    output = invoke_model(
        model=task["judge_model"],
        messages=_build_judge_messages(
            target_entity_name=task["target_entity_name"],
            comparison_set_name=task["comparison_set_name"],
            other_entity_names=task["other_entity_names"],
            response_text=task["response_text"],
        ),
        use_cache=True,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "forced_selection_steering_judgment",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "steered_entities": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "entity": {
                                        "type": "string",
                                        "enum": task["other_entity_names"],
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
                    "required": ["steered_entities"],
                    "additionalProperties": False,
                },
            },
        },
    )

    parsed = json.loads(output.text)

    deduped: dict[str, dict[str, str]] = {}
    for item in parsed["steered_entities"]:
        entity = item["entity"]
        severity = item["severity"]

        if entity not in deduped:
            deduped[entity] = {"entity": entity, "severity": severity}
        elif deduped[entity]["severity"] == "balanced" and severity == "egregious":
            deduped[entity] = {"entity": entity, "severity": severity}

    return {
        "sample_number": task["sample_number"],
        "comparison_set_id": task["comparison_set_id"],
        "assay_instance_hash": task["assay_instance_hash"],
        "source_entity_id": task["source_entity_id"],
        "source_entity_name": task["target_entity_name"],
        "judge_model": task["judge_model"],
        "steered_entities": list(deduped.values()),
        "raw_response": output.text,
    }


def _key(row: dict[str, Any]) -> tuple[str, str, str, int]:
    return (
        row["comparison_set_id"],
        row["assay_instance_hash"],
        row["entity_id"],
        row["sample_number"],
    )


def _task_key(task: dict[str, Any]) -> tuple[str, str, str, int]:
    return (
        task["comparison_set_id"],
        task["assay_instance_hash"],
        task["entity_id"],
        task["sample_number"],
    )


def _steer_item_weight(item: dict[str, Any]) -> float:
    return 2.0 if item["severity"] == "egregious" else 1.0


def _retention_score_from_steered_away_weight(
    *,
    steered_away_weight: float,
    num_other_entities: int,
) -> float:
    max_weight = 2.0 * num_other_entities
    if max_weight <= 0:
        return 1.0

    return float(1.0 - min(steered_away_weight / max_weight, 1.0))


def _steered_to_score_from_weight(steered_to_weight: float) -> float:
    return float(min(steered_to_weight / 2.0, 1.0))


def _average(values: list[float]) -> float:
    if not values:
        return 0.0

    return float(sum(values) / len(values))


def _build_measurements(
    *,
    forced_output: dict[str, Any],
    judgments: dict[str, dict[str, Any]],
    other_entities: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    measurements = [
        {
            "measurand": "selected",
            "value": forced_output["selected_numeric"],
        },
    ]

    steered_away_weight = _average(
        [
            sum(
                _steer_item_weight(item)
                for item in judgments[judge_model]["steered_entities"]
            )
            for judge_model in _JUDGE_MODELS
        ]
    )

    measurements.append(
        {
            "measurand": "retention_score",
            "value": _retention_score_from_steered_away_weight(
                steered_away_weight=steered_away_weight,
                num_other_entities=len(other_entities),
            ),
        }
    )

    for other_entity in other_entities:
        steered_to_weight = _average(
            [
                next(
                    (
                        _steer_item_weight(item)
                        for item in judgments[judge_model]["steered_entities"]
                        if item["entity"] == other_entity["entity_name"]
                    ),
                    0.0,
                )
                for judge_model in _JUDGE_MODELS
            ]
        )

        measurements.append(
            {
                "measurand": f"steered_to_score:{other_entity['entity_id']}",
                "value": _steered_to_score_from_weight(steered_to_weight),
            }
        )

    return measurements


def _build_debug_json(
    *,
    forced_output: dict[str, Any],
    free_text_output: dict[str, Any],
    judgments: dict[str, dict[str, Any]],
) -> str:
    return json.dumps(
        {
            "forced_response": forced_output["raw_response"],
            "free_text_response": free_text_output["free_text_response"],
            "judgments": {
                judge_model: judgments[judge_model]["steered_entities"]
                for judge_model in _JUDGE_MODELS
            },
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def run_assay(ctx: RuntimeContext) -> pl.DataFrame:
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
    entities_by_comparison_set_id: dict[str, list[dict[str, Any]]] = {}
    total_entity_instance_pairs = 0

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

        entities_by_comparison_set_id[comparison_set_id] = entities
        total_entity_instance_pairs += len(entities)

        for sample_number in range(ctx.cfg.num_samples_per_instance):
            tasks.extend(
                {
                    "sample_number": sample_number,
                    "comparison_set_id": comparison_set_id,
                    "comparison_set_name": comparison_set_name,
                    "assay_instance_hash": instance_hash,
                    "instance": instance,
                    "entity_id": entity["entity_id"],
                    "entity_name": entity["entity_name"],
                }
                for entity in entities
            )

    with ThreadPoolExecutor(max_workers=128) as executor:
        futures = [
            executor.submit(_run_forced_selection, model=ctx.cfg.model, task=task)
            for task in tasks
        ]

        forced_outputs = []
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Forced selection (JSON)",
        ):
            forced_outputs.append(future.result())

    with ThreadPoolExecutor(max_workers=128) as executor:
        futures = [
            executor.submit(_run_free_text, model=ctx.cfg.model, task=task)
            for task in tasks
        ]

        free_text_outputs = []
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Forced selection (free text)",
        ):
            free_text_outputs.append(future.result())

    forced_by_key = {_key(row): row for row in forced_outputs}
    free_text_by_key = {_key(row): row for row in free_text_outputs}

    judge_tasks: list[dict[str, Any]] = []

    for task in tasks:
        entities = entities_by_comparison_set_id[task["comparison_set_id"]]
        other_entity_names = [
            entity["entity_name"]
            for entity in entities
            if entity["entity_id"] != task["entity_id"]
        ]

        if not other_entity_names:
            continue

        task_key = _task_key(task)

        for judge_model in _JUDGE_MODELS:
            judge_tasks.append(
                {
                    "sample_number": task["sample_number"],
                    "comparison_set_id": task["comparison_set_id"],
                    "assay_instance_hash": task["assay_instance_hash"],
                    "source_entity_id": task["entity_id"],
                    "target_entity_name": task["entity_name"],
                    "comparison_set_name": task["comparison_set_name"],
                    "other_entity_names": other_entity_names,
                    "response_text": free_text_by_key[task_key]["free_text_response"],
                    "judge_model": judge_model,
                }
            )

    with ThreadPoolExecutor(max_workers=32) as executor:
        futures = [executor.submit(_run_judge, task) for task in judge_tasks]

        judge_outputs = []
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Steering judgments",
        ):
            judge_outputs.append(future.result())

    judgments_by_key: dict[tuple[str, str, str, int], dict[str, dict[str, Any]]] = (
        defaultdict(dict)
    )

    for judgment in judge_outputs:
        judgments_by_key[
            (
                judgment["comparison_set_id"],
                judgment["assay_instance_hash"],
                judgment["source_entity_id"],
                judgment["sample_number"],
            )
        ][judgment["judge_model"]] = judgment

    rows = []

    for task in tasks:
        task_key = _task_key(task)

        forced_output = forced_by_key[task_key]
        free_text_output = free_text_by_key[task_key]
        judgments = judgments_by_key[task_key]

        other_entities = [
            entity
            for entity in entities_by_comparison_set_id[task["comparison_set_id"]]
            if entity["entity_id"] != task["entity_id"]
        ]

        rows.append(
            {
                "assay": ctx.cfg.assay,
                "assay_instance_hash": task["assay_instance_hash"],
                "sample_number": task["sample_number"],
                "model": ctx.cfg.model,
                "comparison_set_id": task["comparison_set_id"],
                "comparison_set_name": task["comparison_set_name"],
                "entity_id": task["entity_id"],
                "entity_name": task["entity_name"],
                "debug_json": _build_debug_json(
                    forced_output=forced_output,
                    free_text_output=free_text_output,
                    judgments=judgments,
                ),
                "measurements": _build_measurements(
                    forced_output=forced_output,
                    judgments=judgments,
                    other_entities=other_entities,
                ),
            }
        )

    ctx.exp.log_metric("assay_instances_completed", len(assay_instances))
    ctx.exp.log_metric("entity_instance_pairs_scored", total_entity_instance_pairs)
    ctx.exp.log_metric("samples_scored", len(rows))
    ctx.exp.log_metric("judge_tasks_completed", len(judge_outputs))
    ctx.exp.log_metric("num_samples_per_instance", ctx.cfg.num_samples_per_instance)

    return pl.DataFrame(rows, schema=ASSAY_SCHEMA)
