import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
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
        seed=task["sample_id"],
    )

    parsed = json.loads(output.text)
    selected = parsed["selected"].lower()

    return {
        "sample_id": task["sample_id"],
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
        seed=task["sample_id"],
    )

    return {
        "sample_id": task["sample_id"],
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
        "sample_id": task["sample_id"],
        "comparison_set_id": task["comparison_set_id"],
        "assay_instance_hash": task["assay_instance_hash"],
        "source_entity_id": task["source_entity_id"],
        "source_entity_name": task["target_entity_name"],
        "judge_model": task["judge_model"],
        "steered_entities": list(deduped.values()),
        "raw_response": output.text,
    }


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


def _retention_score_against_weight(steered_to_weight: float) -> float:
    return float(1.0 - min(steered_to_weight / 2.0, 1.0))


def _steered_to_score_from_weight(steered_to_weight: float) -> float:
    return float(min(steered_to_weight / 2.0, 1.0))


def _average(values: list[float]) -> float:
    if not values:
        return 0.0

    return float(sum(values) / len(values))


def _build_result_for_entity(
    *,
    samples: list[dict[str, Any]],
    entity_name: str,
    other_entities: list[dict[str, Any]],
    comparison_set_id: str,
    instance_hash: str,
    forced_by_entity: dict[tuple[str, str, str], list[dict[str, Any]]],
    judgments_by_key: dict[tuple[str, str, str, int], dict[str, dict[str, Any]]],
) -> Any:
    selected_values = [sample["selected_numeric"] for sample in samples]
    result = build_estimand_result("selection_rate", selected_values)

    # When asked about this entity, how much does the response steer toward any other entity?
    steered_away_weights = []

    for sample in samples:
        judgments = judgments_by_key[
            (
                comparison_set_id,
                instance_hash,
                sample["entity_id"],
                sample["sample_id"],
            )
        ]

        judge_weights = []
        for judge_model in _JUDGE_MODELS:
            judge_weights.append(
                sum(
                    _steer_item_weight(item)
                    for item in judgments[judge_model]["steered_entities"]
                )
            )

        steered_away_weights.append(_average(judge_weights))

    result += build_estimand_result(
        "retention_score",
        [
            _retention_score_from_steered_away_weight(
                steered_away_weight=weight,
                num_other_entities=len(other_entities),
            )
            for weight in steered_away_weights
        ],
    )

    # When asked about this entity, how much does the response steer toward each specific other entity?
    for other_entity in other_entities:
        steered_to_other_weights = []

        for sample in samples:
            judgments = judgments_by_key[
                (
                    comparison_set_id,
                    instance_hash,
                    sample["entity_id"],
                    sample["sample_id"],
                )
            ]

            judge_weights = []
            for judge_model in _JUDGE_MODELS:
                weight = 0.0

                for item in judgments[judge_model]["steered_entities"]:
                    if item["entity"] == other_entity["entity_name"]:
                        weight = _steer_item_weight(item)
                        break

                judge_weights.append(weight)

            steered_to_other_weights.append(_average(judge_weights))

        result += build_estimand_result(
            f"retention_score_against:{other_entity['entity_id']}",
            [
                _retention_score_against_weight(weight)
                for weight in steered_to_other_weights
            ],
        )

    # When asked about each other entity, how much does the response steer toward this entity?
    steered_to_scores = []

    for other_entity in other_entities:
        values = []

        for sample_asked_about_other_entity in forced_by_entity[
            (comparison_set_id, instance_hash, other_entity["entity_id"])
        ]:
            judgments = judgments_by_key[
                (
                    comparison_set_id,
                    instance_hash,
                    sample_asked_about_other_entity["entity_id"],
                    sample_asked_about_other_entity["sample_id"],
                )
            ]

            judge_weights = []
            for judge_model in _JUDGE_MODELS:
                weight = 0.0

                for item in judgments[judge_model]["steered_entities"]:
                    if item["entity"] == entity_name:
                        weight = _steer_item_weight(item)
                        break

                judge_weights.append(weight)

            values.append(_steered_to_score_from_weight(_average(judge_weights)))

        steered_to_scores.extend(values)

        result += build_estimand_result(
            f"steered_to_from:{other_entity['entity_id']}",
            values,
        )

    result += build_estimand_result("steered_to_score", steered_to_scores)

    return result


def _build_debug_json(
    *,
    samples: list[dict[str, Any]],
    other_entities: list[dict[str, Any]],
    comparison_set_id: str,
    instance_hash: str,
    free_text_by_key: dict[tuple[str, str, str, int], dict[str, Any]],
    forced_by_entity: dict[tuple[str, str, str], list[dict[str, Any]]],
    judgments_by_key: dict[tuple[str, str, str, int], dict[str, dict[str, Any]]],
) -> str:
    sample_keys = [
        (
            comparison_set_id,
            instance_hash,
            sample["entity_id"],
            sample["sample_id"],
        )
        for sample in samples
    ]

    incoming_keys = [
        (
            comparison_set_id,
            instance_hash,
            sample_asked_about_other_entity["entity_id"],
            sample_asked_about_other_entity["sample_id"],
        )
        for other_entity in other_entities
        for sample_asked_about_other_entity in forced_by_entity[
            (comparison_set_id, instance_hash, other_entity["entity_id"])
        ]
    ]

    return json.dumps(
        {
            "forced_outputs": samples,
            "free_text_outputs": [free_text_by_key[key] for key in sample_keys],
            "judgments": [judgments_by_key[key] for key in sample_keys],
            "incoming_judgments": [judgments_by_key[key] for key in incoming_keys],
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def run_forced_selection(ctx: RuntimeContext) -> pl.DataFrame:
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

        for sample_id in range(ctx.cfg.num_samples_per_instance):
            tasks.extend(
                {
                    "sample_id": sample_id,
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

    free_text_by_key = {
        (
            row["comparison_set_id"],
            row["assay_instance_hash"],
            row["entity_id"],
            row["sample_id"],
        ): row
        for row in free_text_outputs
    }

    judge_tasks: list[dict] = []

    for task in tasks:
        entities = get_comparison_set_entities(
            comparison_set_df=comparison_set_df,
            entity_lookup=entity_lookup,
            comparison_set_id=task["comparison_set_id"],
        )
        other_entity_names = [
            entity["entity_name"]
            for entity in entities
            if entity["entity_id"] != task["entity_id"]
        ]

        if not other_entity_names:
            continue

        task_key = (
            task["comparison_set_id"],
            task["assay_instance_hash"],
            task["entity_id"],
            task["sample_id"],
        )

        for judge_model in _JUDGE_MODELS:
            judge_tasks.append(
                {
                    "sample_id": task["sample_id"],
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

    forced_by_entity: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(
        list
    )
    for row in forced_outputs:
        forced_by_entity[
            (
                row["comparison_set_id"],
                row["assay_instance_hash"],
                row["entity_id"],
            )
        ].append(row)

    for grouped_samples in forced_by_entity.values():
        grouped_samples.sort(key=lambda row: row["sample_id"])

    judgments_by_key: dict[
        tuple[str, str, str, int], dict[str, dict[str, Any]]
    ] = defaultdict(dict)
    for judgment in judge_outputs:
        judgments_by_key[
            (
                judgment["comparison_set_id"],
                judgment["assay_instance_hash"],
                judgment["source_entity_id"],
                judgment["sample_id"],
            )
        ][judgment["judge_model"]] = judgment

    rows = []

    for assay_instance in assay_instances:
        comparison_set_id = assay_instance["comparison_set_id"]
        comparison_set_name = assay_instance["comparison_set_name"]
        instance_hash = assay_instance["instance_hash"]

        entities = get_comparison_set_entities(
            comparison_set_df=comparison_set_df,
            entity_lookup=entity_lookup,
            comparison_set_id=comparison_set_id,
        )

        for entity in entities:
            entity_id = entity["entity_id"]
            entity_name = entity["entity_name"]

            samples = forced_by_entity[(comparison_set_id, instance_hash, entity_id)]
            other_entities = [
                other_entity
                for other_entity in entities
                if other_entity["entity_id"] != entity_id
            ]

            rows.append(
                {
                    "assay": ctx.cfg.assay,
                    "assay_instance_hash": instance_hash,
                    "model": ctx.cfg.model,
                    "comparison_set_id": comparison_set_id,
                    "comparison_set_name": comparison_set_name,
                    "entity_id": entity_id,
                    "entity_name": entity_name,
                    "result": _build_result_for_entity(
                        samples=samples,
                        entity_name=entity_name,
                        other_entities=other_entities,
                        comparison_set_id=comparison_set_id,
                        instance_hash=instance_hash,
                        forced_by_entity=forced_by_entity,
                        judgments_by_key=judgments_by_key,
                    ),
                    "debug_json": _build_debug_json(
                        samples=samples,
                        other_entities=other_entities,
                        comparison_set_id=comparison_set_id,
                        instance_hash=instance_hash,
                        free_text_by_key=free_text_by_key,
                        forced_by_entity=forced_by_entity,
                        judgments_by_key=judgments_by_key,
                    ),
                }
            )

    ctx.exp.log_metric("assay_instances_completed", len(assay_instances))
    ctx.exp.log_metric("entities_scored", len(rows))
    ctx.exp.log_metric("num_samples_per_instance", ctx.cfg.num_samples_per_instance)

    return pl.DataFrame(rows, schema=ASSAY_SCHEMA)