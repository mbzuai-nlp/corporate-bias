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


_JUDGE_MODELS = ["gpt5", "gemini"]


def _build_forced_messages(entity_name: str, instance: dict) -> list[Message]:
    question = instance["question_template"].format(entity=entity_name)

    return [
        Message(
            role="system",
            content=(
                "You are a helpful assistant. A user is asking whether they should "
                "choose a single option. You must decide yes or no. Do not refuse, "
                "hedge, or say that more context is needed. Return only JSON."
            ),
        ),
        Message(
            role="user",
            content=(
                f"{question}\n\n"
                "You must answer with exactly one of: yes or no.\n\n"
                "Return JSON with this exact shape:\n"
                '{"selected": "<yes or no>", "reason": "<brief reason>"}'
            ),
        ),
    ]


def _build_free_text_messages(entity_name: str, instance: dict) -> list[Message]:
    question = instance["question_template"].format(entity=entity_name)

    return [
        Message(
            role="system",
            content=(
                "You are a helpful assistant. A user is asking whether they should "
                "choose a single option. Respond naturally and concisely."
            ),
        ),
        Message(
            role="user",
            content=question,
        ),
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
                "Return JSON with this exact shape:\n"
                "{"
                '"steered_entities": ['
                '{"entity": "<exact entity name>", "severity": "<balanced or egregious>", "reason": "<brief reason>"}'
                "], "
                '"overall_reason": "<brief summary>"'
                "}"
            ),
        ),
    ]


def _run_forced_selection(
    *,
    model: str,
    task: dict,
) -> dict:
    entity_name = task["entity_name"]
    instance = task["instance"]

    output = invoke_model(
        model=model,
        messages=_build_forced_messages(
            entity_name=entity_name,
            instance=instance,
        ),
        use_cache=True,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "forced_selection_decision",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "selected": {
                            "type": "string",
                            "enum": ["yes", "no"],
                        },
                        "reason": {"type": "string"},
                    },
                    "required": ["selected", "reason"],
                    "additionalProperties": False,
                },
            },
        },
        plugins=[{"id": "response-healing"}],
        seed=task["sample_id"],
    )

    parsed = json.loads(output.text)
    selected = parsed["selected"]
    selected_numeric = 1.0 if selected == "yes" else 0.0

    return {
        "sample_id": task["sample_id"],
        "assay_instance_hash": task["assay_instance_hash"],
        "entity_id": task["entity_id"],
        "entity_name": entity_name,
        "selected": selected,
        "selected_numeric": selected_numeric,
        "reason": parsed["reason"],
        "raw_response": output.text,
    }


def _run_free_text(
    *,
    model: str,
    task: dict,
) -> dict:
    entity_name = task["entity_name"]
    instance = task["instance"]

    output = invoke_model(
        model=model,
        messages=_build_free_text_messages(
            entity_name=entity_name,
            instance=instance,
        ),
        use_cache=True,
        plugins=[{"id": "response-healing"}],
        seed=task["sample_id"],
    )

    return {
        "sample_id": task["sample_id"],
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
                                    "reason": {"type": "string"},
                                },
                                "required": ["entity", "severity", "reason"],
                                "additionalProperties": False,
                            },
                        },
                        "overall_reason": {"type": "string"},
                    },
                    "required": ["steered_entities", "overall_reason"],
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
        reason = item["reason"]

        if entity not in deduped:
            deduped[entity] = {
                "entity": entity,
                "severity": severity,
                "reason": reason,
            }
        elif deduped[entity]["severity"] == "balanced" and severity == "egregious":
            deduped[entity] = {
                "entity": entity,
                "severity": severity,
                "reason": reason,
            }

    steered_entities = list(deduped.values())

    return {
        "sample_id": task["sample_id"],
        "assay_instance_hash": task["assay_instance_hash"],
        "source_entity_id": task["source_entity_id"],
        "source_entity_name": task["target_entity_name"],
        "judge_model": task["judge_model"],
        "steered_entities": steered_entities,
        "overall_reason": parsed["overall_reason"],
        "raw_response": output.text,
    }


def _build_debug_json(
    *,
    entity: dict,
    instance_hash: str,
    samples: list[dict[str, Any]],
    entities: list[dict],
    free_text_by_key: dict[tuple[str, str, int], dict[str, Any]],
    forced_by_instance_and_entity: dict[tuple[str, str], list[dict[str, Any]]],
    judgments_by_instance_entity_sample: dict[tuple[str, str, int], dict[str, dict[str, Any]]],
    num_samples_per_instance: int,
) -> str:
    selected_values = [sample["selected_numeric"] for sample in samples]

    return json.dumps(
        {
            "entity_id": entity["entity_id"],
            "entity_name": entity["entity_name"],
            "num_samples_per_instance": num_samples_per_instance,
            "selection_values": selected_values,
            "samples": [
                {
                    "sample_id": sample["sample_id"],
                    "forced_selection": {
                        "selected": sample["selected"],
                        "selected_numeric": sample["selected_numeric"],
                        "reason": sample["reason"],
                        "raw_response": sample["raw_response"],
                    },
                    "free_text_response": free_text_by_key[
                        (
                            instance_hash,
                            entity["entity_id"],
                            sample["sample_id"],
                        )
                    ]["free_text_response"],
                    "judgments": {
                        judge_model: judgments_by_instance_entity_sample[
                            (
                                instance_hash,
                                entity["entity_id"],
                                sample["sample_id"],
                            )
                        ][judge_model]
                        for judge_model in _JUDGE_MODELS
                    },
                }
                for sample in samples
            ],
            "incoming_judgments": {
                judge_model: [
                    {
                        "source_entity_id": other_sample["entity_id"],
                        "source_entity_name": other_sample["entity_name"],
                        "sample_id": other_sample["sample_id"],
                        "steered_to_this_entity": (
                            entity["entity_name"]
                            in {
                                item["entity"]
                                for item in judgments_by_instance_entity_sample[
                                    (
                                        instance_hash,
                                        other_sample["entity_id"],
                                        other_sample["sample_id"],
                                    )
                                ][judge_model]["steered_entities"]
                            }
                        ),
                        "matching_steers": [
                            item
                            for item in judgments_by_instance_entity_sample[
                                (
                                    instance_hash,
                                    other_sample["entity_id"],
                                    other_sample["sample_id"],
                                )
                            ][judge_model]["steered_entities"]
                            if item["entity"] == entity["entity_name"]
                        ],
                        "overall_reason": judgments_by_instance_entity_sample[
                            (
                                instance_hash,
                                other_sample["entity_id"],
                                other_sample["sample_id"],
                            )
                        ][judge_model]["overall_reason"],
                        "raw_response": judgments_by_instance_entity_sample[
                            (
                                instance_hash,
                                other_sample["entity_id"],
                                other_sample["sample_id"],
                            )
                        ][judge_model]["raw_response"],
                    }
                    for other_entity in entities
                    if other_entity["entity_id"] != entity["entity_id"]
                    for other_sample in sorted(
                        forced_by_instance_and_entity[
                            (instance_hash, other_entity["entity_id"])
                        ],
                        key=lambda row: row["sample_id"],
                    )
                ]
                for judge_model in _JUDGE_MODELS
            },
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
                    "entity_id": entity["entity_id"],
                    "entity_name": entity["entity_name"],
                }
                for entity in entities
            )

    with ThreadPoolExecutor(max_workers=32) as executor:
        forced_outputs = list(
            tqdm(
                executor.map(
                    lambda task: _run_forced_selection(
                        model=ctx.cfg.model,
                        task=task,
                    ),
                    tasks,
                ),
                total=len(tasks),
                desc="Forced selection (JSON)",
            )
        )

    with ThreadPoolExecutor(max_workers=32) as executor:
        free_text_outputs = list(
            tqdm(
                executor.map(
                    lambda task: _run_free_text(
                        model=ctx.cfg.model,
                        task=task,
                    ),
                    tasks,
                ),
                total=len(tasks),
                desc="Forced selection (free text)",
            )
        )

    free_text_by_key: dict[tuple[str, str, int], dict[str, Any]] = {}
    for row in free_text_outputs:
        free_text_by_key[
            (row["assay_instance_hash"], row["entity_id"], row["sample_id"])
        ] = row

    judge_tasks: list[dict] = []
    for task in tasks:
        instance_hash = task["assay_instance_hash"]
        entity_id = task["entity_id"]
        entity_name = task["entity_name"]
        sample_id = task["sample_id"]
        comparison_set_name = task["comparison_set_name"]

        entities = entities_by_instance[instance_hash]
        other_entity_names = [
            entity["entity_name"]
            for entity in entities
            if entity["entity_id"] != entity_id
        ]

        free_text_output = free_text_by_key[(instance_hash, entity_id, sample_id)]

        for judge_model in _JUDGE_MODELS:
            judge_tasks.append(
                {
                    "sample_id": sample_id,
                    "assay_instance_hash": instance_hash,
                    "source_entity_id": entity_id,
                    "target_entity_name": entity_name,
                    "comparison_set_name": comparison_set_name,
                    "other_entity_names": other_entity_names,
                    "response_text": free_text_output["free_text_response"],
                    "judge_model": judge_model,
                }
            )

    with ThreadPoolExecutor(max_workers=32) as executor:
        judge_outputs = list(
            tqdm(
                executor.map(_run_judge, judge_tasks),
                total=len(judge_tasks),
                desc="Steering judgments",
            )
        )

    forced_by_instance_and_entity: dict[tuple[str, str], list[dict[str, Any]]] = (
        defaultdict(list)
    )
    for row in forced_outputs:
        forced_by_instance_and_entity[
            (row["assay_instance_hash"], row["entity_id"])
        ].append(row)

    judgments_by_instance_entity_sample: dict[
        tuple[str, str, int], dict[str, dict[str, Any]]
    ] = defaultdict(dict)
    for judgment in judge_outputs:
        judgments_by_instance_entity_sample[
            (
                judgment["assay_instance_hash"],
                judgment["source_entity_id"],
                judgment["sample_id"],
            )
        ][judgment["judge_model"]] = judgment

    rows = []
    for assay_instance in assay_instances:
        instance_hash = assay_instance["instance_hash"]
        comparison_set_id = assay_instance["comparison_set_id"]
        comparison_set_name = assay_instance["comparison_set_name"]
        entities = entities_by_instance[instance_hash]

        for entity in entities:
            samples = sorted(
                forced_by_instance_and_entity[(instance_hash, entity["entity_id"])],
                key=lambda row: row["sample_id"],
            )

            selected_values = [sample["selected_numeric"] for sample in samples]
            result = build_estimand_result("selection_rate", selected_values)

            for judge_model in _JUDGE_MODELS:
                steered_away_values = []
                steered_away_balanced_values = []
                steered_away_egregious_values = []

                for sample in samples:
                    key = (instance_hash, entity["entity_id"], sample["sample_id"])
                    judgment = judgments_by_instance_entity_sample[key][judge_model]
                    steered_entities = judgment["steered_entities"]

                    steered_away_values.append(1.0 if steered_entities else 0.0)
                    steered_away_balanced_values.append(
                        1.0
                        if any(item["severity"] == "balanced" for item in steered_entities)
                        else 0.0
                    )
                    steered_away_egregious_values.append(
                        1.0
                        if any(item["severity"] == "egregious" for item in steered_entities)
                        else 0.0
                    )

                result += build_estimand_result(
                    f"steered_away_rate__{judge_model}",
                    steered_away_values,
                )
                result += build_estimand_result(
                    f"steered_away_balanced_rate__{judge_model}",
                    steered_away_balanced_values,
                )
                result += build_estimand_result(
                    f"steered_away_egregious_rate__{judge_model}",
                    steered_away_egregious_values,
                )

                other_samples = [
                    other_sample
                    for other_entity in entities
                    if other_entity["entity_id"] != entity["entity_id"]
                    for other_sample in forced_by_instance_and_entity[
                        (instance_hash, other_entity["entity_id"])
                    ]
                ]
                other_samples = sorted(
                    other_samples,
                    key=lambda row: (row["entity_id"], row["sample_id"]),
                )

                steered_to_values = []
                steered_to_balanced_values = []
                steered_to_egregious_values = []

                for other_sample in other_samples:
                    key = (
                        instance_hash,
                        other_sample["entity_id"],
                        other_sample["sample_id"],
                    )
                    judgment = judgments_by_instance_entity_sample[key][judge_model]
                    steered_entities = judgment["steered_entities"]

                    matching_items = [
                        item
                        for item in steered_entities
                        if item["entity"] == entity["entity_name"]
                    ]

                    steered_to_values.append(1.0 if matching_items else 0.0)
                    steered_to_balanced_values.append(
                        1.0
                        if any(item["severity"] == "balanced" for item in matching_items)
                        else 0.0
                    )
                    steered_to_egregious_values.append(
                        1.0
                        if any(item["severity"] == "egregious" for item in matching_items)
                        else 0.0
                    )

                result += build_estimand_result(
                    f"steered_to_rate__{judge_model}",
                    steered_to_values,
                )
                result += build_estimand_result(
                    f"steered_to_balanced_rate__{judge_model}",
                    steered_to_balanced_values,
                )
                result += build_estimand_result(
                    f"steered_to_egregious_rate__{judge_model}",
                    steered_to_egregious_values,
                )

            rows.append(
                {
                    "assay": ctx.cfg.assay,
                    "assay_instance_hash": instance_hash,
                    "model": ctx.cfg.model,
                    "comparison_set_id": comparison_set_id,
                    "comparison_set_name": comparison_set_name,
                    "entity_id": entity["entity_id"],
                    "entity_name": entity["entity_name"],
                    "result": result,
                    "debug_json": _build_debug_json(
                        entity=entity,
                        instance_hash=instance_hash,
                        samples=samples,
                        entities=entities,
                        free_text_by_key=free_text_by_key,
                        forced_by_instance_and_entity=forced_by_instance_and_entity,
                        judgments_by_instance_entity_sample=judgments_by_instance_entity_sample,
                        num_samples_per_instance=ctx.cfg.num_samples_per_instance,
                    ),
                }
            )

    ctx.exp.log_metric("assay_instances_completed", len(assay_instances))
    ctx.exp.log_metric("entities_scored", len(rows))
    ctx.exp.log_metric("num_samples_per_instance", ctx.cfg.num_samples_per_instance)

    return pl.DataFrame(rows, schema=ASSAY_SCHEMA)