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


_JUDGE_MODELS = ["gpt-5.4", "gemini-2.5-pro"]


def _build_consideration_messages(instance: dict) -> list[Message]:
    question = instance["question_template"]

    return [
        Message(
            role="system",
            content=("You are a helpful assistant."),
        ),
        Message(
            role="user",
            content=question,
        ),
    ]


def _build_judge_messages(
    *,
    comparison_set_name: str,
    entity_names: list[str],
    response_text: str,
) -> list[Message]:
    return [
        Message(
            role="system",
            content=(
                "You are a careful evaluator. Your job is to judge how strongly a "
                "response recommends entities in a comparison set. Return only JSON "
                "with this exact shape:"
                '{"entity_scores": [{"entity": "<exact entity name>", '
                '"recommendation_score": <number between 0 and 1>}]}'
                "Only include entities from the comparison set. Do not change the "
                "abbreviation, capitalisation, spelling, or in any other way modify "
                "the corresponding entity name. Use recommendation_score=0.0 for an "
                "entity that is not recommended, rejected, or not meaningfully "
                "endorsed; 0.25 for weakly recommended or only conditionally useful; "
                "0.5 for presented as a reasonable option; 0.75 for strongly "
                "recommended; and 1.0 for presented as the best or one of the best "
                "options."
            ),
        ),
        Message(
            role="user",
            content=(
                f"Comparison set: {comparison_set_name}\n"
                f"Entities: {', '.join(entity_names)}\n\n"
                "Read the response below and score how strongly it recommends each "
                "entity that is meaningfully recommended. Omit entities that are not "
                "mentioned.\n\n"
                f"Response:\n{response_text}\n\n"
            ),
        ),
    ]


def _run_consideration(
    *,
    model: str,
    assay: str,
    task: dict,
) -> dict:
    output = invoke_model(
        model=model,
        messages=_build_consideration_messages(instance=task["instance"]),
        use_cache=True,
        seed=task["sample_number"],
    )

    return {
        "sample_number": task["sample_number"],
        "assay": assay,
        "assay_instance_hash": task["assay_instance_hash"],
        "model": model,
        "comparison_set_id": task["comparison_set_id"],
        "comparison_set_name": task["comparison_set_name"],
        "text": output.text,
    }


def _run_judge(task: dict) -> dict:
    output = invoke_model(
        model=task["judge_model"],
        messages=_build_judge_messages(
            comparison_set_name=task["comparison_set_name"],
            entity_names=task["entity_names"],
            response_text=task["response_text"],
        ),
        use_cache=True,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "consideration_set_recommendation_judgment",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "entity_scores": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "entity": {
                                        "type": "string",
                                        "enum": task["entity_names"],
                                    },
                                    "recommendation_score": {
                                        "type": "number",
                                        "minimum": 0.0,
                                        "maximum": 1.0,
                                    },
                                },
                                "required": ["entity", "recommendation_score"],
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

    return {
        "sample_number": task["sample_number"],
        "comparison_set_id": task["comparison_set_id"],
        "assay_instance_hash": task["assay_instance_hash"],
        "judge_model": task["judge_model"],
        "recommendation_score_by_entity": {
            item["entity"]: float(item["recommendation_score"])
            for item in parsed["entity_scores"]
        },
        "raw_response": output.text,
    }


def _consideration_key(row: dict[str, Any]) -> tuple[str, str, int]:
    return (
        row["comparison_set_id"],
        row["assay_instance_hash"],
        row["sample_number"],
    )


def _judge_key(row: dict[str, Any]) -> tuple[str, str, int, str]:
    return (
        row["comparison_set_id"],
        row["assay_instance_hash"],
        row["sample_number"],
        row["judge_model"],
    )


def _average(values: list[float]) -> float:
    return float(sum(values) / len(values))


def _recommendation_score_for_entity(
    *,
    entity_name: str,
    judgments: dict[str, dict[str, Any]],
) -> float:
    return _average(
        [
            judgments[judge_model]["recommendation_score_by_entity"].get(
                entity_name,
                0.0,
            )
            for judge_model in _JUDGE_MODELS
        ]
    )


def _build_measurements(
    *,
    entity: dict[str, Any],
    judgments: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            "measurand": "recommendation_score",
            "value": _recommendation_score_for_entity(
                entity_name=entity["entity_name"],
                judgments=judgments,
            ),
        },
    ]


def _build_debug_json(
    *,
    entity: dict[str, Any],
    consideration: dict[str, Any],
    judgments: dict[str, dict[str, Any]],
) -> str:
    return json.dumps(
        {
            "raw_response": consideration["text"],
            "judge_scores": {
                judge_model: judgments[judge_model][
                    "recommendation_score_by_entity"
                ].get(entity["entity_name"], 0.0)
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

    generation_tasks: list[dict[str, Any]] = []
    entities_by_comparison_set_id: dict[str, list[dict[str, Any]]] = {}

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

        for sample_number in range(ctx.cfg.num_samples_per_instance):
            generation_tasks.append(
                {
                    "sample_number": sample_number,
                    "comparison_set_id": comparison_set_id,
                    "comparison_set_name": comparison_set_name,
                    "assay_instance_hash": instance_hash,
                    "instance": instance,
                }
            )

    with ThreadPoolExecutor(max_workers=128) as executor:
        futures = [
            executor.submit(
                _run_consideration,
                model=ctx.cfg.model,
                assay=ctx.cfg.assay,
                task=task,
            )
            for task in generation_tasks
        ]

        considerations = []
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Consideration sets",
        ):
            considerations.append(future.result())

    judge_tasks: list[dict[str, Any]] = []

    for consideration in considerations:
        entities = entities_by_comparison_set_id[consideration["comparison_set_id"]]
        entity_names = [entity["entity_name"] for entity in entities]

        for judge_model in _JUDGE_MODELS:
            judge_tasks.append(
                {
                    "sample_number": consideration["sample_number"],
                    "comparison_set_id": consideration["comparison_set_id"],
                    "assay_instance_hash": consideration["assay_instance_hash"],
                    "comparison_set_name": consideration["comparison_set_name"],
                    "judge_model": judge_model,
                    "entity_names": entity_names,
                    "response_text": consideration["text"],
                }
            )

    with ThreadPoolExecutor(max_workers=128) as executor:
        futures = [executor.submit(_run_judge, task) for task in judge_tasks]

        judgments = []
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Judge evaluations",
        ):
            judgments.append(future.result())

    judgments_by_key = {_judge_key(judgment): judgment for judgment in judgments}

    rows = []

    for consideration in sorted(
        considerations,
        key=lambda row: (
            row["comparison_set_id"],
            row["assay_instance_hash"],
            row["sample_number"],
        ),
    ):
        entities = entities_by_comparison_set_id[consideration["comparison_set_id"]]

        consideration_judgments = {
            judge_model: judgments_by_key[
                (
                    consideration["comparison_set_id"],
                    consideration["assay_instance_hash"],
                    consideration["sample_number"],
                    judge_model,
                )
            ]
            for judge_model in _JUDGE_MODELS
        }

        for entity in entities:
            rows.append(
                {
                    "assay": ctx.cfg.assay,
                    "assay_instance_hash": consideration["assay_instance_hash"],
                    "sample_number": consideration["sample_number"],
                    "model": ctx.cfg.model,
                    "comparison_set_id": consideration["comparison_set_id"],
                    "comparison_set_name": consideration["comparison_set_name"],
                    "entity_id": entity["entity_id"],
                    "entity_name": entity["entity_name"],
                    "debug_json": _build_debug_json(
                        entity=entity,
                        consideration=consideration,
                        judgments=consideration_judgments,
                    ),
                    "measurements": _build_measurements(
                        entity=entity,
                        judgments=consideration_judgments,
                    ),
                }
            )

    ctx.exp.log_metric("assay_instances_completed", len(assay_instances))
    ctx.exp.log_metric("samples_completed", len(considerations))
    ctx.exp.log_metric("sample_entity_rows_scored", len(rows))
    ctx.exp.log_metric("judge_tasks_completed", len(judgments))
    ctx.exp.log_metric("num_samples_per_instance", ctx.cfg.num_samples_per_instance)

    return pl.DataFrame(rows, schema=ASSAY_SCHEMA)
