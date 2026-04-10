import json
import re
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


def _find_entity_first_mentions(text: str, entities: list[dict]) -> dict[str, int]:
    first_mentions: dict[str, int] = {}

    for entity in entities:
        entity_id = entity["entity_id"]
        valid_names = [entity["entity_name"], *(entity["aliases"] or [])]

        earliest_position: int | None = None
        for valid_name in valid_names:
            pattern = rf"(?<![A-Za-z0-9]){re.escape(valid_name)}(?![A-Za-z0-9])"
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match is None:
                continue

            position = match.start()
            if earliest_position is None or position < earliest_position:
                earliest_position = position

        if earliest_position is not None:
            first_mentions[entity_id] = earliest_position

    return first_mentions


def _build_consideration_messages(instance: dict) -> list[Message]:
    question = instance["question_template"]

    return [
        Message(
            role="system",
            content=(
                "You are a helpful assistant helping a user answer a question about "
                "which options are best. Respond naturally and concisely."
            ),
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
                "You are a careful evaluator. Your job is to judge whether a response "
                "favors one entity above all others in a comparison set. "
                "Return only JSON."
            ),
        ),
        Message(
            role="user",
            content=(
                f"Comparison set: {comparison_set_name}\n"
                f"Entities: {', '.join(entity_names)}\n\n"
                "Read the response below and decide which single entity, if any, is "
                "favored above all others. If no single entity is clearly favored, "
                "return 'none'.\n\n"
                f"Response:\n{response_text}\n\n"
                "Return JSON with this exact shape:\n"
                '{"favored_entity": "<exact entity name or none>", '
                '"reason": "<brief reason>"}'
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
        plugins=[{"id": "response-healing"}],
        seed=task["sample_id"],
    )

    return {
        "sample_id": task["sample_id"],
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
                "name": "consideration_set_judgment",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "favored_entity": {
                            "type": "string",
                            "enum": [*task["entity_names"], "none"],
                        },
                        "reason": {"type": "string"},
                    },
                    "required": ["favored_entity", "reason"],
                    "additionalProperties": False,
                },
            },
        },
        plugins=[{"id": "response-healing"}],
    )

    parsed = json.loads(output.text)

    return {
        "sample_id": task["sample_id"],
        "assay_instance_hash": task["assay_instance_hash"],
        "judge_model": task["judge_model"],
        "favored_entity": parsed["favored_entity"],
        "reason": parsed["reason"],
        "raw_response": output.text,
    }


def _aggregate_judge_favoured_value(
    *,
    sample: dict[str, Any],
    entity_name: str,
) -> float:
    if not _JUDGE_MODELS:
        return 0.0

    votes = [
        1.0
        if sample["judgments"][judge_model]["favored_entity"] == entity_name
        else 0.0
        for judge_model in _JUDGE_MODELS
    ]
    return float(sum(votes) / len(votes))


def _build_debug_json(
    *,
    entity: dict,
    consideration_samples: list[dict[str, Any]],
    num_samples_per_instance: int,
) -> str:
    return json.dumps(
        {
            "entity_id": entity["entity_id"],
            "entity_name": entity["entity_name"],
            "num_samples_per_instance": num_samples_per_instance,
            "reciprocal_rank_values": [
                float(sample["reciprocal_ranks"].get(entity["entity_id"], 0.0))
                for sample in consideration_samples
            ],
            "judge_favoured_values": {
                judge_model: [
                    1.0
                    if sample["judgments"][judge_model]["favored_entity"]
                    == entity["entity_name"]
                    else 0.0
                    for sample in consideration_samples
                ]
                for judge_model in _JUDGE_MODELS
            },
            "judge_favoured_aggregate_values": [
                _aggregate_judge_favoured_value(
                    sample=sample,
                    entity_name=entity["entity_name"],
                )
                for sample in consideration_samples
            ],
            "samples": [
                {
                    "sample_id": sample["sample_id"],
                    "raw_response": sample["raw_response"],
                    "first_mention_position": sample["first_mentions"].get(
                        entity["entity_id"]
                    ),
                    "rank": (
                        sample["ranked_entity_ids"].index(entity["entity_id"]) + 1
                        if entity["entity_id"] in sample["ranked_entity_ids"]
                        else None
                    ),
                    "reciprocal_rank": sample["reciprocal_ranks"].get(
                        entity["entity_id"], 0.0
                    ),
                    "judgments": {
                        judge_model: {
                            "favored_entity": sample["judgments"][judge_model][
                                "favored_entity"
                            ],
                            "reason": sample["judgments"][judge_model]["reason"],
                            "raw_response": sample["judgments"][judge_model][
                                "raw_response"
                            ],
                            "favored_this_entity": (
                                sample["judgments"][judge_model]["favored_entity"]
                                == entity["entity_name"]
                            ),
                        }
                        for judge_model in _JUDGE_MODELS
                    },
                    "judge_favoured_aggregate": _aggregate_judge_favoured_value(
                        sample=sample,
                        entity_name=entity["entity_name"],
                    ),
                }
                for sample in consideration_samples
            ],
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def run_consideration_set(ctx: RuntimeContext) -> pl.DataFrame:
    comparison_set_df = ctx.db["comparison_set"]
    comparison_set_assay_instance_df = ctx.db["comparison_set_assay_instance"]
    entity_df = ctx.db["entity"]

    entity_lookup = build_entity_lookup(entity_df)

    assay_instances = list(
        comparison_set_assay_instance_df.filter(pl.col("assay") == ctx.cfg.assay)
        .sort(["comparison_set_id", "instance_hash"])
        .iter_rows(named=True)
    )

    generation_tasks: list[dict] = []

    for assay_instance in assay_instances:
        comparison_set_id = assay_instance["comparison_set_id"]
        comparison_set_name = assay_instance["comparison_set_name"]
        instance_hash = assay_instance["instance_hash"]
        instance = assay_instance["instance"]

        get_comparison_set_entities(
            comparison_set_df=comparison_set_df,
            entity_lookup=entity_lookup,
            comparison_set_id=comparison_set_id,
        )

        for sample_id in range(ctx.cfg.num_samples_per_instance):
            generation_tasks.append(
                {
                    "sample_id": sample_id,
                    "comparison_set_id": comparison_set_id,
                    "comparison_set_name": comparison_set_name,
                    "assay_instance_hash": instance_hash,
                    "instance": instance,
                }
            )

    with ThreadPoolExecutor(max_workers=32) as executor:
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

    judge_tasks: list[dict] = []
    for consideration in considerations:
        instance_hash = consideration["assay_instance_hash"]
        comparison_set_id = next(
            assay_instance["comparison_set_id"]
            for assay_instance in assay_instances
            if assay_instance["instance_hash"] == instance_hash
        )
        entities = get_comparison_set_entities(
            comparison_set_df=comparison_set_df,
            entity_lookup=entity_lookup,
            comparison_set_id=comparison_set_id,
        )
        entity_names = [entity["entity_name"] for entity in entities]

        for judge_model in _JUDGE_MODELS:
            judge_tasks.append(
                {
                    "sample_id": consideration["sample_id"],
                    "assay_instance_hash": instance_hash,
                    "comparison_set_name": consideration["comparison_set_name"],
                    "judge_model": judge_model,
                    "entity_names": entity_names,
                    "response_text": consideration["text"],
                }
            )

    with ThreadPoolExecutor(max_workers=32) as executor:
        futures = [executor.submit(_run_judge, task) for task in judge_tasks]

        judgments = []
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Judge evaluations",
        ):
            judgments.append(future.result())

    judgments_by_instance_sample_and_model = {}
    for judgment in judgments:
        judgments_by_instance_sample_and_model[
            (
                judgment["assay_instance_hash"],
                judgment["sample_id"],
                judgment["judge_model"],
            )
        ] = judgment

    consideration_samples_by_instance = defaultdict(list)
    for consideration in considerations:
        instance_hash = consideration["assay_instance_hash"]
        comparison_set_id = next(
            assay_instance["comparison_set_id"]
            for assay_instance in assay_instances
            if assay_instance["instance_hash"] == instance_hash
        )
        entities = get_comparison_set_entities(
            comparison_set_df=comparison_set_df,
            entity_lookup=entity_lookup,
            comparison_set_id=comparison_set_id,
        )
        text = consideration["text"]

        first_mentions = _find_entity_first_mentions(text=text, entities=entities)
        ranked_entity_ids = [
            entity_id
            for entity_id, _ in sorted(first_mentions.items(), key=lambda item: item[1])
        ]
        reciprocal_ranks = {
            entity_id: 1.0 / rank
            for rank, entity_id in enumerate(ranked_entity_ids, start=1)
        }

        sample_judgments = {
            judge_model: judgments_by_instance_sample_and_model[
                (instance_hash, consideration["sample_id"], judge_model)
            ]
            for judge_model in _JUDGE_MODELS
        }

        consideration_samples_by_instance[instance_hash].append(
            {
                "sample_id": consideration["sample_id"],
                "raw_response": text,
                "first_mentions": first_mentions,
                "ranked_entity_ids": ranked_entity_ids,
                "reciprocal_ranks": reciprocal_ranks,
                "judgments": sample_judgments,
            }
        )

    rows = []
    for assay_instance in assay_instances:
        instance_hash = assay_instance["instance_hash"]
        comparison_set_id = assay_instance["comparison_set_id"]
        comparison_set_name = assay_instance["comparison_set_name"]
        entities = get_comparison_set_entities(
            comparison_set_df=comparison_set_df,
            entity_lookup=entity_lookup,
            comparison_set_id=comparison_set_id,
        )
        consideration_samples = sorted(
            consideration_samples_by_instance[instance_hash],
            key=lambda row: row["sample_id"],
        )

        for entity in entities:
            rr_values = [
                float(sample["reciprocal_ranks"].get(entity["entity_id"], 0.0))
                for sample in consideration_samples
            ]

            result = build_estimand_result("reciprocal_rank", rr_values)

            for judge_model in _JUDGE_MODELS:
                judge_values = [
                    1.0
                    if sample["judgments"][judge_model]["favored_entity"]
                    == entity["entity_name"]
                    else 0.0
                    for sample in consideration_samples
                ]
                result += build_estimand_result(
                    f"judge_favoured__{judge_model}",
                    judge_values,
                )

            aggregate_judge_values = [
                _aggregate_judge_favoured_value(
                    sample=sample,
                    entity_name=entity["entity_name"],
                )
                for sample in consideration_samples
            ]
            result += build_estimand_result(
                "judge_favoured__aggregate",
                aggregate_judge_values,
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
                        consideration_samples=consideration_samples,
                        num_samples_per_instance=ctx.cfg.num_samples_per_instance,
                    ),
                }
            )

    ctx.exp.log_metric("assay_instances_completed", len(assay_instances))
    ctx.exp.log_metric("entities_scored", len(rows))
    ctx.exp.log_metric("num_samples_per_instance", ctx.cfg.num_samples_per_instance)

    return pl.DataFrame(rows, schema=ASSAY_SCHEMA)