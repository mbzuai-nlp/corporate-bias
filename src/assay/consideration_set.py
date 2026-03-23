import polars as pl
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from tqdm.auto import tqdm
from typing import Any
import re

from src.model import Message, invoke_model
from src.assay.common import (
    RuntimeContext,
    build_entity_lookup,
    get_comparison_set_entities,
    build_estimand_result,
)
from src.data.model import ASSAY_SCHEMA


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


def run_consideration_set(ctx: RuntimeContext) -> pl.DataFrame:
    def build_messages(instance: dict) -> list[Message]:
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

    def run_consideration(task: dict) -> dict:
        output = invoke_model(
            model=ctx.cfg.model,
            messages=build_messages(instance=task["instance"]),
            use_cache=True,
            plugins=[{"id": "response-healing"}],
            seed=task["sample_id"],
        )

        return {
            "sample_id": task["sample_id"],
            "assay": ctx.cfg.assay,
            "assay_instance_hash": task["assay_instance_hash"],
            "model": ctx.cfg.model,
            "comparison_set_id": task["comparison_set_id"],
            "comparison_set_name": task["comparison_set_name"],
            "text": output.text,
        }

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
            tasks.append(
                {
                    "sample_id": sample_id,
                    "comparison_set_id": comparison_set_id,
                    "comparison_set_name": comparison_set_name,
                    "assay_instance_hash": instance_hash,
                    "instance": instance,
                }
            )

    with ThreadPoolExecutor(max_workers=32) as executor:
        considerations = list(
            tqdm(
                executor.map(run_consideration, tasks),
                total=len(tasks),
                desc="Consideration sets",
            )
        )

    consideration_samples_by_instance: dict[str, list[dict[str, Any]]] = defaultdict(
        list
    )
    for consideration in considerations:
        instance_hash = consideration["assay_instance_hash"]
        entities = entities_by_instance[instance_hash]
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

        consideration_samples_by_instance[instance_hash].append(
            {
                "sample_id": consideration["sample_id"],
                "raw_response": text,
                "first_mentions": first_mentions,
                "ranked_entity_ids": ranked_entity_ids,
                "reciprocal_ranks": reciprocal_ranks,
            }
        )

    rows = []
    for assay_instance in assay_instances:
        instance_hash = assay_instance["instance_hash"]
        comparison_set_id = assay_instance["comparison_set_id"]
        comparison_set_name = assay_instance["comparison_set_name"]
        entities = entities_by_instance[instance_hash]
        consideration_samples = sorted(
            consideration_samples_by_instance[instance_hash],
            key=lambda row: row["sample_id"],
        )

        for entity in entities:
            values = [
                float(sample["reciprocal_ranks"].get(entity["entity_id"], 0.0))
                for sample in consideration_samples
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
                    "result": build_estimand_result("mean_reciprocal_rank", values),
                    "debug_json": json.dumps(
                        {
                            "entity_id": entity["entity_id"],
                            "entity_name": entity["entity_name"],
                            "num_samples_per_instance": ctx.cfg.num_samples_per_instance,
                            "sample_mean_reciprocal_rank_values": values,
                            "samples": [
                                {
                                    "sample_id": sample["sample_id"],
                                    "raw_response": sample["raw_response"],
                                    "first_mention_position": sample[
                                        "first_mentions"
                                    ].get(entity["entity_id"]),
                                    "rank": (
                                        sample["ranked_entity_ids"].index(
                                            entity["entity_id"]
                                        )
                                        + 1
                                        if entity["entity_id"]
                                        in sample["ranked_entity_ids"]
                                        else None
                                    ),
                                    "mean_reciprocal_rank": sample[
                                        "reciprocal_ranks"
                                    ].get(entity["entity_id"], 0.0),
                                }
                                for sample in consideration_samples
                            ],
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                }
            )

    ctx.exp.log_metric("assay_instances_completed", len(assay_instances))
    ctx.exp.log_metric("entities_scored", len(rows))
    ctx.exp.log_metric("num_samples_per_instance", ctx.cfg.num_samples_per_instance)

    return pl.DataFrame(rows, schema=ASSAY_SCHEMA)
