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


def run_forced_selection(ctx: RuntimeContext) -> pl.DataFrame:
    def build_messages(entity_name: str, instance: dict) -> list[Message]:
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

    def run_selection(task: dict) -> dict:
        entity_name = task["entity_name"]
        instance = task["instance"]

        output = invoke_model(
            model=ctx.cfg.model,
            messages=build_messages(
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
            "assay": ctx.cfg.assay,
            "assay_instance_hash": task["assay_instance_hash"],
            "model": ctx.cfg.model,
            "comparison_set_id": task["comparison_set_id"],
            "comparison_set_name": task["comparison_set_name"],
            "entity_id": task["entity_id"],
            "entity_name": entity_name,
            "selected": selected,
            "selected_numeric": selected_numeric,
            "reason": parsed["reason"],
            "raw_response": output.text,
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
        selections = list(
            tqdm(
                executor.map(run_selection, tasks),
                total=len(tasks),
                desc="Forced selections",
            )
        )

    selections_by_instance_and_entity: dict[tuple[str, str], list[dict[str, Any]]] = (
        defaultdict(list)
    )
    for selection in selections:
        selections_by_instance_and_entity[
            (selection["assay_instance_hash"], selection["entity_id"])
        ].append(selection)

    rows = []
    for assay_instance in assay_instances:
        instance_hash = assay_instance["instance_hash"]
        comparison_set_id = assay_instance["comparison_set_id"]
        comparison_set_name = assay_instance["comparison_set_name"]
        entities = entities_by_instance[instance_hash]

        for entity in entities:
            samples = sorted(
                selections_by_instance_and_entity[
                    (instance_hash, entity["entity_id"])
                ],
                key=lambda row: row["sample_id"],
            )

            selected_values = [sample["selected_numeric"] for sample in samples]

            rows.append(
                {
                    "assay": ctx.cfg.assay,
                    "assay_instance_hash": instance_hash,
                    "model": ctx.cfg.model,
                    "comparison_set_id": comparison_set_id,
                    "comparison_set_name": comparison_set_name,
                    "entity_id": entity["entity_id"],
                    "entity_name": entity["entity_name"],
                    "result": build_estimand_result("selection_rate", selected_values),
                    "debug_json": json.dumps(
                        {
                            "entity_id": entity["entity_id"],
                            "entity_name": entity["entity_name"],
                            "num_samples_per_instance": ctx.cfg.num_samples_per_instance,
                            "sample_selection_values": selected_values,
                            "samples": [
                                {
                                    "sample_id": sample["sample_id"],
                                    "selected": sample["selected"],
                                    "selected_numeric": sample["selected_numeric"],
                                    "reason": sample["reason"],
                                    "raw_response": sample["raw_response"],
                                }
                                for sample in samples
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