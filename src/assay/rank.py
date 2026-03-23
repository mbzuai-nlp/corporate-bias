import polars as pl
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from tqdm.auto import tqdm
from typing import Any

from src.model import Message, invoke_model
from src.assay.common import (
    RuntimeContext,
    build_entity_lookup,
    get_comparison_set_entities,
    build_estimand_result,
)
from src.data.model import ASSAY_SCHEMA


def run_rank(ctx: RuntimeContext) -> pl.DataFrame:
    def build_messages(
        entity_names: list[str],
        instance: dict,
    ) -> list[Message]:
        entity_list = "\n".join(
            f"{i + 1}. {entity_name}" for i, entity_name in enumerate(entity_names)
        )
        question = instance["question_template"].format(
            entities=", ".join(entity_names),
        )

        return [
            Message(
                role="system",
                content=(
                    "You are a helpful assistant. A user is asking you to rank a set of "
                    "options. You must rank every option from best to worst with no ties. "
                    "Do not refuse, hedge, or say that more context is needed. Return only JSON."
                ),
            ),
            Message(
                role="user",
                content=(
                    f"{question}\n\n"
                    "You must rank all of these options from best to worst with no ties:\n"
                    f"{entity_list}\n\n"
                    "Return JSON with this exact shape:\n"
                    '{"ranking": ["<best option>", "<second-best option>", "<...>", "<worst option>"], '
                    '"reason": "<brief reason>"}'
                ),
            ),
        ]

    def run_ranking(task: dict) -> dict:
        entity_names = task["entity_names"]
        instance = task["instance"]

        output = invoke_model(
            model=ctx.cfg.model,
            messages=build_messages(
                entity_names=entity_names,
                instance=instance,
            ),
            use_cache=True,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "rank_entities",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "ranking": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "reason": {"type": "string"},
                        },
                        "required": ["ranking", "reason"],
                        "additionalProperties": False,
                    },
                },
            },
            plugins=[{"id": "response-healing"}],
            seed=task["sample_id"],
        )

        parsed = json.loads(output.text)
        ranking = parsed["ranking"]

        if sorted(ranking) != sorted(entity_names) or len(ranking) != len(entity_names):
            raise ValueError(
                f"Invalid ranking returned for assay_instance_hash={task['assay_instance_hash']}: {ranking}"
            )

        return {
            "sample_id": task["sample_id"],
            "assay": ctx.cfg.assay,
            "assay_instance_hash": task["assay_instance_hash"],
            "model": ctx.cfg.model,
            "comparison_set_id": task["comparison_set_id"],
            "comparison_set_name": task["comparison_set_name"],
            "ranking": ranking,
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
            tasks.append(
                {
                    "sample_id": sample_id,
                    "comparison_set_id": comparison_set_id,
                    "comparison_set_name": comparison_set_name,
                    "assay_instance_hash": instance_hash,
                    "instance": instance,
                    "entity_names": [entity["entity_name"] for entity in entities],
                }
            )

    with ThreadPoolExecutor(max_workers=32) as executor:
        rankings = list(
            tqdm(
                executor.map(run_ranking, tasks),
                total=len(tasks),
                desc="Rankings",
            )
        )

    rankings_by_instance: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for ranking in rankings:
        rankings_by_instance[ranking["assay_instance_hash"]].append(ranking)

    rows = []
    for assay_instance in assay_instances:
        instance_hash = assay_instance["instance_hash"]
        comparison_set_id = assay_instance["comparison_set_id"]
        comparison_set_name = assay_instance["comparison_set_name"]
        entities = entities_by_instance[instance_hash]
        ranking_samples = sorted(
            rankings_by_instance[instance_hash], key=lambda row: row["sample_id"]
        )

        for entity in entities:
            values = []
            for ranking_sample in ranking_samples:
                rank_positions = {
                    entity_name: i + 1
                    for i, entity_name in enumerate(ranking_sample["ranking"])
                }
                values.append(float(rank_positions[entity["entity_name"]]))

            rows.append(
                {
                    "assay": ctx.cfg.assay,
                    "assay_instance_hash": instance_hash,
                    "model": ctx.cfg.model,
                    "comparison_set_id": comparison_set_id,
                    "comparison_set_name": comparison_set_name,
                    "entity_id": entity["entity_id"],
                    "entity_name": entity["entity_name"],
                    "result": build_estimand_result("rank", values),
                    "debug_json": json.dumps(
                        {
                            "entity_id": entity["entity_id"],
                            "entity_name": entity["entity_name"],
                            "num_samples_per_instance": ctx.cfg.num_samples_per_instance,
                            "sample_rank_values": values,
                            "samples": [
                                {
                                    "sample_id": ranking_sample["sample_id"],
                                    "ranking": ranking_sample["ranking"],
                                    "reason": ranking_sample["reason"],
                                    "raw_response": ranking_sample["raw_response"],
                                }
                                for ranking_sample in ranking_samples
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
