import json
import random
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


SYSTEM_PROMPT = """
You are a helpful assistant. When the user provides a list of options, you must return
a JSON object that ranks the options based on the user's query.

Your response must be JSON of the shape.

{"ranking": "<complete, deduplicated list of options provided by the user>"}

Your selection should not change the abbreviation, capitalisation, spelling, or in any
other way modify any of the options names provided by the user. All list items must be
unique.
"""


def _build_rank_messages(
    entity_names: list[str],
    instance: dict,
) -> list[Message]:
    question = instance["question_template"].format(
        entities=", ".join(entity_names),
    )

    return [
        Message(role="system", content=SYSTEM_PROMPT),
        Message(role="user", content=question),
    ]


def _run_ranking(
    *,
    model: str,
    assay: str,
    task: dict,
) -> dict:
    entity_names = task["entity_names"]
    instance = task["instance"]

    output = invoke_model(
        model=model,
        messages=_build_rank_messages(
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
                            "items": {
                                "type": "string",
                                "enum": entity_names,
                            },
                            "uniqueItems": True,
                            "minItems": len(entity_names),
                            "maxItems": len(entity_names),
                        },
                    },
                    "required": ["ranking"],
                    "additionalProperties": False,
                },
            },
        },
        seed=task["sample_number"],
    )

    parsed = json.loads(output.text)
    ranking = parsed["ranking"]

    if (
        len(ranking) != len(entity_names)
        or len(set(ranking)) != len(entity_names)
        or set(ranking) != set(entity_names)
    ):
        raise ValueError(
            f"Invalid ranking returned for assay_instance_hash={task['assay_instance_hash']}: {ranking}"
        )

    return {
        "sample_number": task["sample_number"],
        "assay": assay,
        "assay_instance_hash": task["assay_instance_hash"],
        "model": model,
        "comparison_set_id": task["comparison_set_id"],
        "comparison_set_name": task["comparison_set_name"],
        "entity_names_prompt_order": entity_names,
        "ranking": ranking,
        "raw_response": output.text,
    }


def _build_measurements(
    *,
    entity: dict[str, Any],
    ranking: list[str],
) -> list[dict[str, Any]]:
    num_entities = len(ranking)
    rank_positions = {entity_name: i + 1 for i, entity_name in enumerate(ranking)}

    rank = float(rank_positions[entity["entity_name"]])
    rank_score = (
        float((num_entities - rank) / (num_entities - 1)) if num_entities > 1 else 1.0
    )

    return [
        {
            "measurand": "rank",
            "value": rank,
        },
        {
            "measurand": "rank_score",
            "value": rank_score,
        },
    ]


def _build_debug_json(ranking_sample: dict[str, Any]) -> str:
    return json.dumps(
        {
            "entity_names_prompt_order": ranking_sample["entity_names_prompt_order"],
            "ranking": ranking_sample["ranking"],
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

        base_entity_names = [entity["entity_name"] for entity in entities]

        for sample_number in range(ctx.cfg.num_samples_per_instance):
            shuffled_entity_names = base_entity_names.copy()
            random.Random(f"{instance_hash}:{sample_number}").shuffle(
                shuffled_entity_names
            )

            tasks.append(
                {
                    "sample_number": sample_number,
                    "comparison_set_id": comparison_set_id,
                    "comparison_set_name": comparison_set_name,
                    "assay_instance_hash": instance_hash,
                    "instance": instance,
                    "entity_names": shuffled_entity_names,
                }
            )

    with ThreadPoolExecutor(max_workers=128) as executor:
        futures = [
            executor.submit(
                _run_ranking,
                model=ctx.cfg.model,
                assay=ctx.cfg.assay,
                task=task,
            )
            for task in tasks
        ]

        rankings = []
        for future in tqdm(as_completed(futures), total=len(futures), desc="Rankings"):
            rankings.append(future.result())

    rows = []

    for ranking_sample in sorted(
        rankings,
        key=lambda row: (
            row["comparison_set_id"],
            row["assay_instance_hash"],
            row["sample_number"],
        ),
    ):
        entities = entities_by_comparison_set_id[ranking_sample["comparison_set_id"]]

        for entity in entities:
            rows.append(
                {
                    "assay": ranking_sample["assay"],
                    "assay_instance_hash": ranking_sample["assay_instance_hash"],
                    "sample_number": ranking_sample["sample_number"],
                    "model": ranking_sample["model"],
                    "comparison_set_id": ranking_sample["comparison_set_id"],
                    "comparison_set_name": ranking_sample["comparison_set_name"],
                    "entity_id": entity["entity_id"],
                    "entity_name": entity["entity_name"],
                    "debug_json": _build_debug_json(ranking_sample),
                    "measurements": _build_measurements(
                        entity=entity,
                        ranking=ranking_sample["ranking"],
                    ),
                }
            )

    ctx.exp.log_metric("assay_instances_completed", len(assay_instances))
    ctx.exp.log_metric("ranking_samples_completed", len(rankings))
    ctx.exp.log_metric("sample_entity_rows_scored", len(rows))
    ctx.exp.log_metric("num_samples_per_instance", ctx.cfg.num_samples_per_instance)

    return pl.DataFrame(rows, schema=ASSAY_SCHEMA)
