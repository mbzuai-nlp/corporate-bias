from dataclasses import dataclass
from typing import Literal, Mapping, Protocol
import logging
from pathlib import Path
import polars as pl
from dvclive import Live
import json
from concurrent.futures import ThreadPoolExecutor
from tqdm.auto import tqdm

from src.model import Model, Message, invoke_model
from src.data.model import ASSAY_SCHEMA


Assay = Literal["head-to-head", "rank"]


@dataclass(frozen=True)
class Config:
    save: str
    assay: Assay
    model: Model


@dataclass
class RuntimeContext:
    cfg: Config
    exp: Live
    db: Mapping[str, pl.DataFrame]


class AssayDelegate(Protocol):
    def __call__(self, ctx: RuntimeContext) -> pl.DataFrame: ...


def _save_assay_df(df: pl.DataFrame, save_path: str) -> None:
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)


def run_head_to_head(ctx: RuntimeContext) -> pl.DataFrame:
    def build_messages(
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

    def run_preference(task: dict) -> dict:
        left_entity_name = task["left_entity_name"]
        right_entity_name = task["right_entity_name"]
        instance = task["instance"]

        output = invoke_model(
            model=ctx.cfg.model,
            messages=build_messages(
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
                            "preferred": {"type": "string"},
                            "reason": {"type": "string"},
                        },
                        "required": ["preferred", "reason"],
                        "additionalProperties": False,
                    },
                },
            },
            plugins=[{"id": "response-healing"}],
        )

        parsed = json.loads(output.text)
        preferred_entity_name = parsed["preferred"]
        non_preferred_entity_name = (
            right_entity_name
            if preferred_entity_name == left_entity_name
            else left_entity_name
        )

        return {
            "assay": ctx.cfg.assay,
            "assay_instance_hash": task["assay_instance_hash"],
            "model": ctx.cfg.model,
            "comparison_set_id": task["comparison_set_id"],
            "comparison_set_name": task["comparison_set_name"],
            "left_entity_id": task["left_entity_id"],
            "left_entity_name": left_entity_name,
            "right_entity_id": task["right_entity_id"],
            "right_entity_name": right_entity_name,
            "preferred_entity_name": preferred_entity_name,
            "non_preferred_entity_name": non_preferred_entity_name,
            "reason": parsed["reason"],
        }

    comparison_set_df = ctx.db["comparison_set"]
    comparison_set_assay_instance_df = ctx.db["comparison_set_assay_instance"]

    assay_instances = list(
        comparison_set_assay_instance_df
        .filter(pl.col("assay") == ctx.cfg.assay)
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

        entities = (
            comparison_set_df
            .filter(pl.col("id") == comparison_set_id)
            .select(
                pl.col("entities").list.eval(
                    pl.element().sort_by(pl.element().struct.field("entity_id"))
                )
            )
            .to_series()
            .item()
        )

        entities_by_instance[instance_hash] = entities

        tasks.extend(
            {
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
                executor.map(run_preference, tasks),
                total=len(tasks),
                desc="Preferences",
            )
        )

    wins: dict[tuple[str, str], int] = {}

    for preference in preferences:
        key = (
            preference["assay_instance_hash"],
            preference["preferred_entity_name"],
        )
        wins[key] = wins.get(key, 0) + 1

    rows = []
    for assay_instance in assay_instances:
        instance_hash = assay_instance["instance_hash"]
        comparison_set_id = assay_instance["comparison_set_id"]
        comparison_set_name = assay_instance["comparison_set_name"]
        entities = entities_by_instance[instance_hash]

        rows.extend(
            {
                "assay": ctx.cfg.assay,
                "assay_instance_hash": instance_hash,
                "model": ctx.cfg.model,
                "comparison_set_id": comparison_set_id,
                "comparison_set_name": comparison_set_name,
                "entity_id": entity["entity_id"],
                "entity_name": entity["entity_name"],
                "result": [
                    {
                        "estimand": "num_wins",
                        "value": str(wins.get((instance_hash, entity["entity_name"]), 0)),
                    }
                ],
            }
            for entity in entities
        )

    ctx.exp.log_metric("assay_instances_completed", len(assay_instances))
    ctx.exp.log_metric("entities_scored", len(rows))

    return pl.DataFrame(rows, schema=ASSAY_SCHEMA)


def run_rank(ctx: RuntimeContext) -> pl.DataFrame:
    def build_messages(
        entity_names: list[str],
        instance: dict,
    ) -> list[Message]:
        entity_list = "\n".join(f"{i + 1}. {entity_name}" for i, entity_name in enumerate(entity_names))
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
        )

        parsed = json.loads(output.text)
        ranking = parsed["ranking"]

        if sorted(ranking) != sorted(entity_names) or len(ranking) != len(entity_names):
            raise ValueError(
                f"Invalid ranking returned for assay_instance_hash={task['assay_instance_hash']}: {ranking}"
            )

        return {
            "assay": ctx.cfg.assay,
            "assay_instance_hash": task["assay_instance_hash"],
            "model": ctx.cfg.model,
            "comparison_set_id": task["comparison_set_id"],
            "comparison_set_name": task["comparison_set_name"],
            "ranking": ranking,
            "reason": parsed["reason"],
        }

    comparison_set_df = ctx.db["comparison_set"]
    comparison_set_assay_instance_df = ctx.db["comparison_set_assay_instance"]

    assay_instances = list(
        comparison_set_assay_instance_df
        .filter(pl.col("assay") == ctx.cfg.assay)
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

        entities = (
            comparison_set_df
            .filter(pl.col("id") == comparison_set_id)
            .select(
                pl.col("entities").list.eval(
                    pl.element().sort_by(pl.element().struct.field("entity_id"))
                )
            )
            .to_series()
            .item()
        )

        entities_by_instance[instance_hash] = entities

        tasks.append(
            {
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

    ranking_by_instance = {
        ranking["assay_instance_hash"]: ranking["ranking"]
        for ranking in rankings
    }

    rows = []
    for assay_instance in assay_instances:
        instance_hash = assay_instance["instance_hash"]
        comparison_set_id = assay_instance["comparison_set_id"]
        comparison_set_name = assay_instance["comparison_set_name"]
        entities = entities_by_instance[instance_hash]
        rank_positions = {
            entity_name: i + 1
            for i, entity_name in enumerate(ranking_by_instance[instance_hash])
        }

        rows.extend(
            {
                "assay": ctx.cfg.assay,
                "assay_instance_hash": instance_hash,
                "model": ctx.cfg.model,
                "comparison_set_id": comparison_set_id,
                "comparison_set_name": comparison_set_name,
                "entity_id": entity["entity_id"],
                "entity_name": entity["entity_name"],
                "result": [
                    {
                        "estimand": "rank",
                        "value": str(rank_positions[entity["entity_name"]]),
                    }
                ],
            }
            for entity in entities
        )

    ctx.exp.log_metric("assay_instances_completed", len(assay_instances))
    ctx.exp.log_metric("entities_scored", len(rows))

    return pl.DataFrame(rows, schema=ASSAY_SCHEMA)


ASSAY_DELEGATES: Mapping[Assay, AssayDelegate] = {
    "head-to-head": run_head_to_head,
    "rank": run_rank,
}


def assay_model(ctx: RuntimeContext) -> None:
    assay_delegate = ASSAY_DELEGATES[ctx.cfg.assay]

    logging.info(f"Running assay={ctx.cfg.assay} model={ctx.cfg.model}.")

    assay_df = assay_delegate(ctx)

    _save_assay_df(assay_df, ctx.cfg.save)

    logging.info(f"Saved assay results to {ctx.cfg.save}.")