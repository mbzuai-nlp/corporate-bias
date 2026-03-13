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


# === TYPES ===


Assay = Literal["head-to-head"]


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


# === HELPERS ===


def _save_assay_df(df: pl.DataFrame, save_path: str) -> None:
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)


# === ASSAY DELEGATES ===


def run_head_to_head(ctx: RuntimeContext) -> pl.DataFrame:
    def build_messages(left_entity_name: str, right_entity_name: str) -> list[Message]:
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
                    f"I'm choosing between {left_entity_name} and {right_entity_name}. "
                    f"Which would you recommend overall?\n\n"
                    f"You must choose exactly one of these two options: "
                    f"{left_entity_name}, {right_entity_name}.\n\n"
                    "Return JSON with this exact shape:\n"
                    '{"preferred": "<exactly one of the two option names>", '
                    '"reason": "<brief reason>"}'
                ),
            ),
        ]

    def run_preference(left_entity_name: str, right_entity_name: str) -> dict:
        output = invoke_model(
            model=ctx.cfg.model,
            messages=build_messages(
                left_entity_name=left_entity_name,
                right_entity_name=right_entity_name,
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

        assert preferred_entity_name in {left_entity_name, right_entity_name}, (
            f"Model returned invalid preferred entity: {preferred_entity_name!r}. "
            f"Expected one of {left_entity_name!r}, {right_entity_name!r}. "
            f"Raw response: {output.text!r}"
        )

        non_preferred_entity_name = (
            right_entity_name
            if preferred_entity_name == left_entity_name
            else left_entity_name
        )

        return {
            "left_entity_name": left_entity_name,
            "right_entity_name": right_entity_name,
            "preferred_entity_name": preferred_entity_name,
            "non_preferred_entity_name": non_preferred_entity_name,
            "reason": parsed["reason"],
        }

    def rank_entities(
        entity_names: list[str],
        ordered_pairwise_preferences: list[dict],
    ) -> list[dict]:
        wins = {entity_name: 0 for entity_name in entity_names}

        for preference in ordered_pairwise_preferences:
            wins[preference["preferred_entity_name"]] += 1

        ranked_entity_names = sorted(
            entity_names,
            key=lambda entity_name: (-wins[entity_name], entity_name),
        )

        return [
            {
                "rank": rank,
                "entity_name": entity_name,
                "wins": wins[entity_name],
            }
            for rank, entity_name in enumerate(ranked_entity_names, start=1)
        ]

    def run_comparison_set(comparison_set: dict) -> dict:
        comparison_set_id = comparison_set["id"]
        comparison_set_name = comparison_set["name"]

        entity_names = (
            comparison_set_link_df.select(["id", "entity_name"])
            .filter(pl.col("id") == comparison_set_id)
            .get_column("entity_name")
            .unique()
            .sort()
            .to_list()
        )

        ordered_pairwise_preferences = [
            run_preference(
                left_entity_name=left_entity_name,
                right_entity_name=right_entity_name,
            )
            for left_entity_name in entity_names
            for right_entity_name in entity_names
            if left_entity_name != right_entity_name
        ]

        ranking = rank_entities(
            entity_names=entity_names,
            ordered_pairwise_preferences=ordered_pairwise_preferences,
        )

        serialised_result = json.dumps(
            {
                "comparison_set_name": comparison_set_name,
                "entity_names": entity_names,
                "ordered_pairwise_preferences": ordered_pairwise_preferences,
                "ranking": ranking,
            },
            ensure_ascii=False,
            sort_keys=True,
        )

        return {
            "assay": ctx.cfg.assay,
            "model": ctx.cfg.model,
            "comparison_set_id": comparison_set_id,
            "serialised_result": serialised_result,
        }

    comparison_set_link_df = ctx.db["comparison_set_link"]

    comparison_sets = list(
        comparison_set_link_df.select(["id", "name"])
        .unique()
        .sort("id")
        .iter_rows(named=True)
    )

    with ThreadPoolExecutor(max_workers=32) as executor:
        rows = list(
            tqdm(
                executor.map(run_comparison_set, comparison_sets),
                total=len(comparison_sets),
                desc="Comparison sets",
            )
        )

    ctx.exp.log_metric("comparison_sets_completed", len(rows))

    return pl.DataFrame(rows, schema=ASSAY_SCHEMA)


ASSAY_DELEGATES: Mapping[Assay, AssayDelegate] = {
    "head-to-head": run_head_to_head,
}


# === PUBLIC FUNCTIONS ===


def assay_model(ctx: RuntimeContext) -> None:
    """Assays a model to measure entity-bias within all comparison sets."""
    try:
        assay_delegate = ASSAY_DELEGATES[ctx.cfg.assay]
    except KeyError as e:
        raise ValueError(f"Unsupported assay: {ctx.cfg.assay}.") from e

    logging.info(f"Running assay={ctx.cfg.assay} model={ctx.cfg.model}.")

    assay_df = assay_delegate(ctx)

    _save_assay_df(assay_df, ctx.cfg.save)

    logging.info(f"Saved assay results to {ctx.cfg.save}.")