from pathlib import Path
import argparse
import importlib
import json
import logging
from typing import Mapping

import polars as pl
from dotenv import load_dotenv
from dvclive import Live

from pipelines.utils import configure_logging, silence_superfluous_warnings
from src.data.model import (
    CLAIM_SCHEMA,
    COMPARISON_SET_ASSAY_INSTANCE_SCHEMA,
    COMPARISON_SET_SCHEMA,
    ENTITY_SCHEMA,
)
from src.assay.common import Config, RuntimeContext, AssayDelegate, save_assay_df


configure_logging()
silence_superfluous_warnings()
load_dotenv()


ASSAY_MODULES: Mapping[str, tuple[str, str]] = {
    "head-to-head": ("src.assay.head_to_head", "run_head_to_head"),
    "rank": ("src.assay.rank", "run_rank"),
    "consideration-set": ("src.assay.consideration_set", "run_consideration_set"),
    "describe-sentiment": ("src.assay.describe_sentiment", "run_describe_sentiment"),
    "forced-selection": ("src.assay.forced_selection", "run_forced_selection"),
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--assay", type=str, required=True, choices=ASSAY_MODULES.keys())
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--db", type=str, required=True)
    parser.add_argument("--save-path", type=str, required=True)
    parser.add_argument("--exp", type=str, required=True)
    parser.add_argument("--samples-per-instance", type=int, required=True)
    return parser.parse_args()


def load_db(db_dir: Path) -> Mapping[str, pl.DataFrame]:
    claim_df = pl.read_parquet(
        db_dir / "claim.parquet",
        schema=CLAIM_SCHEMA,
        missing_columns="insert",
    )

    comparison_set_assay_instance_df = (
        pl.read_parquet(db_dir / "comparison_set_assay_instance.parquet")
        .with_columns(
            pl.col("instance_json")
            .map_elements(json.loads, return_dtype=pl.Object)
            .alias("instance")
        )
        .select(
            "comparison_set_id",
            "comparison_set_name",
            "assay",
            "instance_hash",
            "instance",
        )
        .cast(COMPARISON_SET_ASSAY_INSTANCE_SCHEMA)
    )

    comparison_set_df = pl.read_parquet(
        db_dir / "comparison_set.parquet",
        schema=COMPARISON_SET_SCHEMA,
        missing_columns="insert",
    )

    entity_df = pl.read_parquet(
        db_dir / "entity.parquet",
        schema=ENTITY_SCHEMA,
        missing_columns="insert",
    )

    return {
        "claim": claim_df,
        "comparison_set_assay_instance": comparison_set_assay_instance_df,
        "comparison_set": comparison_set_df,
        "entity": entity_df,
    }


def load_assay_delegate(assay: str) -> AssayDelegate:
    try:
        module_path, fn_name = ASSAY_MODULES[assay]
    except KeyError as exc:
        raise ValueError(f"Unsupported assay: {assay}") from exc

    module = importlib.import_module(module_path)
    delegate = getattr(module, fn_name)
    return delegate


def assay_model(ctx: RuntimeContext) -> None:
    assay_delegate = load_assay_delegate(ctx.cfg.assay)

    logging.info(
        f"Running assay={ctx.cfg.assay} model={ctx.cfg.model} "
        f"num_samples_per_instance={ctx.cfg.num_samples_per_instance}."
    )

    assay_df = assay_delegate(ctx)
    save_assay_df(assay_df, ctx.cfg.save)

    logging.info(f"Saved assay results to {ctx.cfg.save}.")


def main():
    """Runs a specified assay of a model against all comparison sets."""
    args = parse_args()

    db_dir = Path(args.db).resolve()
    save_path = Path(args.save_path).resolve()
    exp_dir = Path(args.exp).resolve()

    db = load_db(db_dir=db_dir)

    with Live(dir=exp_dir) as exp:
        ctx = RuntimeContext(
            cfg=Config(
                save=save_path,
                assay=args.assay,
                model=args.model,
                num_samples_per_instance=args.samples_per_instance,
            ),
            exp=exp,
            db=db,
        )
        assay_model(ctx)


if __name__ == "__main__":
    main()