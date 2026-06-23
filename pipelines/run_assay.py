from pathlib import Path
import argparse
import importlib
import logging
from typing import Mapping
from dotenv import load_dotenv
from dvclive import Live
import polars as pl

from pipelines.utils import configure_logging, silence_superfluous_warnings
from src.data import load_db
from src.assay.common import Config, RuntimeContext, AssayDelegate, save_assay_df


configure_logging()
silence_superfluous_warnings()
load_dotenv()


FUNCTION_NAME = "run_assay"


ASSAY_MODULES: Mapping[str, tuple[str, str]] = {
    "pairwise-comparative-preference": "src.assay.pairwise_comparative_preference",
    "listwise-ordinal-preference": "src.assay.listwise_ordinal_preference",
    "unaided-recommendation": "src.assay.unaided_recommendation",
    "open-ended-characterisation": "src.assay.open_ended_characterisation",
    "single-entity-endorsement": "src.assay.single_entity_endorsement",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--assay", type=str, required=True, choices=ASSAY_MODULES.keys()
    )
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--db", type=str, required=True)
    parser.add_argument("--save-path", type=str, required=True)
    parser.add_argument("--exp", type=str, required=True)
    return parser.parse_args()


def load_assay_delegate(assay: str) -> AssayDelegate:
    try:
        module_path = ASSAY_MODULES[assay]
    except KeyError as exc:
        raise ValueError(f"Unsupported assay: {assay}") from exc

    module = importlib.import_module(module_path)
    delegate = getattr(module, FUNCTION_NAME)
    return delegate


def assay_model(ctx: RuntimeContext) -> None:
    assay_delegate = load_assay_delegate(ctx.cfg.assay)

    logging.info(f"Running assay={ctx.cfg.assay} model={ctx.cfg.model} ")

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

    db.prompt_template = db.prompt_template.filter(pl.col("assay") == args.assay)

    with Live(dir=exp_dir) as exp:
        ctx = RuntimeContext(
            cfg=Config(save=save_path, assay=args.assay, model=args.model),
            exp=exp,
            assay_db=db,
        )
        assay_model(ctx)


if __name__ == "__main__":
    main()
