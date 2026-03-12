from pathlib import Path
import argparse
import json
import polars as pl
from typing import Mapping
from dotenv import load_dotenv

from pipelines.utils import configure_logging
from src.model import invoke_model


configure_logging()
load_dotenv()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--assay",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--db",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--save-path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--exp",
        type=str,
        required=True,
    )
    return parser.parse_args()


def main():
    """Runs a specified assay of a model."""

    args = parse_args()

    db_dir = Path(args.db).resolve()
    save_path = Path(args.save_path).resolve()
    exp_dir = Path(args.exp).resolve()


if __name__ == "__main__":
    main()
