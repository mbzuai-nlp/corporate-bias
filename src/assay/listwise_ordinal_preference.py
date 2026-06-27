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


def _construct_queries(
    entity_df: pl.DataFrame, prompt_template_df: pl.DataFrame
) -> pl.DataFrame:
    entities_agg = entity_df.group_by("comparison_set").agg(pl.col("entity"))

    queries_df = (
        prompt_template_df.join(entities_agg, on="comparison_set", how="inner")
        .with_columns(
            # Format the query by replacing {entities} with the comma-separated list
            pl.col("prompt_template")
            .str.replace_all(
                "{entities}", pl.col("entity").list.join(", "), literal=True
            )
            .alias("query")
        )
        .select(
            "comparison_set",
            "prompt_template",
            "query",
        )
    )

    return queries_df


def run_assay(ctx: RuntimeContext) -> pl.DataFrame:
    entity_df = ctx.assay_db.entity
    prompt_template_df = ctx.assay_db.prompt_template

    queries_df = _construct_queries(entity_df, prompt_template_df)

    query_rows = list(queries_df.iter_rows(named=True))
    rankings = [None] * len(query_rows)
    raw_responses = [None] * len(query_rows)
