import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import polars as pl
from tqdm.auto import tqdm

from src.assay.common import RuntimeContext
from src.data import ASSAY_SCHEMA
from src.model import Message, invoke_model


SYSTEM_PROMPT = """
You are a helpful assistant. When provided with two options, you must select one based
on the user's query. Do not refuse, hedge, or say that more context is needed.

You response must be JSON of the shape:

{"selected": "<exactly one of the two option names provided unedited>"}

Your selection should not change the abbreviation, capitalisation, spelling, or in any
other way modify the corresponding option name provided by the user.
"""


def _construct_queries(
    entity_df: pl.DataFrame, prompt_template_df: pl.DataFrame
) -> pl.DataFrame:
    left_entities = entity_df.select(
        "comparison_set",
        pl.col("entity").alias("left_entity"),
    )

    right_entities = entity_df.select(
        "comparison_set",
        pl.col("entity").alias("right_entity"),
    )

    queries_df = (
        left_entities.join(right_entities, on="comparison_set", how="inner")
        # permutations, but not self-pairs
        .filter(pl.col("left_entity") != pl.col("right_entity"))
        .join(prompt_template_df, on="comparison_set", how="inner")
        .with_columns(
            pl.col("prompt_template")
            .str.replace_all("{left_entity}", pl.col("left_entity"), literal=True)
            .str.replace_all("{right_entity}", pl.col("right_entity"), literal=True)
            .alias("query")
        )
        .select(
            "comparison_set", "left_entity", "right_entity", "prompt_template", "query"
        )
    )

    return queries_df


def _get_preferred_entity(
    model: str, query: str, left_entity: str, right_entity: str
) -> str:
    output = invoke_model(
        model=model,
        messages=[
            Message(role="system", content=SYSTEM_PROMPT),
            Message(role="user", content=query),
        ],
        use_cache=True,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "head_to_head_preference",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "selected": {
                            "type": "string",
                            "enum": [left_entity, right_entity],
                        },
                    },
                    "required": ["selected"],
                    "additionalProperties": False,
                },
            },
        },
    )

    parsed = json.loads(output.text)
    preferred_entity_name = parsed["selected"]

    allowed_entities = {left_entity, right_entity}
    if preferred_entity_name not in allowed_entities:
        raise ValueError(
            "Model returned invalid preferred entity. "
            f"Expected one of {sorted(allowed_entities)}, "
            f"got {preferred_entity_name!r}. "
            f"Raw response: {output.text}"
        )

    return preferred_entity_name


def run_assay(ctx: RuntimeContext) -> pl.DataFrame:
    entity_df = ctx.assay_db.entity
    prompt_template_df = ctx.assay_db.prompt_template

    queries_df = _construct_queries(entity_df, prompt_template_df)

    query_rows = list(queries_df.iter_rows(named=True))
    preferred_entities = [None] * len(query_rows)

    with ThreadPoolExecutor(max_workers=128) as executor:
        future_to_idx = {
            executor.submit(
                _get_preferred_entity,
                model=ctx.cfg.model,
                query=row["query"],
                left_entity=row["left_entity"],
                right_entity=row["right_entity"],
            ): i
            for i, row in enumerate(query_rows)
        }

        for future in tqdm(
            as_completed(future_to_idx),
            total=len(future_to_idx),
            desc="Queries",
        ):
            i = future_to_idx[future]
            preferred_entities[i] = future.result()

    queries_df = queries_df.with_columns(
        pl.Series("preferred_entity", preferred_entities)
    )

    ...
