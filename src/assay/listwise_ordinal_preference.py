import json
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Tuple
import polars as pl
from tqdm.auto import tqdm

from src.assay.common import RuntimeContext
from src.data import ASSAY_SCHEMA
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

    # For each comparison_set, sample n permutations (n = number of entities)
    sampled_queries = []
    for row in entities_agg.iter_rows(named=True):
        comparison_set = row["comparison_set"]
        entities = row["entity"]
        n = len(entities)

        # Sample n random permutations (with replacement)
        for _ in range(n):
            shuffled_entities = random.sample(entities, k=len(entities))
            sampled_queries.append(
                {
                    "comparison_set": comparison_set,
                    "entities": shuffled_entities,
                }
            )

    sampled_df = pl.DataFrame(sampled_queries)

    queries_df = (
        sampled_df.join(prompt_template_df, on="comparison_set", how="inner")
        .with_columns(
            pl.col("prompt_template")
            .str.replace_all(
                "{entities}",
                pl.col("entities").list.join(", "),
                literal=True,
            )
            .alias("query")
        )
        .select("comparison_set", "prompt_template", "query", "entities")
    )

    return queries_df


def _get_ranking(model: str, query: str, entities: list[str]) -> Tuple[list[str], Any]:
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
                "name": "rank_entities",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "ranking": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "enum": entities,
                            },
                            "uniqueItems": True,
                            "minItems": len(entities),
                            "maxItems": len(entities),
                        },
                    },
                    "required": ["ranking"],
                    "additionalProperties": False,
                },
            },
        },
    )

    parsed = json.loads(output.text)
    ranking = parsed["ranking"]

    if (
        len(ranking) != len(entities)
        or len(set(ranking)) != len(entities)
        or set(ranking) != set(entities)
    ):
        raise ValueError(f"Invalid ranking returned: {ranking}")

    return ranking, output.raw


def run_assay(ctx: RuntimeContext) -> pl.DataFrame:
    entity_df = ctx.assay_db.entity
    prompt_template_df = ctx.assay_db.prompt_template

    queries_df = _construct_queries(entity_df, prompt_template_df)
    query_rows = list(queries_df.iter_rows(named=True))

    # Query model
    rankings = [None] * len(query_rows)
    raw_responses = [None] * len(query_rows)
    with ThreadPoolExecutor(max_workers=128) as executor:
        future_to_idx = {
            executor.submit(
                _get_ranking,
                model=ctx.cfg.model,
                query=row["query"],
                entities=row["entities"],
            ): i
            for i, row in enumerate(query_rows)
        }

        for future in tqdm(
            as_completed(future_to_idx),
            total=len(future_to_idx),
            desc="Queries",
        ):
            i = future_to_idx[future]
            result = future.result()
            rankings[i] = result[0]
            raw_responses[i] = result[1]

    # Construct results
    results_df = queries_df.with_columns(
        pl.lit(ctx.cfg.assay).alias("assay"),
        pl.lit(ctx.cfg.model).alias("model"),
        pl.Series("rankings", rankings),
        pl.Series(
            "debug_json",
            [
                json.dumps({"raw_response": r}, ensure_ascii=False, default=str)
                for r in raw_responses
            ],
        ),
    ).select(
        "assay",
        "prompt_template",
        "model",
        "comparison_set",
        "entities",
        "rankings",
        "debug_json",
    )

    ctx.exp.log_metric("total_queries_run", queries_df.height)

    return results_df
