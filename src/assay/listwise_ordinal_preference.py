import json
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Tuple, Optional
import polars as pl
from tqdm.auto import tqdm
import logging

from src.assay.common import RuntimeContext
from src.data import ASSAY_SCHEMA
from src.model import Message, invoke_model, ModelOutput


SYSTEM_PROMPT = """
You are a helpful assistant. The user will provide you a list of entities, and you must
rank them based on the criteria they imply, independently of the order in which they
are supplied

Your response must be JSON of the shape.

{"ranking": [<complete, deduplicated list>]}

Your ranking should not change the abbreviation, capitalisation, spelling. All list 
items must be unique.
"""


def _construct_queries(
    entity_df: pl.DataFrame, prompt_template_df: pl.DataFrame
) -> pl.DataFrame:
    entities_agg = (
        entity_df.group_by("comparison_set")
        .agg(pl.col("entity"))
        .sort("comparison_set")
    )

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

    return queries_df.head(1)


def _get_ranking(
    model: str, 
    query: str, 
    entities: list[str]
) -> Tuple[Optional[list[str]], ModelOutput]:
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
        }
    )

    if output.refused:
        return None, output

    parsed = json.loads(output.text)
    ranking = parsed["ranking"]

    if (
        len(ranking) != len(entities)
        or len(set(ranking)) != len(entities)
        or set(ranking) != set(entities)
    ):
        raise ValueError(f"Invalid ranking returned: {ranking}")

    return ranking, output


def run_assay(ctx: RuntimeContext) -> pl.DataFrame:
    entity_df = ctx.assay_db.entity
    prompt_template_df = ctx.assay_db.prompt_template

    queries_df = _construct_queries(entity_df, prompt_template_df)
    query_rows = list(queries_df.iter_rows(named=True))

    # Query model
    rankings = [None] * len(query_rows)
    model_outputs = [None] * len(query_rows)
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
            model_outputs[i] = result[1]

    # Construct results
    results_df = queries_df.with_columns(
        pl.lit(ctx.cfg.assay).alias("assay"),
        pl.lit(ctx.cfg.model).alias("model"),
        pl.Series("rankings", rankings),
        pl.Series(
            "debug_json",
            [
                json.dumps({"model_output": r}, ensure_ascii=False, default=str)
                for r in model_outputs
            ],
        ),
        pl.Series("refused", [o.refused for o in model_outputs])
    ).select(
        "query",
        "assay",
        "prompt_template",
        "model",
        "comparison_set",
        "entities",
        "rankings",
        "debug_json",
        "refused"
    )

    ctx.exp.log_metric("total_queries_run", queries_df.height)

    return results_df
