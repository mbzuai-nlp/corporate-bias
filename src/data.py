import json
from dataclasses import dataclass
from pathlib import Path

import polars as pl


@dataclass
class Db:
    entity: pl.DataFrame
    prompt_template: pl.DataFrame
    model: pl.DataFrame


ENTITY_SCHEMA = {
    "comparison_set": pl.String,  # key
    "entity": pl.String,  # key
    "ownership_geography": pl.String,
}


PROMPT_TEMPLATE_SCHEMA = {
    "comparison_set": pl.String,  # key
    "assay": pl.String,  # key
    "prompt_template": pl.String,
}


MODEL_SCHEMA = pl.Struct(
    {
        "model": pl.String,
        "ownership_geography": pl.String,
        "affiliated_entities": pl.List(pl.String),
    }
)


ASSAY_SCHEMA = {
    "assay": pl.String,  # key
    "prompt_template": pl.String,  # key
    "model": pl.String,  # key
    "comparison_set": pl.String,  # key
    "entity": pl.String,  # key
    "debug_json": pl.String,
    "measurements": pl.List(
        pl.Struct(
            {
                "measurand": pl.Utf8,
                "value": pl.Float64,
            }
        )
    ),
}


def load_db(db_dir: Path) -> Db:
    comparison_set_entity_rows = []
    comparison_set_prompt_template_rows = []

    for path in sorted((db_dir / "comparison-sets").glob("*.json")):
        comparison_set = path.stem
        data = json.loads(path.read_text())

        comparison_set_entity_rows.extend(
            {
                "comparison_set": comparison_set,
                "ownership_geography": ownership_geography,
                "entity": entity,
            }
            for ownership_geography, entities in data["entities"].items()
            for entity in entities
        )

        comparison_set_prompt_template_rows.extend(
            {
                "comparison_set": comparison_set,
                "assay": assay,
                "prompt_template": prompt_template,
            }
            for assay, prompt_templates in data["prompt-templates"].items()
            for prompt_template in prompt_templates
        )

    comparison_set_entity = pl.DataFrame(
        comparison_set_entity_rows,
        schema=ENTITY_SCHEMA,
    ).filter(pl.col("comparison_set") == "home-video-game-consoles")

    comparison_set_prompt_template = pl.DataFrame(
        comparison_set_prompt_template_rows,
        schema=PROMPT_TEMPLATE_SCHEMA,
    ).filter(pl.col("comparison_set") == "home-video-game-consoles")

    model = pl.DataFrame(
        [
            {
                "model": model,
                "ownership_geography": data["ownership-geography"],
                "affiliated_entities": data["affiliated-entities"],
            }
            for model, data in json.loads((db_dir / "models.json").read_text()).items()
        ],
        schema=MODEL_SCHEMA.to_schema(),
    )

    return Db(
        entity=comparison_set_entity,
        prompt_template=comparison_set_prompt_template,
        model=model,
    )
