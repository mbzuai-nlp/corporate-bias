import json
from dataclasses import dataclass
from pathlib import Path

import polars as pl


@dataclass
class Db:
    entity: pl.DataFrame
    alias_map: dict[str, str] # alias: entity
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


# Each row should correspond to one invocation of the model
ASSAY_SCHEMA = {
    "assay": pl.String,
    "prompt_template": pl.String,
    "model": pl.String,
    "comparison_set": pl.String,
    "debug_json": pl.String,
}


def load_db(db_dir: Path) -> Db:
    comparison_set_entity_rows = []
    comparison_set_prompt_template_rows = []

    aliases = json.loads((db_dir / "entity_aliases.json").read_text())
    alias_map =  {
        alias: entity
        for entity, aliases in aliases.items()
        for alias in aliases
    }

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
    ).filter(pl.col("comparison_set").is_in(["home-video-game-consoles", "search-engine"]))

    comparison_set_prompt_template = pl.DataFrame(
        comparison_set_prompt_template_rows,
        schema=PROMPT_TEMPLATE_SCHEMA,
    ).filter(pl.col("comparison_set").is_in(["home-video-game-consoles", "search-engine"]))

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
        alias_map=alias_map,
        prompt_template=comparison_set_prompt_template,
        model=model,
    )
