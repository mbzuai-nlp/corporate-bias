import polars as pl

ENTITY_SCHEMA = {
    "id": pl.String,  # key
    "name": pl.String,
    "aliases": pl.List(pl.String),
    "source": pl.String,
    "facets": pl.List(pl.Struct({"dimension": pl.Utf8, "value": pl.Utf8})),
}

CLAIM_SCHEMA = {
    "predicate": pl.String,  # key
    "subject_id": pl.String,  # key
    "subject_name": pl.String,
    "object_id": pl.String,  # key
    "object_name": pl.String,
    "source": pl.String,
}

COMPARISON_SET_SCHEMA = {
    "id": pl.String,  # key
    "name": pl.String,
    "entity_id": pl.String,  # key
    "entity_name": pl.String,
}
