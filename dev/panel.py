import polars as pl
import panel as pn
import hvplot.polars  # registers .hvplot on Polars DataFrames

pn.extension()

gpt_path = "/home/harry/code/corporate-bias/data/assays/forced-selection/gpt5.parquet"
gemini_path = "/home/harry/code/corporate-bias/data/assays/forced-selection/gemini.parquet"

PLOT_SPECS = [
    ("steered_away_rate__gpt5_mean", "Steered Away Rate", "GPT-5 judge"),
    ("steered_away_rate__gemini_mean", "Steered Away Rate", "Gemini 2.5 Flash judge"),
    ("steered_to_rate__gpt5_mean", "Steered To Rate", "GPT-5 judge"),
    ("steered_to_rate__gemini_mean", "Steered To Rate", "Gemini 2.5 Flash judge"),
]

MODEL_PATHS = {
    "GPT-5": gpt_path,
    "Gemini 2.5 Flash": gemini_path,
}


def load_rate(path: str, estimand: str) -> pl.DataFrame:
    df = pl.read_parquet(path)

    return (
        df.explode("result")
        .with_columns(
            pl.col("result").struct.field("estimand").alias("estimand"),
            pl.col("result").struct.field("value").alias("value"),
        )
        .filter(pl.col("estimand") == estimand)
        .with_columns(pl.col("value").cast(pl.Float64).alias("rate"))
        .group_by("entity_name")
        .agg(pl.col("rate").mean().alias("rate"))
        .rename({"entity_name": "entity"})
    )


def build_long_df() -> pl.DataFrame:
    parts = []

    for output_model, path in MODEL_PATHS.items():
        for estimand, metric, judge in PLOT_SPECS:
            part = (
                load_rate(path, estimand)
                .with_columns(
                    pl.lit(output_model).alias("output_model"),
                    pl.lit(metric).alias("metric"),
                    pl.lit(judge).alias("judge"),
                )
                .select("output_model", "judge", "metric", "entity", "rate")
            )
            parts.append(part)

    return pl.concat(parts)


df = build_long_df()

model_options = sorted(df["output_model"].unique().to_list())
metric_options = sorted(df["metric"].unique().to_list())
judge_options = sorted(df["judge"].unique().to_list())

comparison = pn.widgets.Select(
    name="Comparison set",
    options=[
        "output_model",
        "judge",
        "metric",
        "model + judge",
        "model + metric",
        "judge + metric",
    ],
    value="output_model",
)

models = pn.widgets.MultiChoice(
    name="Models",
    options=model_options,
    value=model_options,
)

metrics = pn.widgets.MultiChoice(
    name="Metrics",
    options=metric_options,
    value=metric_options,
)

judges = pn.widgets.MultiChoice(
    name="Judges",
    options=judge_options,
    value=judge_options,
)

top_n = pn.widgets.IntSlider(name="Top N entities", start=5, end=50, step=1, value=20)


@pn.depends(
    comparison.param.value,
    models.param.value,
    metrics.param.value,
    judges.param.value,
    top_n.param.value,
)
def plot_view(comparison_value, models_value, metrics_value, judges_value, top_n_value):
    filtered = df.filter(
        pl.col("output_model").is_in(models_value)
        & pl.col("metric").is_in(metrics_value)
        & pl.col("judge").is_in(judges_value)
    )

    if filtered.is_empty():
        return pn.pane.Markdown("No rows match the current filters.")

    if comparison_value == "output_model":
        filtered = filtered.with_columns(
            pl.col("output_model").alias("comparison_label")
        )
    elif comparison_value == "judge":
        filtered = filtered.with_columns(
            pl.col("judge").alias("comparison_label")
        )
    elif comparison_value == "metric":
        filtered = filtered.with_columns(
            pl.col("metric").alias("comparison_label")
        )
    elif comparison_value == "model + judge":
        filtered = filtered.with_columns(
            (pl.col("output_model") + " • " + pl.col("judge")).alias("comparison_label")
        )
    elif comparison_value == "model + metric":
        filtered = filtered.with_columns(
            (pl.col("output_model") + " • " + pl.col("metric")).alias("comparison_label")
        )
    elif comparison_value == "judge + metric":
        filtered = filtered.with_columns(
            (pl.col("judge") + " • " + pl.col("metric")).alias("comparison_label")
        )

    # aggregate in case multiple metrics/judges/models collapse into the same label
    plot_df = (
        filtered
        .group_by(["entity", "comparison_label"])
        .agg(pl.col("rate").mean().alias("rate"))
    )

    # choose top entities by max rate across selected comparison labels
    top_entities = (
        plot_df
        .group_by("entity")
        .agg(pl.col("rate").max().alias("max_rate"))
        .sort("max_rate", descending=True)
        .head(top_n_value)
        ["entity"]
        .to_list()
    )

    plot_df = (
        plot_df
        .filter(pl.col("entity").is_in(top_entities))
        .sort(["entity", "comparison_label"])
    )

    return plot_df.hvplot.barh(
        x="entity",
        y="rate",
        by="comparison_label",
        height=700,
        width=1100,
        stacked=False,
        legend="right",
        xlim=(0, 1),
        title="Forced Selection Free-Text Steering Rates",
        hover_cols=["comparison_label"],
    ).opts(invert_axes=True)


app = pn.Row(
    pn.WidgetBox(
        "## Controls",
        comparison,
        models,
        metrics,
        judges,
        top_n,
        width=320,
    ),
    plot_view,
)

app.servable()