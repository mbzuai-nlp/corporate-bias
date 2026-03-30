import math

import panel as pn
import polars as pl
from bokeh.models import ColumnDataSource, FactorRange, HoverTool, Whisker
from bokeh.plotting import figure

pn.extension()

ASSAY_PARQUET_PATH = "data/combined_assays.parquet"
ASSAY_DF = pl.read_parquet(ASSAY_PARQUET_PATH)


def get_comparison_sets(assay: str) -> list[str]:
    return (
        ASSAY_DF
        .filter(pl.col("assay") == assay)
        .select("comparison_set_name")
        .unique()
        .sort("comparison_set_name")
        .get_column("comparison_set_name")
        .to_list()
    )


def get_estimands(assay: str) -> list[str]:
    return (
        ASSAY_DF
        .filter(pl.col("assay") == assay)
        .explode("result")
        .select(pl.col("result").struct.field("estimand").alias("estimand"))
        .unique()
        .sort("estimand")
        .get_column("estimand")
        .to_list()
    )


def family_mean_and_se(
    means: list[float],
    stds: list[float],
    n_per_prompt: int,
) -> tuple[float, float]:
    k = len(means)
    family_mean = sum(means) / k
    within_var = sum(s * s for s in stds) / k

    if k == 1:
        return family_mean, math.sqrt(within_var / n_per_prompt)

    mean_var = sum((m - family_mean) ** 2 for m in means) / (k - 1)
    between_var = max(0.0, mean_var - within_var / n_per_prompt)
    family_se = math.sqrt(between_var / k + within_var / (k * n_per_prompt))
    return family_mean, family_se


def get_plot_df(assay: str, comparison_set_name: str, estimand: str):
    exploded = (
        ASSAY_DF
        .filter(
            (pl.col("assay") == assay)
            & (pl.col("comparison_set_name") == comparison_set_name)
        )
        .explode("result")
        .select(
            "entity_name",
            "model",
            pl.col("result").struct.field("estimand").alias("estimand"),
            pl.col("result").struct.field("num_samples").alias("num_samples"),
            pl.col("result").struct.field("sample_mean").alias("sample_mean"),
            pl.col("result").struct.field("sample_std").alias("sample_std"),
        )
        .filter(pl.col("estimand") == estimand)
        .sort("entity_name", "model")
    )

    rows = []
    for _, g in exploded.group_by("entity_name", "model"):
        mean, se = family_mean_and_se(
            g["sample_mean"].to_list(),
            g["sample_std"].to_list(),
            g["num_samples"][0],
        )
        rows.append(
            {
                "entity_name": g["entity_name"][0],
                "model": g["model"][0],
                "mean": mean,
                "se": se,
            }
        )

    pdf = (
        pl.DataFrame(rows)
        .sort("entity_name", "model")
        .to_pandas()
    )

    pdf["factor"] = list(zip(pdf["entity_name"], pdf["model"]))
    pdf["lower"] = pdf["mean"] - pdf["se"]
    pdf["upper"] = pdf["mean"] + pdf["se"]

    return pdf


def make_bar_plot(pdf, ylabel: str):
    source = ColumnDataSource(pdf)
    factors = pdf["factor"].tolist()

    y_max = pdf["upper"].max()
    y_min = min(0, pdf["lower"].min())
    pad = 0.05 * (y_max - y_min if y_max > y_min else 1.0)

    p = figure(
        x_range=FactorRange(*factors),
        y_range=(y_min - pad, y_max + pad),
        height=500,
        width=1100,
        toolbar_location="above",
        tools="pan,wheel_zoom,box_zoom,reset,save",
        y_axis_label=ylabel,
    )

    bars = p.vbar(
        x="factor",
        top="mean",
        width=0.8,
        source=source,
    )

    whisker = Whisker(
        source=source,
        base="factor",
        upper="upper",
        lower="lower",
    )
    p.add_layout(whisker)

    p.xaxis.major_label_orientation = 0.9

    p.add_tools(
        HoverTool(
            renderers=[bars],
            tooltips=[
                ("Entity", "@entity_name"),
                ("Model", "@model"),
                ("Mean", "@mean"),
                ("SE", "@se"),
            ],
        )
    )

    return p


class AssayView:
    def __init__(self, assay: str):
        self.assay = assay

    def view(self):
        comparison_set_options = get_comparison_sets(self.assay)
        comparison_set = pn.widgets.Select(
            name="Comparison set",
            options=comparison_set_options,
            value=comparison_set_options[0],
        )

        estimand_options = get_estimands(self.assay)
        estimand = pn.widgets.Select(
            name="Estimand",
            options=estimand_options,
            value=estimand_options[0],
        )

        @pn.depends(comparison_set, estimand)
        def plot(comparison_set, estimand):
            pdf = get_plot_df(self.assay, comparison_set, estimand)
            return make_bar_plot(pdf, estimand)

        return pn.Column(
            f"## {self.assay}",
            pn.Row(comparison_set, estimand),
            plot,
        )


def app():
    assays = [
        "consideration-set",
        "describe-sentiment",
        "forced-selection",
        "head-to-head",
        "rank"
    ]

    return pn.Tabs(
        *((assay, AssayView(assay).view()) for assay in assays),
        dynamic=True,
    )


app().servable()