import math

import panel as pn
import polars as pl
import plotly.graph_objects as go

pn.extension("plotly")

APP_WIDTH = "90vh"
BASE_PLOT_WIDTH = 0
WIDTH_PER_BAR = 15
MIN_PLOT_WIDTH = 800
PLOT_CONTAINER_HEIGHT = "50vh"

ASSAY_PARQUET_PATH = "data/combined_assays.parquet"
ASSAY_DF = pl.read_parquet(ASSAY_PARQUET_PATH)


def get_assays() -> list[str]:
    return (
        ASSAY_DF
        .select("assay")
        .unique()
        .sort("assay")
        .get_column("assay")
        .to_list()
    )


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


def get_exploded_df(
    assay: str,
    comparison_set_name: str,
    estimand: str,
) -> pl.DataFrame:
    return (
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


def get_plot_df(
    assay: str,
    comparison_set_name: str,
    estimand: str,
):
    exploded = get_exploded_df(assay, comparison_set_name, estimand)

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

    return (
        pl.DataFrame(rows)
        .sort("entity_name", "model")
        .to_pandas()
    )


def get_plot_width(pdf) -> int:
    if pdf.empty:
        return MIN_PLOT_WIDTH

    num_entities = pdf["entity_name"].nunique()
    num_models = pdf["model"].nunique()
    num_bars = num_entities * num_models

    return max(MIN_PLOT_WIDTH, BASE_PLOT_WIDTH + WIDTH_PER_BAR * num_bars)


def make_bar_plot(pdf, ylabel: str, plot_width: int) -> go.Figure:
    if pdf.empty:
        fig = go.Figure()
        fig.update_layout(
            template="plotly_white",
            width=plot_width,
            height=500,
            autosize=False,
            margin=dict(l=40, r=20, t=40, b=180),
            xaxis_title="Entity / Model",
            yaxis_title=ylabel,
            annotations=[
                dict(
                    text="No data for this selection",
                    x=0.5,
                    y=0.5,
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                )
            ],
        )
        fig.update_xaxes(
            tickangle=-60,
            tickfont=dict(size=10),
            automargin=True,
        )
        return fig

    y_upper = (pdf["mean"] + pdf["se"]).max()
    y_lower = min(0, (pdf["mean"] - pdf["se"]).min())
    pad = 0.05 * (y_upper - y_lower if y_upper > y_lower else 1.0)

    fig = go.Figure(
        data=[
            go.Bar(
                x=[pdf["entity_name"].tolist(), pdf["model"].tolist()],
                y=pdf["mean"],
                error_y=dict(
                    type="data",
                    array=pdf["se"],
                    visible=True,
                ),
                customdata=pdf[["entity_name", "model", "mean", "se"]].to_numpy(),
                hovertemplate=(
                    "Entity: %{customdata[0]}<br>"
                    "Model: %{customdata[1]}<br>"
                    "Mean: %{customdata[2]:.4f}<br>"
                    "SE: %{customdata[3]:.4f}<extra></extra>"
                ),
            )
        ]
    )

    fig.update_layout(
        template="plotly_white",
        width=plot_width,
        height=500,
        autosize=False,
        margin=dict(l=40, r=20, t=40, b=220),
        xaxis_title="Entity / Model",
        yaxis_title=ylabel,
        yaxis=dict(range=[y_lower - pad, y_upper + pad]),
    )

    fig.update_xaxes(
        tickangle=-60,
        tickfont=dict(size=10),
        automargin=True,
    )

    return fig


def make_assay_pane(assay: str) -> pn.Column:
    comparison_sets = get_comparison_sets(assay)
    estimands = get_estimands(assay)

    comparison_set_select = pn.widgets.Select(
        name="Comparison set",
        options=comparison_sets,
        value=comparison_sets[0] if comparison_sets else None,
        width=250,
    )

    estimand_select = pn.widgets.Select(
        name="Estimand",
        options=estimands,
        value=estimands[0] if estimands else None,
        width=250,
    )

    controls = pn.Row(
        comparison_set_select,
        estimand_select,
        sizing_mode="fixed",
        margin=(0, 0, 10, 0),
    )

    @pn.depends(
        comparison_set_select.param.value,
        estimand_select.param.value,
    )
    def make_plot(comparison_set_name: str, estimand: str):
        if comparison_set_name is None or estimand is None:
            plot_width = MIN_PLOT_WIDTH
            plot_obj = pn.pane.Markdown(
                "No data available.",
                width=plot_width,
                sizing_mode="fixed",
            )
        else:
            pdf = get_plot_df(
                assay=assay,
                comparison_set_name=comparison_set_name,
                estimand=estimand,
            )
            plot_width = get_plot_width(pdf)
            fig = make_bar_plot(pdf, estimand, plot_width)

            plot_obj = pn.pane.Plotly(
                fig,
                config={"responsive": False},
                width=plot_width,
                height=500,
                sizing_mode="fixed",
                width_policy="fixed",
            )

        plot_inner = pn.Row(
            plot_obj,
            width=plot_width,
            sizing_mode="fixed",
            width_policy="fixed",
            margin=0,
        )

        plot_container = pn.Column(
            plot_inner,
            styles={
                "width": "100%",
                "height": PLOT_CONTAINER_HEIGHT,
                "overflow-x": "auto",
                "overflow-y": "hidden",
                "border": "1px solid #cbd5e1",
                "border-radius": "8px",
                "box-sizing": "border-box",
            },
            margin=0,
        )

        return plot_container

    return pn.Column(
        controls,
        make_plot,
        sizing_mode="stretch_width",
        margin=0,
    )


def make_assay_tab(assay: str):
    return (assay, make_assay_pane(assay))


tabs = pn.Tabs(
    *[make_assay_tab(assay) for assay in get_assays()]
)

app = pn.Column(
    tabs,
    styles={
        "width": APP_WIDTH,
        "margin": "0 auto",
    },
)

app.servable()