import json
from pathlib import Path

import panel as pn
import polars as pl
import plotly.graph_objects as go
import yaml

pn.extension("plotly")

APP_WIDTH = "90vw"
BASE_PLOT_WIDTH = 0
WIDTH_PER_BAR = 15
MIN_PLOT_WIDTH = 800
PLOT_CONTAINER_HEIGHT = "50vh"
INSTANCE_CONTAINER_HEIGHT = "35vh"

ASSAY_PARQUET_PATH = "data/summarised_assays.parquet"
INSTANCE_PARQUET_PATH = "data/db/comparison_set_assay_instance.parquet"
TOOLTIPS_YAML_PATH = Path("app/tooltips.yaml")

ASSAY_DF = pl.read_parquet(ASSAY_PARQUET_PATH)
INSTANCE_DF = pl.read_parquet(INSTANCE_PARQUET_PATH)


def load_tooltips() -> dict[str, dict[str, str]]:
    if not TOOLTIPS_YAML_PATH.exists():
        return {}

    with TOOLTIPS_YAML_PATH.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    return data if isinstance(data, dict) else {}


TOOLTIPS = load_tooltips()


def get_estimand_tooltip(assay: str, estimand: str | None) -> str:
    if estimand is None:
        return ""

    assay_tooltips = TOOLTIPS.get(assay, {})
    if not isinstance(assay_tooltips, dict):
        return ""

    tooltip = assay_tooltips.get(estimand, "")
    return str(tooltip) if tooltip is not None else ""


def get_num_samples_per_instance(
    assay: str,
    comparison_set_name: str,
    estimand: str,
) -> int | None:
    df = (
        ASSAY_DF
        .filter(
            (pl.col("assay") == assay)
            & (pl.col("comparison_set_name") == comparison_set_name)
            & (pl.col("estimand") == estimand)
        )
        .select("num_samples_per_instance")
        .unique()
    )

    if df.is_empty():
        return None

    return int(df["num_samples_per_instance"][0])


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


def get_estimands(
    assay: str,
    comparison_set_name: str | None = None,
) -> list[str]:
    df = ASSAY_DF.filter(pl.col("assay") == assay)

    if comparison_set_name is not None:
        df = df.filter(pl.col("comparison_set_name") == comparison_set_name)

    return (
        df
        .select("estimand")
        .unique()
        .sort("estimand")
        .get_column("estimand")
        .to_list()
    )


def get_plot_df(
    assay: str,
    comparison_set_name: str,
    estimand: str,
):
    df = (
        ASSAY_DF
        .filter(
            (pl.col("assay") == assay)
            & (pl.col("comparison_set_name") == comparison_set_name)
            & (pl.col("estimand") == estimand)
        )
        .select(
            "entity_name",
            "model",
            pl.col("estimate_mean").alias("mean"),
            pl.col("estimate_se").alias("se"),
        )
        .sort("entity_name", "model")
    )

    if df.is_empty():
        return pl.DataFrame(
            schema={
                "entity_name": pl.String,
                "model": pl.String,
                "mean": pl.Float64,
                "se": pl.Float64,
            }
        ).to_pandas()

    return df.to_pandas()


def get_instance_jsons(
    assay: str,
    comparison_set_name: str,
) -> list[str]:
    df = (
        INSTANCE_DF
        .filter(
            (pl.col("assay") == assay)
            & (pl.col("comparison_set_name") == comparison_set_name)
        )
        .select("instance_json")
    )

    values = df.get_column("instance_json").to_list()

    rendered = []
    for value in values:
        if value is None:
            continue

        if isinstance(value, str):
            try:
                parsed = json.loads(value)
                rendered.append(json.dumps(parsed, indent=2, ensure_ascii=False))
            except json.JSONDecodeError:
                rendered.append(value)
        else:
            rendered.append(json.dumps(value, indent=2, ensure_ascii=False))

    return rendered


def make_instance_list_pane(
    assay: str,
    comparison_set_name: str,
    estimand: str,
) -> pn.Column:
    instance_jsons = get_instance_jsons(assay, comparison_set_name)
    num_samples_per_instance = get_num_samples_per_instance(
        assay=assay,
        comparison_set_name=comparison_set_name,
        estimand=estimand,
    )

    if not instance_jsons:
        return pn.Column(
            pn.pane.Markdown("### Instances"),
            pn.pane.Markdown("No instance JSON found for this assay/comparison set."),
            sizing_mode="stretch_width",
            margin=(10, 0, 0, 0),
        )

    blocks = [pn.pane.Markdown("### Instances", margin=(0, 0, 10, 0))]

    if num_samples_per_instance is not None:
        blocks.append(
            pn.pane.Markdown(
                f"{len(instance_jsons)} instance(s) · {num_samples_per_instance} sample(s) per instance",
                margin=(0, 0, 10, 0),
            )
        )
    else:
        blocks.append(
            pn.pane.Markdown(
                f"{len(instance_jsons)} instance(s)",
                margin=(0, 0, 10, 0),
            )
        )

    for i, payload in enumerate(instance_jsons, start=1):
        blocks.append(
            pn.pane.Markdown(
                f"**Instance {i}**",
                margin=(8, 0, 4, 0),
            )
        )
        blocks.append(
            pn.pane.JSON(
                json.loads(payload) if payload.strip().startswith(("{", "[")) else payload,
                depth=-1,
                sizing_mode="stretch_width",
            )
        )

    return pn.Column(
        *blocks,
        styles={
            "width": "100%",
            "height": INSTANCE_CONTAINER_HEIGHT,
            "overflow-y": "auto",
            "overflow-x": "auto",
            "border": "1px solid #cbd5e1",
            "border-radius": "8px",
            "padding": "10px",
            "box-sizing": "border-box",
        },
        sizing_mode="stretch_width",
        margin=(10, 0, 0, 0),
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
    initial_comparison_set = comparison_sets[0] if comparison_sets else None
    estimands = (
        get_estimands(assay, initial_comparison_set)
        if initial_comparison_set is not None
        else []
    )

    comparison_set_select = pn.widgets.Select(
        name="Comparison set",
        options=comparison_sets,
        value=initial_comparison_set,
        width=250,
    )

    estimand_select = pn.widgets.Select(
        name="Estimand",
        options=estimands,
        value=estimands[0] if estimands else None,
        width=250,
        description=get_estimand_tooltip(assay, estimands[0] if estimands else None),
    )

    def _update_estimand_options(event) -> None:
        new_estimands = get_estimands(assay, event.new)

        estimand_select.options = new_estimands
        estimand_select.value = new_estimands[0] if new_estimands else None
        estimand_select.description = get_estimand_tooltip(
            assay,
            estimand_select.value,
        )

    comparison_set_select.param.watch(_update_estimand_options, "value")

    def _update_estimand_tooltip(event) -> None:
        estimand_select.description = get_estimand_tooltip(assay, event.new)

    estimand_select.param.watch(_update_estimand_tooltip, "value")

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
            instances_obj = pn.pane.Markdown("No instances available.")
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

            instances_obj = make_instance_list_pane(
                assay=assay,
                comparison_set_name=comparison_set_name,
                estimand=estimand,
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

        return pn.Column(
            plot_container,
            instances_obj,
            sizing_mode="stretch_width",
            margin=0,
        )

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