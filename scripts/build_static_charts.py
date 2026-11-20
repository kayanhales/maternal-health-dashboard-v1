from pathlib import Path
import plotly.graph_objects as go
import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PARQUET_PATH = PROJECT_ROOT / "data" / "processed" / "detroit_maternal_clean.parquet"
OUT_DIR = PROJECT_ROOT / "dist"
OUT_DIR.mkdir(parents=True, exist_ok=True)

GEO_MAP = {
    "detroit_pct": "Detroit",
    "wayne_pct": "Wayne County",
    "michigan_pct": "Michigan",
}
GEO_ORDER = ["Detroit", "Wayne County", "Michigan"]
GEO_RANK = {name: i for i, name in enumerate(GEO_ORDER)}
GEO_COLORS = {
    "Detroit": "#6b2d5c",
    "Wayne County": "#9aa5b1",
    "Michigan": "#cbd2d9",
}

df_clean = pl.read_parquet(PARQUET_PATH)

df_with_metrics = df_clean.with_columns(
    (pl.col("detroit_pct") - pl.col("michigan_pct"))
    .round(1)
    .alias("detroit_vs_michigan_pp"),
    (pl.col("detroit_pct") - pl.col("wayne_pct"))
    .round(1)
    .alias("detroit_vs_wayne_pp"),
    (pl.col("detroit_pct") / pl.col("michigan_pct"))
    .round(2)
    .alias("detroit_to_michigan_ratio"),
)

df_disparity = df_with_metrics.filter(
    pl.all_horizontal(pl.col(col).is_not_null() for col in GEO_MAP.keys())
).sort("detroit_to_michigan_ratio", descending=True)

indicator_order = df_disparity["indicator"].to_list()
indicator_rank = {name: i for i, name in enumerate(indicator_order)}

df_long = (
    df_disparity.unpivot(
        index=[
            "indicator",
            "year",
            "source_doc",
            "notes",
            "detroit_vs_michigan_pp",
            "detroit_vs_wayne_pp",
            "detroit_to_michigan_ratio",
        ],
        on=list(GEO_MAP.keys()),
        variable_name="geo_col",
        value_name="pct",
    )
    .with_columns(pl.col("geo_col").replace(GEO_MAP).alias("geography"))
    .drop("geo_col")
    .with_columns(
        pl.col("indicator").replace_strict(indicator_rank, default=999).alias("_ind_rank"),
        pl.col("geography").replace_strict(GEO_RANK, default=999).alias("_geo_rank"),
    )
    .sort(["_ind_rank", "_geo_rank"])
    .drop(["_ind_rank", "_geo_rank"])
)


def build_overview_html() -> None:
  fig = go.Figure()

  for geo in GEO_ORDER:
    sub = df_long.filter(pl.col("geography") == geo)
    pct_list = sub["pct"].to_list()
    ind_list = sub["indicator"].to_list()
    custom = list(
        zip(
            sub["year"].to_list(),
            sub["detroit_vs_michigan_pp"].to_list(),
            sub["detroit_to_michigan_ratio"].to_list(),
        )
    )

    fig.add_trace(
        go.Bar(
            name=geo,
            x=pct_list,
            y=ind_list,
            orientation="h",
            marker_color=GEO_COLORS[geo],
            text=[f"{v:.1f}%" if v is not None else "" for v in pct_list],
            textposition="outside",
            cliponaxis=False,
            marker_line_width=0,
            customdata=custom,
            hovertemplate=(
                "<b>%{y}</b> (%{customdata[0]})<br>"
                "%{fullData.name}: <b>%{x:.1f}%</b><br>"
                "Detroit vs. MI gap: %{customdata[1]:+.1f} pp<br>"
                "Detroit : MI ratio: %{customdata[2]:.2f}x<extra></extra>"
            ),
        )
    )

  all_pcts = [v for v in df_long["pct"].to_list() if v is not None]
  max_pct = max(all_pcts) if all_pcts else 10.0

  fig.update_layout(
      barmode="group",
      template="plotly_white",
      title="Multi-Area Indicators Ranked by Detroit-to-Michigan Ratio",
      xaxis={
          "range": [0, max(max_pct * 1.18, 15)],
          "ticksuffix": "%",
          "gridcolor": "#f1f5f9",
          "zerolinecolor": "#e5e7eb",
      },
      yaxis={
          "showgrid": False,
          "categoryorder": "array",
          "categoryarray": list(reversed(indicator_order)),
      },
      legend={"orientation": "h", "y": 1.08, "x": 1.0, "xanchor": "right", "title_text": ""},
      margin={"t": 72, "b": 32, "l": 18, "r": 48},
      height=max(340, 95 * len(indicator_order)),
      paper_bgcolor="#ffffff",
      plot_bgcolor="#ffffff",
      font={"family": "Inter, system-ui, -apple-system, sans-serif"},
  )

  fig.write_html(
      OUT_DIR / "overview.html",
      include_plotlyjs="cdn",
      full_html=True,
  )


def build_deep_dive_html() -> None:
  if not indicator_order:
    return

  # Build one trace per indicator so a native Plotly dropdown switches indicators client-side
  fig = go.Figure()
  buttons = []

  for idx, ind in enumerate(indicator_order):
    sub = df_long.filter(pl.col("indicator") == ind)
    year_label = sub["year"][0]
    gap = sub["detroit_vs_michigan_pp"][0]
    ratio = sub["detroit_to_michigan_ratio"][0]

    geos, pcts, colors = [], [], []
    for geo in GEO_ORDER:
      row = sub.filter(pl.col("geography") == geo)
      if row.height > 0:
        geos.append(geo)
        pcts.append(row["pct"][0])
        colors.append(GEO_COLORS[geo])

    pct_vals = [v for v in pcts if v is not None]
    max_val = max(pct_vals) if pct_vals else 10.0

    fig.add_trace(
        go.Bar(
            name=ind,
            x=geos,
            y=pcts,
            visible=(idx == 0),
            marker_color=colors,
            text=[f"{v:.1f}%" if v is not None else "" for v in pcts],
            textposition="outside",
            cliponaxis=False,
            marker_line_width=0,
            width=0.52,
            hovertemplate="%{x}: <b>%{y:.1f}%</b><extra></extra>",
        )
    )

    visibility = [i == idx for i in range(len(indicator_order))]
    gap_str = f"{gap:+.1f} pp" if gap is not None else "—"
    ratio_str = f"{ratio:.2f}x" if ratio is not None else "—"
    subtitle = f" ({year_label}) — Gap vs. MI: {gap_str} | Ratio: {ratio_str}"

    buttons.append(
        {
            "label": ind,
            "method": "update",
            "args": [
                {"visible": visibility},
                {
                    "title": f"{ind}{subtitle}",
                    "yaxis.range": [0, max(max_val * 1.22, 10)],
                },
            ],
        }
    )

  first_ind = indicator_order[0]
  first_sub = df_long.filter(pl.col("indicator") == first_ind)
  first_max = max(
      [v for v in first_sub["pct"].to_list() if v is not None] or [10.0]
  )

  fig.update_layout(
      template="plotly_white",
      title=f"{first_ind} ({first_sub['year'][0]})",
      showlegend=False,
      updatemenus=[
          {
              "buttons": buttons,
              "direction": "down",
              "showactive": True,
              "x": 0.0,
              "xanchor": "left",
              "y": 1.22,
              "yanchor": "top",
          }
      ],
      xaxis={"showgrid": False, "categoryorder": "array", "categoryarray": GEO_ORDER},
      yaxis={
          "range": [0, max(first_max * 1.22, 10)],
          "ticksuffix": "%",
          "gridcolor": "#f1f5f9",
          "zerolinecolor": "#e5e7eb",
      },
      margin={"t": 95, "b": 32, "l": 48, "r": 24},
      height=400,
      paper_bgcolor="#ffffff",
      plot_bgcolor="#ffffff",
      font={"family": "Inter, system-ui, -apple-system, sans-serif"},
  )

  fig.write_html(
      OUT_DIR / "deep_dive.html",
      include_plotlyjs="cdn",
      full_html=True,
  )


if __name__ == "__main__":
  build_overview_html()
  build_deep_dive_html()
  print(f"Generated charts in {OUT_DIR}")