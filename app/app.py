from pathlib import Path

from dash import Dash, Input, Output, dcc, html
import plotly.graph_objects as go
import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PARQUET_PATH = PROJECT_ROOT / "data" / "processed" / "detroit_maternal_clean.parquet"

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

CARD_STYLE = {
    "backgroundColor": "#ffffff",
    "border": "1px solid #e5e7eb",
    "borderRadius": "12px",
    "padding": "16px 18px",
    "boxShadow": "0 1px 2px rgba(15, 23, 42, 0.04)",
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
    .with_columns(
        pl.col("geo_col").replace(GEO_MAP).alias("geography"),
    )
    .drop("geo_col")
    .with_columns(
        pl.col("indicator").replace_strict(indicator_rank, default=999).alias("_ind_rank"),
        pl.col("geography").replace_strict(GEO_RANK, default=999).alias("_geo_rank"),
    )
    .sort(["_ind_rank", "_geo_rank"])
    .drop(["_ind_rank", "_geo_rank"])
)

df_detroit_only = df_with_metrics.filter(
    pl.col("detroit_pct").is_not_null()
    & (pl.col("wayne_pct").is_null() | pl.col("michigan_pct").is_null())
)


def build_overview_figure():
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
      legend={
          "orientation": "h",
          "y": 1.08,
          "x": 1.0,
          "xanchor": "right",
          "title_text": "",
      },
      margin={"t": 72, "b": 32, "l": 18, "r": 48},
      height=max(320, 95 * len(indicator_order)),
      paper_bgcolor="#ffffff",
      plot_bgcolor="#ffffff",
      font={"family": "Inter, system-ui, -apple-system, sans-serif"},
  )
  return fig


def build_disparity_table():
  rows_html = []
  for r in df_disparity.iter_rows(named=True):
    gap = r["detroit_vs_michigan_pp"]
    ratio = r["detroit_to_michigan_ratio"]
    rows_html.append(
        html.Tr(
            style={"borderBottom": "1px solid #f1f5f9"},
            children=[
                html.Td(r["indicator"], style={"padding": "10px 12px", "fontWeight": "600"}),
                html.Td(r["year"], style={"padding": "10px 12px", "color": "#6b7280"}),
                html.Td(
                    f"{r['detroit_pct']:.1f}%",
                    style={"padding": "10px 12px", "color": "#6b2d5c", "fontWeight": "700"},
                ),
                html.Td(f"{r['wayne_pct']:.1f}%", style={"padding": "10px 12px"}),
                html.Td(f"{r['michigan_pct']:.1f}%", style={"padding": "10px 12px"}),
                html.Td(
                    f"{gap:+.1f} pp" if gap is not None else "—",
                    style={"padding": "10px 12px", "fontWeight": "600"},
                ),
                html.Td(
                    f"{ratio:.2f}x" if ratio is not None else "—",
                    style={"padding": "10px 12px", "fontWeight": "600"},
                ),
            ],
        )
    )

  headers = [
      "Indicator",
      "Year",
      "Detroit",
      "Wayne Co.",
      "Michigan",
      "Gap (DET − MI)",
      "Ratio (DET : MI)",
  ]

  return html.Table(
      style={"width": "100%", "borderCollapse": "collapse", "fontSize": "13px"},
      children=[
          html.Thead(
              html.Tr(
                  style={
                      "borderBottom": "2px solid #e5e7eb",
                      "textAlign": "left",
                      "color": "#4b5563",
                  },
                  children=[html.Th(h, style={"padding": "8px 12px"}) for h in headers],
              )
          ),
          html.Tbody(rows_html),
      ],
  )


def build_detroit_only_banner():
  if df_detroit_only.height == 0:
    return None
  items = [
      html.Li(
          (
              f"{r['indicator']} ({r['year']}): {r['detroit_pct']:.1f}% in Detroit — "
              f"{r['notes'] or 'Detroit-only metric'}"
          ),
          style={"marginBottom": "4px"},
      )
      for r in df_detroit_only.iter_rows(named=True)
  ]
  return html.Div(
      style={
          **CARD_STYLE,
          "marginTop": "18px",
          "backgroundColor": "#faf9fb",
          "borderColor": "#e9d8e6",
      },
      children=[
          html.Div(
              "DETROIT-ONLY METRICS IN CLEANED DATASET (EXCLUDED FROM 3-AREA RANKING)",
              style={
                  "fontSize": "11px",
                  "fontWeight": "700",
                  "letterSpacing": "0.06em",
                  "color": "#6b2d5c",
                  "marginBottom": "6px",
              },
          ),
          html.Ul(
              items,
              style={"margin": "0", "paddingLeft": "18px", "fontSize": "13px", "color": "#374151"},
          ),
      ],
  )


app = Dash(__name__)
server = app.server
app.title = "Mothering Justice — Detroit Maternal Health"

default_indicator = indicator_order[0] if indicator_order else None

app.layout = html.Div(
    style={
        "minHeight": "100vh",
        "backgroundColor": "#f8f9fb",
        "color": "#111827",
        "fontFamily": "Inter, system-ui, -apple-system, BlinkMacSystemFont, sans-serif",
        "padding": "36px 20px",
    },
    children=[
        html.Div(
            style={"maxWidth": "980px", "margin": "0 auto"},
            children=[
                html.Div(
                    style={"marginBottom": "22px"},
                    children=[
                        html.Div(
                            "MOTHERING JUSTICE · MATERNAL HEALTH EQUITY",
                            style={
                                "fontSize": "12px",
                                "fontWeight": "700",
                                "letterSpacing": "0.08em",
                                "color": "#6b2d5c",
                                "marginBottom": "6px",
                            },
                        ),
                        html.H1(
                            "Detroit Maternal & Structural Health Indicators",
                            style={
                                "fontSize": "28px",
                                "lineHeight": "1.2",
                                "fontWeight": "700",
                                "margin": "0 0 8px 0",
                            },
                        ),
                        html.P(
                            "Comparing Detroit, Wayne County, and Michigan across reported maternal and structural context indicators.",
                            style={"fontSize": "15px", "color": "#4b5563", "margin": "0"},
                        ),
                    ],
                ),
                dcc.Tabs(
                    id="view-tabs",
                    value="overview",
                    style={"marginBottom": "18px"},
                    children=[
                        dcc.Tab(
                            label="All Indicators Overview",
                            value="overview",
                            style={"padding": "10px 16px", "fontWeight": "600", "fontSize": "14px"},
                            selected_style={
                                "padding": "10px 16px",
                                "fontWeight": "700",
                                "fontSize": "14px",
                                "color": "#6b2d5c",
                                "borderTop": "3px solid #6b2d5c",
                            },
                        ),
                        dcc.Tab(
                            label="Single Indicator Deep Dive",
                            value="detail",
                            style={"padding": "10px 16px", "fontWeight": "600", "fontSize": "14px"},
                            selected_style={
                                "padding": "10px 16px",
                                "fontWeight": "700",
                                "fontSize": "14px",
                                "color": "#6b2d5c",
                                "borderTop": "3px solid #6b2d5c",
                            },
                        ),
                    ],
                ),
                html.Div(
                    id="overview-tab-content",
                    children=[
                        html.Div(
                            style={**CARD_STYLE, "marginBottom": "18px"},
                            children=[
                                dcc.Graph(
                                    id="overview-chart",
                                    figure=build_overview_figure(),
                                    config={"displayModeBar": False},
                                )
                            ],
                        ),
                        html.Div(
                            style={**CARD_STYLE, "overflowX": "auto"},
                            children=[
                                html.Div(
                                    "DISPARITY SUMMARY TABLE",
                                    style={
                                        "fontSize": "11px",
                                        "fontWeight": "700",
                                        "letterSpacing": "0.06em",
                                        "color": "#6b7280",
                                        "marginBottom": "10px",
                                    },
                                ),
                                build_disparity_table(),
                            ],
                        ),
                    ],
                ),
                html.Div(
                    id="detail-tab-content",
                    style={"display": "none"},
                    children=[
                        html.Div(
                            style={**CARD_STYLE, "marginBottom": "18px"},
                            children=[
                                html.Label(
                                    "Choose an indicator",
                                    htmlFor="indicator-dropdown",
                                    style={
                                        "display": "block",
                                        "fontSize": "13px",
                                        "fontWeight": "600",
                                        "color": "#374151",
                                        "marginBottom": "8px",
                                    },
                                ),
                                dcc.Dropdown(
                                    id="indicator-dropdown",
                                    options=[{"label": ind, "value": ind} for ind in indicator_order],
                                    value=default_indicator,
                                    clearable=False,
                                    style={"maxWidth": "520px", "fontSize": "14px"},
                                ),
                            ],
                        ),
                        html.Div(
                            id="kpi-row",
                            style={
                                "display": "grid",
                                "gridTemplateColumns": "repeat(auto-fit, minmax(220px, 1fr))",
                                "gap": "14px",
                                "marginBottom": "18px",
                            },
                        ),
                        html.Div(
                            style=CARD_STYLE,
                            children=[
                                dcc.Graph(
                                    id="comparison-chart",
                                    config={"displayModeBar": False},
                                )
                            ],
                        ),
                    ],
                ),
                build_detroit_only_banner(),
            ],
        )
    ],
)


@app.callback(
    Output("overview-tab-content", "style"),
    Output("detail-tab-content", "style"),
    Input("view-tabs", "value"),
)
def toggle_tabs(active_tab: str):
  if active_tab == "detail":
    return {"display": "none"}, {"display": "block"}
  return {"display": "block"}, {"display": "none"}


@app.callback(
    Output("comparison-chart", "figure"),
    Output("kpi-row", "children"),
    Input("indicator-dropdown", "value"),
)
def update_detail_view(selected_indicator: str | None):
  if not selected_indicator or df_long.height == 0:
    return {}, []

  sub = df_long.filter(pl.col("indicator") == selected_indicator)
  if sub.height == 0:
    return {}, []

  year_label = sub["year"][0]
  geos = []
  pcts = []
  colors = []
  for geo in GEO_ORDER:
    row = sub.filter(pl.col("geography") == geo)
    if row.height > 0:
      geos.append(geo)
      pcts.append(row["pct"][0])
      colors.append(GEO_COLORS[geo])

  pct_vals = [v for v in pcts if v is not None]
  max_val = max(pct_vals) if pct_vals else 10.0

  fig = go.Figure(
      data=[
          go.Bar(
              x=geos,
              y=pcts,
              marker_color=colors,
              text=[f"{v:.1f}%" if v is not None else "" for v in pcts],
              textposition="outside",
              cliponaxis=False,
              marker_line_width=0,
              width=0.52,
          )
      ]
  )
  fig.update_layout(
      template="plotly_white",
      title=f"{selected_indicator} ({year_label})",
      showlegend=False,
      xaxis={"showgrid": False, "categoryorder": "array", "categoryarray": GEO_ORDER},
      yaxis={
          "range": [0, max(max_val * 1.22, 10)],
          "ticksuffix": "%",
          "gridcolor": "#f1f5f9",
          "zerolinecolor": "#e5e7eb",
      },
      margin={"t": 56, "b": 28, "l": 48, "r": 24},
      height=360,
      paper_bgcolor="#ffffff",
      plot_bgcolor="#ffffff",
      font={"family": "Inter, system-ui, -apple-system, sans-serif"},
  )

  detroit_rows = sub.filter(pl.col("geography") == "Detroit")["pct"]
  detroit_val = (
      float(detroit_rows[0])
      if len(detroit_rows) and detroit_rows[0] is not None
      else None
  )
  gap = sub["detroit_vs_michigan_pp"][0]
  ratio = sub["detroit_to_michigan_ratio"][0]
  source_doc = sub["source_doc"][0]
  notes = sub["notes"][0]

  detroit_display = f"{detroit_val:.1f}%" if detroit_val is not None else "—"
  gap_display = f"{gap:+.1f} pp" if gap is not None else "—"
  ratio_sub = (
      f"{ratio:.2f}x Detroit : Michigan ratio"
      if ratio is not None
      else "Percentage-point difference (Detroit − MI)"
  )

  cards = [
      html.Div(
          style=CARD_STYLE,
          children=[
              html.Div(
                  "DETROIT SHARE",
                  style={
                      "fontSize": "11px",
                      "fontWeight": "700",
                      "color": "#6b7280",
                      "letterSpacing": "0.06em",
                  },
              ),
              html.Div(
                  detroit_display,
                  style={
                      "fontSize": "28px",
                      "fontWeight": "700",
                      "color": "#6b2d5c",
                      "marginTop": "6px",
                  },
              ),
              html.Div(
                  f"Reporting period: {year_label}",
                  style={"fontSize": "12px", "color": "#6b7280", "marginTop": "4px"},
              ),
          ],
      ),
      html.Div(
          style=CARD_STYLE,
          children=[
              html.Div(
                  "VS. MICHIGAN BASELINE",
                  style={
                      "fontSize": "11px",
                      "fontWeight": "700",
                      "color": "#6b7280",
                      "letterSpacing": "0.06em",
                  },
              ),
              html.Div(
                  gap_display,
                  style={
                      "fontSize": "28px",
                      "fontWeight": "700",
                      "color": "#111827",
                      "marginTop": "6px",
                  },
              ),
              html.Div(
                  ratio_sub,
                  style={"fontSize": "12px", "color": "#6b7280", "marginTop": "4px"},
              ),
          ],
      ),
      html.Div(
          style=CARD_STYLE,
          children=[
              html.Div(
                  "CONTEXT & SOURCE",
                  style={
                      "fontSize": "11px",
                      "fontWeight": "700",
                      "color": "#6b7280",
                      "letterSpacing": "0.06em",
                  },
              ),
              html.Div(
                  notes or "—",
                  style={
                      "fontSize": "13px",
                      "color": "#1f2937",
                      "marginTop": "6px",
                      "lineHeight": "1.45",
                  },
              ),
              html.Div(
                  source_doc or "—",
                  style={"fontSize": "12px", "color": "#6b7280", "marginTop": "6px"},
              ),
          ],
      ),
  ]

  return fig, cards


if __name__ == "__main__":
  app.run(debug=True)