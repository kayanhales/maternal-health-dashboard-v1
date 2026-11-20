import json
from pathlib import Path
import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PARQUET_PATH = PROJECT_ROOT / "data" / "processed" / "detroit_maternal_clean.parquet"
OUT_DIR = PROJECT_ROOT / "dist"
OUT_DIR.mkdir(parents=True, exist_ok=True)

GEO_COLS = {
    "detroit_pct": "Detroit",
    "wayne_pct": "Wayne County",
    "michigan_pct": "Michigan",
}

df = (
    pl.read_parquet(PARQUET_PATH)
    .with_columns(
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
    .filter(pl.all_horizontal(pl.col(col).is_not_null() for col in GEO_COLS))
    .sort("detroit_to_michigan_ratio", descending=True)
)

records = []
for r in df.iter_rows(named=True):
  records.append({
      "indicator": r["indicator"],
      "year": r["year"],
      "source_doc": r["source_doc"],
      "notes": r["notes"],
      "detroit_vs_michigan_pp": r["detroit_vs_michigan_pp"],
      "detroit_vs_wayne_pp": r["detroit_vs_wayne_pp"],
      "detroit_to_michigan_ratio": r["detroit_to_michigan_ratio"],
      "values": [
          {"geography": "Detroit", "pct": r["detroit_pct"]},
          {"geography": "Wayne County", "pct": r["wayne_pct"]},
          {"geography": "Michigan", "pct": r["michigan_pct"]},
      ],
  })

out_file = OUT_DIR / "indicators.json"
out_file.write_text(json.dumps(records, indent=2), encoding="utf-8")
print(f"Wrote {len(records)} indicators to {out_file}")