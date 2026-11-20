# Maternal Health Disparity Explorer (Practice & Portfolio)

A simple hands-on data science practice project comparing maternal health and structural context indicators across **Detroit**, **Wayne County**, and **Michigan**. Built as exploratory portfolio work while testing visualization tools, static web deployment, and interactive notebooks.

> **Live Static View (GitHub Pages):** [index.html](https://kayanhales.github.io/maternal-health-dashboard-v1/index.html)  
> **Interactive D3 View (GitHub Pages):** [d3_view.html](https://kayanhales.github.io/maternal-health-dashboard-v1/d3_view.html)  
> **Repository:** [kayanhales/maternal-health-dashboard-v1](https://github.com/kayanhales/maternal-health-dashboard-v1)

---

## What This Project Does

- **Data prep (`polars`):** Loads `detroit_maternal_clean.parquet`, calculates disparity metrics (`detroit_vs_michigan_pp`, `detroit_vs_wayne_pp`, and `detroit_to_michigan_ratio`), and separates multi-area comparisons from Detroit-only indicators.
- **Static views (`dist/` + GitHub Pages):** Exports `indicators.json` and HTML views rendered with **D3.js** and **Plotly**, deployed automatically via **GitHub Actions**. 
- **Interactive notebook (Deepnote):** Uses `polars` and `plotly` with a dropdown selector and context blocks for exploring indicators interactively. Import the parquet file or the repo into Deepnote.

---

## Running Locally after cloning

```bash
pip install polars pyarrow plotly
python -m http.server 8000

Open http://localhost:8000/d3_view.html (or index.html) in your browser.