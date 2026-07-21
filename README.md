# AI Capex Efficiency

Dollarizes the value of a more efficient AI architecture — cutting model **memory ≈100×**
and **FLOPs ≈4× measured / ≈10× at the ceiling** — for the companies spending the most on
AI infrastructure.

The headline isn't 100× or 1000×. A GPU is **~60% memory / ~40% compute by cost**, so the
cost-weighted saving is Amdahl-floored by the least-reduced component (compute). The
default **measured** scenario uses a 2026-07-21 single-H100 measurement (parameter-matched,
bf16): serving memory **43–315× smaller at 1M tokens** (the ÷100 lever sits inside the
band) and long-context prefill **4× faster** → **~9.4× cost-weighted**. The **ceiling**
scenario (compute ÷10 as kernels mature) is **~22×**. Each is applied bottom-up to each
company's disclosed capex to size the **avoided spend** and its **capitalized value**.

Coverage: the 6 largest AI-capex spenders (Microsoft, Alphabet, Amazon, Meta, Oracle,
SpaceX) plus a grossed-up **global estimate**.

## Live app

Deploy free on [Streamlit Community Cloud](https://share.streamlit.io) — see **Deploy** below.
The app is a **tab-for-tab mirror of the workbook**: one tab per worksheet (Totals, each company,
Inputs, Sensitivity, Cost Ladder, Evidence, Methodology), each rendered as the same colored grid.
Edit the 🟡/🟢 cells (globals in the sidebar; per-company in the ✏️ panel on each company tab) and
every grid recomputes live.

## Contents

| File | What it is |
|---|---|
| `app.py` | Streamlit app — interactive tab-for-tab mirror of the workbook |
| `ai_capex_model.py` | **Single source of truth** — all defaults + math (app and Excel both import it, so they can't drift) |
| `ai_capex_efficiency.py` | Generates `AI_Capex_Efficiency.xlsx` (live-formula workbook) |
| `AI_Capex_Efficiency.xlsx` | The model as an auditable spreadsheet — every output is a live formula |
| `requirements.txt` | App dependencies (Streamlit + pandas) |

## Run locally

```bash
# with uv (no venv needed)
uv run --with streamlit --with pandas streamlit run app.py
# or with pip
pip install -r requirements.txt && streamlit run app.py
```

## Deploy (free)

1. This repo is already on GitHub.
2. Go to **share.streamlit.io** → *New app* → pick this repo, branch `main`, main file `app.py`.
3. Deploy. You get a public URL to share.

## How it works

- **Engine:** cost-weighted reduction `= 1 / (mem_share/mem_factor + (1−mem_share)/flop_factor)` — ≈9.4× measured (flop ×4), ≈22× ceiling (flop ×10).
- **Per company:** `total capex (disclosed) × infra share × server share × accelerator share`
  → accelerator capex → fleet → energy/opex → avoided spend → capitalized value. FY2025 actual + FY2026 estimate.
- **Net AI economics (cash basis):** `AI revenue − AI capex − AI opex`; with the architecture, add the spend cut.
- **Global estimate:** the named firms are grossed up by their assumed share of worldwide AI capex.

## Color / assumption convention (mirrored in the spreadsheet)

- 🟡 **assumption** — a lever we chose; editable in the app (sidebar / per-company ✏️ panel)
- 🟢 **disclosed data** — from filings or markets (FY25 capex, market caps)
- 🔵 **derived** — a formula

## Regenerate the spreadsheet

The Excel is generated from the same model module:

```bash
uv run --with openpyxl python ai_capex_efficiency.py
```

## Caveats

Cash basis (capex not depreciated). Totals are disclosed; server/accelerator splits are estimated
from BOM teardowns and CFO commentary (±15–20%). AI revenue is the softest input — Microsoft
($37B) and Amazon ($15B) run-rates are disclosed; the rest are estimates. The capitalization is a
simple perpetuity (benefit ÷ discount rate). This is an analytical estimate, not investment advice.
