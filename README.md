# AI Capex Efficiency

Dollarizes the value of a more efficient AI architecture for the companies spending the most
on AI infrastructure.

The headline isn't 100× or 1000×. A GPU is **~60% memory / ~40% compute by cost**, so the
cost-weighted saving is Amdahl-floored by the least-reduced component (compute), and it
**saturates**: past a compute lever of ~10× the money stops moving.

The default scenario is the **2026-08-14 sealed refit**, which splits the FLOPs lever into two
factors that are labelled separately everywhere they appear:

- **FIT-DERIVED — equal-quality parameter matching.** Both families now have four measured
  ladder rungs (47M–663M params) fitted as `ln(bpb) = a + b·ln(params)`. The fits cross at
  **392M**; above it bAttention reaches the same fitted quality on fewer parameters —
  **84.2% at 1B, 55.2% at 10B, 36.2% at 100B, 23.7% at 1T, 15.5% at 10T**. Per-token cost is
  ~linear in parameters, so 23.7% of the parameters is a **×4.22 compute-per-token advantage
  at equal quality** at trillion-parameter scale. This is a *projection of two fits, not a
  measurement*, and every point past ~1B extrapolates beyond the measured rungs.
- **MEASURED — kernel cost ratio at a chosen operating point.** Default is the **×11.93**
  serving blend at 64k context / 10:1 input:output. Other selectable points: ×4.02
  long-context prefill at 1M tokens (2026-07-21), ×1 parity, and **×0.15 — the single-GPU
  training step at 2,048-token context, where we are ×6.46 *slower***.

At 1T scale with the serving blend the lever is ×50.33 → **~71.7× cost-weighted** →
**~$165B/yr** of FY2025 spend removed across the named firms. Memory stays at the deliberate
**÷100** cap even though the *measured* serving-state ratio is **×65,536 at 64k context**
(0.15234 MB of transformer KV per token of context per stream, d1536/26 layers fp16, against
a constant 0.15 MB decode carry) — past ~100× the memory term is 0.6% of residual cost and
stops moving the blend.

Every earlier scenario is kept in the sidebar: 2026-07-21 measured (compute ×4 → ~9.4×),
ceiling (×10 → ~22×), memory-only (×1 → ~2.5×). Each is applied bottom-up to each company's
disclosed capex to size the **avoided spend** and its **capitalized value**.

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

## Scenarios

| Scenario | FLOPs lever | Cost-weighted | FY25 spend cut | Basis |
|---|---|---|---|---|
| **Quality-matched (2026-08-14)** — default | ×50.33 at 1T scale (×4.22 FIT × ×11.93 MEASURED) | ~71.7× | ~$165B/yr | sealed quality refit + 2026-08-07/08 serving receipts |
| Measured (H100 2026-07-21) | ×4 | ~9.4× | ~$149B/yr | parameter-matched bf16 prefill at 1M tokens |
| Ceiling (kernels mature) | ×10 | ~21.7× | ~$159B/yr | architectural ceiling |
| Memory-only (compute parity) | ×1 | ~2.5× | ~$99B/yr | credits nothing to compute |

The FY25 cut spans only ~$99B–$165B across a 50× spread in the compute lever, because the cut is
`accelerator capex × (1 − 1/reduction)` and saturates. **Doubling the multiple is worth a few $B.**
That is the honest shape of the result and the reason the model refuses multiplicative headlines.

### Receipts for the 2026-08-14 scenario

Every constant in the `2026-08-14 SEALED SCALING RECEIPTS` block of `ai_capex_model.py` names its
source file, and `python ai_capex_model.py` re-derives the published parameter-matching curve from
the fit coefficients and asserts it matches to <1e-6 pp.

| What | Receipt |
|---|---|
| Quality fits, crossover, parameter-matching curve | `bdm/docs/deck/build/quality_fit_v4.json` (built by `docs/deck/build/build_slides_v4.py`) |
| Narrative receipt for the above | `bdm/docs/deck/README.md`, section `## v4` |
| Serving state, decode, memory walls, 8-GPU training scaling | `bdm/docs/deck/build/derived.json` |
| Single-GPU training-step gap (the measurement against us) | `bdm/docs/deck/build/dtype_trio_v4.json` |
| Serving blend (×11.93) over those receipts | the deck build's serving module (`epsilon-rnn ai_capex_model.workload_compute_advantage`) |

## How it works

- **Engine:** cost-weighted reduction `= 1 / (mem_share/mem_factor + (1−mem_share)/flop_factor)` — ≈71.7× quality-matched at 1T (flop ×50.33), ≈9.4× at the 2026-07-21 measured lever (flop ×4), ≈22× ceiling (flop ×10).
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
