# AI Capex Efficiency

Dollarizes the value of the Helarctos architecture for the six companies spending the most on AI
infrastructure (Microsoft, Alphabet, Amazon, Meta, Oracle, SpaceX), plus a grossed-up global estimate.

**Headline (Today scenario):** the six firms spend ~$779B on AI in FY2026 and earn ~$136B from it.
Helarctos makes **~$376–380B/yr** of that spend unnecessary for the same AI output (~48% of all AI
spend, ~99% of the AI-chip bill), shrinking the FY2026 cash burn from ~−$643B to ~−$265B. FY2025:
~$159–161B/yr. Capitalized at 6%: ~$6.3T (FY2026 run-rate).

The headline is dollars, not a multiple. A GPU is bought whole, so a fleet shrinks by whichever need
falls least; once that is ~100×, 99% of the chip bill is already gone and bigger multiples only move
the last 1%. The dollars are set by how much these firms spend on chips.

## Start here: Summary, Value Bridge, Levers

The workbook and the app both open on three plain-language tabs. The engineering ledger sits behind
them as a technical appendix (Totals, Inputs, Sensitivity, CostLadder, ServingTraining, Evidence,
Methodology).

- **Summary**: the FY2026 and FY2025 headline, where the saving comes from (training clusters,
  serving GPUs, power), and a per-company table.
- **Value Bridge**: a step-by-step walk from disclosed capex to the saving. The chip fleet is split
  into a **training fleet** and a **serving fleet** using per-company training shares (25–55%).
  - **Training:** a same-quality Helarctos model is ×4.2 smaller, trains on ×4.2 fewer tokens and is
    ×2.2 cheaper per token on a modern long-context mix — ×40 fewer GPU-hours, so the training cluster
    can be ×40 smaller. The GPUs it no longer needs are capex avoided (~$135B FY2026).
  - **Serving:** memory per conversation falls ÷100 (fixed-size state instead of a growing KV cache) and
    compute per token falls ×793. A GPU is bought whole, so the serving fleet shrinks by the binding
    limit — ×100 (~$230B FY2026).
  - A **one-lever-at-a-time ladder** shows where the needle moves (FY2026, Today):

    | Step | Spend cut |
    |---|---|
    | Smaller model for the same quality (training only — serving GPUs stay memory-bound) | ~$135B |
    | + Fixed-size memory per conversation | ~$318B |
    | + Many more conversations per GPU | ~$372B |
    | + Faster training on long documents | ~$376B |

- **Levers**: the five Helarctos levers in plain English, each with how sure we are of it
  (MEASURED / PROJECTED / ESTIMATE), what it means, and which part of the bill it shrinks; the
  Today / Optimized-kernels switch; the per-company training shares.

The bridge takes no memory credit on training and prices serving on whole GPUs. It lands within ~1% of
the technical Totals tab ($376B vs $380B FY2026), which prices memory (~60% of a GPU's cost) and
compute (~40%) separately. `value_bridge()` in `ai_capex_model.py` is the Python twin of these tabs;
an import-time check asserts it reproduces the Totals engine exactly when given the same levers.

## Scenarios

| Scenario | FLOPs lever | Cost-weighted (technical tabs) | FY2025 cut | FY2026 cut | Basis |
|---|---|---|---|---|---|
| **Today** (default in the workbook) | ×793 | ~154× | ~$161B | ~$380B | banked kernel results + aggregate-decode ESTIMATE |
| Ceiling: optimized kernels (default in the app) | ×816 | ~154× | ~$161B | ~$380B | same, with the funded kernel campaign landed (TARGET) |

Both scenarios use memory ÷100. The FLOPs lever is the full-workload serving lever at a 128k-token
average context (decode 2.5 ms/token × 64 resident streams per GPU — an ESTIMATE awaiting its
aggregate-decode measurement; prefill at the banked ×3.94 Today, ×7.03 at the Ceiling) times the
×4.22 equal-quality parameter ratio at trillion-parameter scale (PROJECTED). The two differ ~3% in
the lever and within rounding in dollars. The previous ×9.24 lever (~20× → ~$159B) was retired on
2026-09-01 because the kernels it was measured on no longer exist.

## Where the numbers come from

- **Capex:** FY2025 actuals and FY2026 guidance/actuals from filings and earnings calls (sources on
  each company tab).
- **Data-center share of capex** (2026-09-29): from the 10-K/10-Q property & equipment notes — gross
  additions by asset class, with offices, leasehold improvements, furniture and "equipment and other"
  counted as non-data-center — or segment notes:

  | Company | FY2025 | FY2026 | Filing |
  |---|---|---|---|
  | Microsoft | 97% | 97% | FY2025 and FY2026 10-K |
  | Alphabet | 93% | 95% | 2025 10-K; Q2 2026 10-Q |
  | Meta | 95% | 98% | 2025 10-K; Q2 2026 10-Q |
  | Amazon (AWS share) | 68% | 76% | 2025 10-K; Q2 2026 10-Q segment note |
  | Oracle | 85% | 90% | estimate: no office line in the P&E note |
  | SpaceX | 100% | 100% | AI-segment capex reported directly (S-1, 10-Q) |

- **Server and accelerator shares:** CFO commentary and BOM teardowns (±15–20%); the filings' server
  share of additions (Microsoft 45% → 62%, Alphabet ~60%) are consistent with the model.
- **Levers:** internal measurements and fits, each labelled MEASURED / PROJECTED / ESTIMATE / TARGET on
  the Levers and ServingTraining tabs.

## How it works

- **Per company:** `total capex × data-center share × server share × accelerator share` → accelerator
  (chip) capex → fleet → power/opex → avoided spend → capitalized value. FY2025 actual + FY2026.
- **Technical engine (Totals, company tabs):** cost-weighted reduction
  `= 1 / (mem_share/mem_factor + (1−mem_share)/flop_factor)` ≈ 154×.
- **Audience engine (Summary, Value Bridge):** training fleet ÷ GPU-hour lever (×40); serving fleet ÷
  min(memory lever, compute lever) (×100); power follows the fleet.
- **Net AI economics (cash basis):** `AI revenue − AI capex − AI power`; with the architecture, add the
  spend cut.
- **Global estimate:** the named firms are grossed up by their assumed share (80%) of worldwide AI capex.

## Live app

Deploy free on [Streamlit Community Cloud](https://share.streamlit.io) — see **Deploy** below.
The app is a **tab-for-tab mirror of the workbook** (Summary, Value Bridge, Levers, each company,
Totals, Inputs, Sensitivity, CostLadder, Serving·Training, Evidence, Methodology). Edit the 🟡/🟢
cells (globals in the sidebar; per-company in the ✏️ panel on each company tab; training shares on
the Levers tab) and every grid recomputes live.

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

## Regenerate the spreadsheet

```bash
uv run --with openpyxl python ai_capex_efficiency.py
```

`python ai_capex_model.py` runs the model's self-checks, including a guard that the quoted headline
figures still match what the model computes.

## Color / assumption convention (mirrored in the spreadsheet)

- 🟡 **assumption** — a lever we chose; editable in the app
- 🟢 **disclosed data** — from filings or markets
- 🔵 **derived** — a formula

## Background: the measured serving cells (2026-08-14)

A full-model bf16 decode measurement on one GH200 (d2048/24 layers, both families, same session)
established what the serving levers rest on:

| Context | Transformer | bAttention | Aggregate ratio |
|---|---|---|---|
| 32,768 | 8 streams, 511.8 tok/s/GPU (16 OOM; ceiling bracketed at 8–15) | 64 streams, 585.4 tok/s/GPU | ×0.97–1.14 bounded |
| 65,536 | 255.9 tok/s/GPU | 562.1 tok/s/GPU | ×2.20 derived |
| 262,144 | 1 stream, 64.0 tok/s/GPU (2 streams OOM) | 64 streams in 1.6 GB, 562.1 tok/s/GPU | **×8.79 measured** |

- **The serving lever is a memory-ceiling lever, not a latency lever.** Per generated token at a single
  stream the transformer is faster at 64k context (~4.9 vs ~5.7 ms GPU-busy). What it cannot do is hold
  many long conversations on one card: its KV cache grows ~197 KB per token of context per stream
  (51.5 GB for one 262k-token stream), against a constant ~25.4 MB state for bAttention — a measured
  ×2,032 memory ratio at 262k, capped at ÷100 in the model.
- The transformer's 1/context throughput law, anchored at 32k, predicts the 262k cell to 0.006%.
- Both measured cells run against us: the transformer was at 95.6% GPU-busy; bAttention's 64-stream
  cell was 9.5% GPU-busy on an unoptimised decode path, a grid cap rather than a ceiling.
- The current scenarios replace the measured decode path with the post-campaign ESTIMATE
  (2.5 ms/token × 64 streams); the receipt that would make it MEASURED is still pending.

Receipt file names for every measured constant are listed in the constants blocks of
`ai_capex_model.py`.

## Caveats

- Cash basis (capex not depreciated). Capitalization is a simple perpetuity (benefit ÷ discount rate;
  default 6%, ~the long bond).
- Only chips and their power are counted as savings. Buildings, power infrastructure and networking
  around the chips would also shrink (upside, not included).
- "AI spend" counts capex and power only. AI staff costs are operating expenses, never capex; they are
  not in the model, so the net AI losses shown are understated and Helarctos does not reduce them.
- No filing splits data-center spend into AI versus ordinary cloud; the accelerator share of servers
  covers part of that.
- AI revenue is the softest input: Microsoft and Amazon run-rates are disclosed; the rest are
  estimates.
- The equal-quality parameter ratio is a projection from measured models (47M–663M parameters) to
  frontier scale.
- The saving is spend no longer needed for the same AI output; firms will likely reinvest it.
- This is an analytical estimate, not investment advice.
