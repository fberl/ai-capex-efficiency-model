# AI Capex Efficiency

Dollarizes the value of the Helarctos architecture for the six companies spending the most on AI
infrastructure (Microsoft, Alphabet, Amazon, Meta, Oracle, SpaceX), plus a grossed-up global estimate.

**Headline (Current kernels, 262k-token average conversation):** the six firms spend ~$782B on AI in
FY2026 and earn ~$136B from it. Helarctos makes **~$377–385B/yr** of that spend unnecessary for the
same AI output (~48% of all AI spend, over 99% of the AI-chip bill), shrinking the FY2026 cash burn from
~−$646B to ~−$269B. FY2025: ~$159–163B/yr. Capitalized at 6%: ~$6.3–6.4T (FY2026 run-rate).

The headline is dollars, not a multiple. A GPU is bought whole, so a fleet shrinks by whichever need
falls least; once that is a few hundred ×, over 99% of the chip bill is already gone and bigger
multiples only move the last fraction of a percent. The dollars are set by how much these firms spend
on chips.

## The six Helarctos levers (H1–H6)

Only six inputs are about the Helarctos architecture. Every surface (workbook and app) labels them
**H1–H6** on a **teal** background; everything else — capex, data-centre / server / accelerator /
training shares, $ per GPU, power, electricity, discount rate — is market, company or modelling data.

| Lever | Value | Shrinks | How sure |
|---|---|---|---|
| **H1** Smaller model for the same quality ★ | ×4.22 (Inputs B23) | Training | PROJECTED |
| **H2** Fewer training tokens | ×4.22 (= H1) | Training | PROJECTED |
| **H3** Training speed per token, same size ◆ | ×1 (Levers D11/E11) | Training | ASSUMPTION (no credit) |
| **H4** Memory per live conversation | ÷2,032 (Inputs B2) | Inference GPUs | MEASURED at 262k on our test model |
| **H5** More conversations per GPU | ×64 (Inputs B28) | Inference GPUs | MEASURED (64 vs 1; a 2nd transformer stream OOMs) |
| **H6** Faster decode per token | ×6.25 (Inputs B29) | Inference GPUs | ESTIMATE (15.6 vs 2.5 ms; ×2.6 measured on older kernels) |

A supporting input — **prompt-processing (prefill) speed vs the transformer**, ×10.3 Current / ×18.4
Optimized kernels (Inputs B24/B25) — is a Helarctos kernel property but not a headline lever; it is the
only inference input that differs by scenario.

Training GPU-hours fall by H1 × H2 × H3 (×18). Inference tokens per GPU rise by H5 × H6 with the prompts
added back, `(1 + rq) / (rq/PF + 1/(H5 × H6))` = ×368 (×382 Optimized), where rq ≈ 0.23% is the
transformer's prompt time over its decode time (`inference_throughput()` in the model; a live formula
at Inputs B27). Inference GPUs fall by the smaller of H4 and that (×368): a GPU is bought whole, so the
fleet covers whichever need runs out first. With H4 at the measured ×2,032, tokens per GPU binds, so
faster decode (H6) counts in the dollars. Inference cost does not depend on model size, so H1 helps
training only.

**Caveat on H4.** ×2,032 is measured on our test model (197 KB of transformer KV per token × 262,144
tokens = 51.6 GB vs a constant 25.4 MB Helarctos state, 2026-08-14). For frontier-size models with
grouped-query attention the model's own frontier-geometry estimate is ~×208 at 262k
(`fleet_memory_lever(262144)`), at which point memory would bind again: about −$0.5B on FY2026.

**Model size.** H1 is quoted at **~1T dense-equivalent**. Frontier labs no longer disclose sizes, and
almost all large models are now mixture-of-experts: Grok 5 is announced at 6T, Kimi K3 is 2.8T, but
each runs only a fraction of its parameters per token. Such a model performs roughly like a dense model
of √(total × active) size, so a 5–6T flagship with ~200B active is about 1T dense-equivalent. Using 5T
dense instead would add ~$3.6B to FY2026 (What Matters tab).

## Start here: Summary, Value Bridge, Levers, What Matters

The workbook and the app both open on four plain-language tabs. The engineering ledger sits behind
them as a technical appendix (Totals, Inputs, Sensitivity, CostLadder, ServingTraining, Evidence,
Methodology).

- **Summary**: the FY2026 / FY2025 headline; where the FY2026 saving comes from — **Training**
  (~$136B) and **Inference** (~$241B), each including the power its GPUs would have drawn — with the
  Helarctos levers behind each fleet listed underneath it; a per-company table; a legend.
- **Value Bridge**: a step-by-step walk from disclosed capex to the saving (spend → split the chip
  fleet into training and inference → training → inference → result), then the levers switched on
  one at a time, a cross-check against the technical Totals tab, and the per-company engine tables.

  | Step | Spend cut, FY2026 |
  |---|---|
  | Smaller model trained on fewer tokens — training (H1–H3) | ~$136B |
  | + Fixed-size memory, more conversations per GPU, faster decode — inference (H4–H6) | ~$377B |

- **Levers**: the scenario switch (C4); the six Helarctos levers with how sure we are of each, what
  it means and where it saves money; how they combine; then, separately, the **market & company
  data** the front tabs use (with live values and where to edit them) and the per-company training
  shares (25–55%).
- **What Matters**: each input moved to a plausible low and high value, one at a time, grouped into
  **Helarctos levers** and **market & company data**, with a tornado chart (teal = Helarctos lever,
  gray = market data). The chip-share inputs (server and accelerator shares, ±15% → ±$50–57B) move the
  answer more than any Helarctos lever. Only H1 (with H2) swings it by $10B or more; H4, H5 and H6 each
  move it by under $2B across their ranges, and a 32k-token average conversation costs ~$4B.

**Cell markers** (workbook and app): **teal + H1–H6** = a Helarctos lever; **★ with an orange
border** = a high-impact input (swings FY2026 by $10B or more); **◆ with a purple fill** = one of the
only values that differ between the two scenarios.

The bridge takes no memory credit on training and prices inference on whole GPUs. It lands within ~2%
of the technical Totals tab ($377B vs $385B FY2026), which prices memory (~60% of a GPU's cost) and
compute (~40%) separately. `value_bridge()` in `ai_capex_model.py` is the Python twin of these tabs;
an import-time check asserts it reproduces the Totals engine exactly when given the same levers.

## Scenarios

Two scenarios, picked on the Levers tab, cell C4 (workbook) or in the sidebar (app). **Only the ◆ cells differ**
between them; every other input is shared.

| ◆ Input | Current kernels (default) | Optimized kernels |
|---|---|---|
| Prompt-processing (prefill) speed vs the transformer (Inputs B24/B25; supporting) | ×10.3 | ×18.4 |
| H3 Training speed per token, same model size (Levers D11/E11) | ×1 | ×1 (no credit yet) |
| → Inference tokens per GPU from H5 × H6 + prompts (Inputs B27; model size doesn't enter) | ×368 | ×382 |

Both are quoted at a 262k-token average conversation. H5 (64 streams) and H6 (the 2.5 ms/token
aggregate-decode ESTIMATE, measurement pending) are the same in both. Current kernels = prefill at the
banked kernel speed-up (×3.94); Optimized kernels = the funded kernel programme lands (×7.03, TARGET).
Prompts are only ~0.23% of the transformer's time at 262k, so the dollars barely move (FY2026 ~$376.83B
vs ~$376.86B front tabs; ~$385.06B vs ~$385.08B Totals). The scenario would matter if the optimized
kernels also sped up training — enter that under H3.

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
  share of additions (Microsoft 45% → 62%, Alphabet ~60%) are consistent with the model. A GB200 NVL72
  rack (~$3M) is ~75–80% GPUs, consistent with the 67–79% accelerator shares.
- **Power per GPU:** 2.3 kW including cooling (GB200 NVL72 racks draw 120–132 kW for 72 GPUs, ×1.3
  PUE); corrected from 1.8 kW on 2026-09-29. Electricity $0.08/kWh (US industrial).
- **Memory share of GPU cost (60%):** consistent with chip teardowns (HBM ~48% + packaging ~16% of a
  B200's manufacturing cost). It only matters on the technical tabs (<$1B); the front tabs price whole
  GPUs, which is how they are bought.
- **Levers:** internal measurements and fits, each labelled MEASURED / PROJECTED / ESTIMATE / TARGET on
  the Levers and ServingTraining tabs.

## How it works

- **Per company:** `total capex × data-center share × server share × accelerator share` → accelerator
  (chip) capex → fleet → power/opex → avoided spend → capitalized value. FY2025 actual + FY2026.
- **Technical engine (Totals, company tabs):** cost-weighted reduction
  `= 1 / (mem_share/mem_factor + (1−mem_share)/flop_factor)` ≈ 724×.
- **Audience engine (Summary, Value Bridge):** training fleet ÷ GPU-hour lever (×18); inference fleet ÷
  min(memory lever H4, tokens per GPU from H5 × H6) (×368); each fleet's power shrinks with it and is folded into its
  number.
- **Net AI economics (cash basis):** `AI revenue − AI capex − AI power`; with the architecture, add the
  spend cut.
- **Global estimate:** the named firms are grossed up by their assumed share (80%) of worldwide AI capex.

## Live app

Deploy free on [Streamlit Community Cloud](https://share.streamlit.io) — see **Deploy** below.
The app is a **tab-for-tab mirror of the workbook** (Summary, Value Bridge, Levers, What Matters,
each company, Totals, Inputs, Sensitivity, CostLadder, Serving·Training, Evidence, Methodology), with
the same colours and markers. The sidebar is grouped the same way: the scenario, then the
**Helarctos levers** (H4, H5, H6, and the workload behind them), then **market & modelling data**
(technical-only inputs folded away). Per-company capex and shares are in the ✏️ panel on each company
tab; training shares on the Levers tab. Every grid recomputes live.

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

## Colour / marker convention (workbook and app)

- **teal, H1–H6** — a Helarctos lever (the only inputs about the architecture)
- **yellow** — a market, company or modelling assumption you can edit
- **green** — disclosed data, from filings or markets
- **blue** — a formula
- **◆ purple** — differs between Current kernels and Optimized kernels
- **★ orange border** — high impact (moves the FY2026 saving by $10B or more)

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
  ×2,032 memory ratio at 262k, used as H4 (was capped at ÷100 until 2026-09-29).
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
- H4 (memory, ×2,032) is measured on our test model. For frontier-size models with grouped-query
  attention the frontier-geometry estimate is ~×208 at 262k, where memory would bind again (≈ −$0.5B
  on FY2026). H6 (faster decode) is an estimate until the aggregate-decode receipt lands (×2.6 measured
  on our older kernels, ≈ −$0.8B).
- The saving is spend no longer needed for the same AI output; firms will likely reinvest it.
- This is an analytical estimate, not investment advice.
