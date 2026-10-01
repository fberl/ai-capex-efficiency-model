# AI Capex Efficiency

Dollarizes the value of the Helarctos architecture for the six companies spending the most on AI
infrastructure (Microsoft, Alphabet, Amazon, Meta, Oracle, SpaceX), plus a grossed-up global estimate.

**Headline (FY2026, the base year; Current kernels; 262k-token average conversation; 2026-10-01 re-base):**
the six firms spend ~$782B on AI in FY2026, with AI revenue of ~$136B. Helarctos makes **~$774B/yr** of
that spend unnecessary for the same AI output (~99% of all AI spend), turning the FY2026 cash result from
~−$646B to ~+$128B (last year, FY2025: ~$360B/yr). Capitalized at 6%: ~$12.9T (FY2026 run-rate). FY2026
is the base year everywhere — the capex is set; guidance vs actual is noted per company — and FY2025
appears only as a trailing "last year" column.

Two sensitivities sit next to the headline. Holding the **data centres** at today's spend (their leases are
contracted to FY33, so that saving arrives later) gives ~$516B/yr for FY2026 (~66% of AI spend). Letting the
funded kernel programme land (**Optimized kernels**: fused-kernel targets, not receipts) gives ~$777B/yr.

The headline is dollars, not a multiple. A GPU is bought whole, so a fleet shrinks by whichever need
falls least; once that is a few hundred ×, over 99% of the chip bill is already gone and bigger
multiples only move the last fraction of a percent. The dollars are set by how much these firms spend
on chips — and, since 2026-10-01, on the servers and data centres sized by those chips.

## What is counted

"AI spend" is **property & equipment additions plus electricity**: the AI chips (GPUs, TPUs), the servers
around them (CPUs, chassis, networking), the data-centre buildings with their power & cooling and network,
and the electricity the chips draw. **No labor is counted anywhere** — AI staff are operating expenses, so
the net AI losses shown are understated and Helarctos does not reduce them. The Helarctos levers act on the
chips directly; the servers and data centres are sized by the GPUs they house, so by default they **follow
the blended chip-fleet cut** (they scale down when no longer necessary). Holding the data centres is the
sensitivity (app: sidebar checkbox; workbook: Levers C6).

## The six Helarctos levers (H1–H6)

Only six inputs are about the Helarctos architecture. Every surface (workbook and app) labels them
**H1–H6** on a **teal** background; everything else — capex, data-centre / server / accelerator /
training shares, $ per GPU, power, electricity, discount rate — is market, company or modelling data.

| Lever | Value | Shrinks | How sure |
|---|---|---|---|
| **H1** Smaller model for the same quality ★ | ×4.22 (Inputs B23; also the inference size factor, B33) | Training AND inference | PROJECTED (fits over 47M–663M-parameter models) |
| **H2** Fewer training tokens | ×4.22 (= H1) | Training | PROJECTED (compute-optimal scaling) |
| **H3** Training speed per token, same size ◆ | ×2.22 Current / ×3.97 Optimized (Levers D11/E11) | Training | step costs MEASURED at 2k/8k, modelled beyond, over a modern 8k/64k/256k curriculum (curriculum shares illustrative) / TARGET |
| **H4** Memory per live conversation | ÷2,032 (Inputs B2) | Inference GPUs | MEASURED at 262k on our test model |
| **H5** More conversations per GPU | ×256 (Inputs B28) | Inference GPUs | MEASURED per layer (2026-10-01 receipt); depth, grouped-query cache and feasible batch MODELLED at a ~90-layer frontier geometry (×1.42 on the measured 24-layer comparator) (256 resident streams vs the transformer's 1 feasible stream at 262k) |
| **H6** Decode step vs the transformer, per layer | ×0.74 at 262k (Inputs B29) | Inference GPUs | MEASURED per layer (2026-10-01 receipt); depth, grouped-query cache and feasible batch MODELLED at a ~90-layer frontier geometry (×1.42 on the measured 24-layer comparator); both per-layer steps measured including their feed-forward blocks (ours as written and uncompiled, so the ratio is a floor) |

A supporting input — **prompt-processing (prefill) speed vs the transformer**, ×10.3 Current / ×18.4
Optimized kernels (Inputs B24/B25) — is a Helarctos kernel property but not a headline lever.

Training GPU-hours fall by H1 × H2 × H3 (**×39.5**; ×70.6 under Optimized kernels). Inference tokens per
GPU **at equal model size** rise by H5 × H6 with the prompts added back,
`(1 + rq) / (rq/PF + 1/(H5 × H6))` = ×182 (×185 Optimized), where rq ≈ 0.23% is the transformer's prompt
time over its decode time (`inference_throughput()` in the model; a live formula at Inputs B34). That is
then multiplied by the **inference size factor** (= H1, ×4.22): a 4.22× smaller equal-quality model costs
~4.22× less per token, in inference exactly as in training. The inference compute lever is therefore
**×768** (×782 Optimized; Inputs B27). Inference GPUs fall by the smaller of H4 and that (×768): a GPU is
bought whole, so the fleet covers whichever need runs out first. With H4 at the measured ×2,032 the
compute lever binds, so H5, H6 and the size factor all count in the dollars.

**The decode receipt (2026-10-01).** H5 and H6 come from a per-layer measurement: one layer, bf16,
graph-captured one-token step on a 94.5 GB card; the transformer at the largest measured batch whose
full-model cache fits the card at that context, us at 256 resident streams (1 MB of state per layer
each — a grid cap, not a ceiling). Both per-layer steps are measured including their feed-forward blocks
(ours as written and uncompiled, so every ratio is a floor). The per-layer step ratio is depth-invariant, but
the transformer's cache capacity and cache-read-bound step time are not, so the levers are quoted at the
**frontier geometry of record**: ~90 layers with grouped-query attention (4 KB of cache per token per
layer), anchored on disclosed frontier depths (Kimi K3 at 93 layers, GLM-5 at 78) — not the receipt's
24-layer d2048 comparator. At the geometry of record H5 × H6 is ×4.0 at 4k, ×24 at 32k, ×114 at 128k and
×189 at 262k (the comparator reads ×364 at 262k). At 262k the transformer fits one stream (90 GiB of cache)
and its single-stream step is faster than our 256-stream step (H6 ×0.74); the gain is the 256 conversations
advancing per step (H5). This receipt supersedes the 2.5 ms-per-token × 64-stream aggregate-decode estimate
the model carried from 2026-08-31.

**Caveat on H4.** ×2,032 is measured on our test model (197 KB of transformer KV per token × 262,144
tokens = 51.6 GB vs a constant 25.4 MB Helarctos state, 2026-08-14). For frontier-size models with
grouped-query attention the model's own frontier-geometry estimate is ~×208 at 262k
(`fleet_memory_lever(262144)`), at which point memory would bind again: about −$2B on FY2026.

**Model size.** H1 is quoted at **~1T dense-equivalent**. Frontier labs no longer disclose sizes, and
almost all large models are now mixture-of-experts: Grok 5 is announced at 6T, Kimi K3 is 2.8T, but
each runs only a fraction of its parameters per token. Such a model performs roughly like a dense model
of √(total × active) size, so a 5–6T flagship with ~200B active is about 1T dense-equivalent. Using 5T
dense instead would add ~$3B to FY2026 (What Matters tab).

## Start here: Summary, Value Bridge, Levers, What Matters

The workbook and the app both open on four plain-language tabs. The engineering ledger sits behind
them as a technical appendix (Totals, Inputs, Sensitivity, CostLadder, ServingTraining, Evidence,
Methodology).

- **Summary**: the one FY2026 headline (FY2025 trailing as "last year") with the spend split into chips,
  servers, data centres and power, and the saving split the same way (chips and power ~$382B, the servers
  around the chips ~$134B, the data centres ~$258B — the last two following the fleet, or the data centres
  held); a "Current → Optimized kernels" table showing what the scenario switch changes (H3 ×2.22 → ×3.97,
  training lever ×39.5 → ×70.6, prompt speed ×10.3 → ×18.4, inference lever ×768 → ×782, GPUs after,
  dollars ~$774B → ~$777B) and why the bill barely moves; then a subordinate by-fleet view that adds to the
  headline — **Training** ~$280B and **Inference** ~$494B, each including its share of the followed servers
  and data centres (the model's attribution) — with the Helarctos levers behind each fleet; a per-company
  table; a legend.
- **Value Bridge**: a step-by-step walk from disclosed capex to the saving (spend → split the chip
  fleet into training and inference → training → inference → the servers and data centres that follow →
  result), then the levers switched on one at a time, a cross-check against the technical Totals tab,
  and the per-company engine tables (with servers / data-centre columns).

  | Step | Spend cut, FY2026 |
  |---|---|
  | Smaller model trained on fewer tokens, faster per token — training (H1–H3) | ~$141B |
  | + Fixed-size memory, more conversations per GPU advancing per decode step — inference (H4–H6) | ~$382B |
  | + the servers and data centres sized by those GPUs follow | ~$774B |

  (Spend cuts in that table are the FY2026 bridge at the current kernels.)

- **Levers**: the scenario switch (C4); the **datacenters-follow switch** (C6: 1 = follow, 0 = held);
  the six Helarctos levers with how sure we are of each, what it means and where it saves money; how
  they combine (including the inference size factor); then, separately, the **market & company data**
  the front tabs use (with live values and where to edit them, and the servers-follow switch at C34) and
  the per-company training shares (25–55%).
- **What Matters**: each input moved to a plausible low and high value, one at a time, grouped into
  **Helarctos levers** and **market & company data**, with a tornado chart (teal = Helarctos lever,
  gray = market data). Whether the **data centres follow the fleet** is now the largest single input
  (held → −$258B); the data-centre share of capex moves it ±$30–40B. Among the Helarctos levers only H1
  (with H2) swings it by $10B or more (×2 → −$26B); H3 at ×1 costs ~$9B; H4, H5 and H6 each move it by
  under $2B across their ranges, and the average conversation length moves it from ~−$4B (32k) to
  ~+$0.5B (1M, modelled).

**Cell markers** (workbook and app): **teal + H1–H6** = a Helarctos lever; **★ with an orange
border** = a high-impact input (swings FY2026 by $10B or more); **◆ with a purple fill** = one of the
only values that differ between the two scenarios.

The bridge takes no memory credit on training and prices inference on whole GPUs. Its chips-and-power
part lands within rounding of the technical Totals tab (~$382B vs ~$385B FY2026), which prices memory
(~60% of a GPU's cost) and compute (~40%) separately and does not count the followed servers and data
centres. `value_bridge()` in `ai_capex_model.py` is the Python twin of these tabs; an import-time check
asserts it reproduces the Totals engine exactly when given the same levers, and the workbook builder
re-derives every bridge total from the sheet's own formula chain and asserts it matches `value_bridge()`
and `savings_ladder()` to well under a cent before saving.

## Scenarios

Two scenarios, picked on the Levers tab, cell C4 (workbook) or in the sidebar (app). **Only the ◆ cells differ**
between them; every other input is shared.

| ◆ Input | Current kernels (default) | Optimized kernels |
|---|---|---|
| Prompt-processing (prefill) speed vs the transformer (Inputs B24/B25; supporting) | ×10.3 (banked ×3.94 kernel speed-up, MEASURED) | ×18.4 (full ×7.03, TARGET) |
| H3 Training speed per token, same model size (Levers D11/E11) | ×2.22 (MEASURED step costs, illustrative curriculum shares) | ×3.97 (fused-kernel TARGET) |
| → Training GPU-hour lever H1 × H2 × H3 | ×39.5 | ×70.6 |
| → Inference tokens per GPU at equal size, H5 × H6 + prompts (Inputs B34) | ×182 | ×185 |
| → Inference compute lever, × the size factor (Inputs B27) | ×768 | ×782 |
| → Spend made unnecessary, FY2026 (front tabs) | ~$774B | ~$777B |

Both are quoted at a 262k-token average conversation. H5 (256) and H6 (×0.74, measured per layer) are the
same in both. The bill barely moves between scenarios because the chip fleet already shrinks ~100× (1 ÷ (1 −
the ~99% fleet cut)): once that is a few hundred ×, kernel maturity changes the multiple, not the dollars —
the remaining bill is the servers and buildings that cannot shrink below one per site plus power. On the
technical Totals tab both scenarios read ~$385B. The datacenter switch (follow / held) is independent of the
scenario.

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
  rack (~$3M) is ~75–80% GPUs, consistent with the 67–79% accelerator shares. The rest of the server
  bucket is the "servers around the chips"; AI data-centre capex minus the server bucket is the
  "data centres" bucket (buildings, power & cooling, network).
- **Power per GPU:** 2.3 kW including cooling (GB200 NVL72 racks draw 120–132 kW for 72 GPUs, ×1.3
  PUE); corrected from 1.8 kW on 2026-09-29. Electricity $0.08/kWh (US industrial).
- **Memory share of GPU cost (60%):** consistent with chip teardowns (HBM ~48% + packaging ~16% of a
  B200's manufacturing cost). It only matters on the technical tabs (<$1B); the front tabs price whole
  GPUs, which is how they are bought.
- **Levers:** internal measurements and fits, each labelled MEASURED / PROJECTED / ESTIMATE / TARGET on
  the Levers and ServingTraining tabs.

## How it works

- **Per company:** `total capex × data-center share × server share × accelerator share` → accelerator
  (chip) capex → fleet → power/opex → avoided spend → capitalized value. FY2026 (the base year) first;
  FY2025 (last year) trailing.
- **Technical ledger (Totals, company tabs; legacy):** accelerators + power only, Amdahl-weighted:
  cost-weighted reduction `= 1 / (mem_share/mem_factor + (1−mem_share)/flop_factor)` ≈ ×1,225 (memory
  ×2,032, inference lever ×768), applied to chips plus the legacy datacenter-scaling share (default 0).
- **Audience engine (Summary, Value Bridge) — the headline:** training fleet ÷ GPU-hour lever (×39.5);
  inference fleet ÷ min(memory lever H4, inference compute lever) (×768); each fleet's power shrinks with it; the servers
  and data centres sized by the GPUs follow the blended fleet cut (switches default to 1).
- **Net AI economics (cash basis):** `AI revenue − AI capex − AI power`; with the architecture, add the
  spend cut.
- **Global estimate:** the named firms are grossed up by their assumed share (80%) of worldwide AI capex.

## Live app

Deploy free on [Streamlit Community Cloud](https://share.streamlit.io) — see **Deploy** below.
The app is a **tab-for-tab mirror of the workbook** (Summary, Value Bridge, Levers, What Matters,
each company, Totals, Inputs, Sensitivity, CostLadder, Serving·Training, Evidence, Methodology), with
the same colours and markers. The sidebar is grouped the same way: the scenario and the
**Hold datacenters (contracted leases to FY33)** checkbox, then the **Helarctos levers** (H4, H5, H6, and
the workload behind them — H5/H6 defaults track the conversation length from the decode receipt), then
**market & modelling data** (technical-only inputs folded away, including the legacy datacenter scaling
factor used by the Totals engine only). Per-company capex and shares are in the ✏️ panel on each company
tab; training shares on the Levers tab. Every grid recomputes live.

## Contents

| File | What it is |
|---|---|
| `app.py` | Streamlit app — interactive tab-for-tab mirror of the workbook |
| `ai_capex_model.py` | **Single source of truth** — all defaults + math (app and Excel both import it, so they can't drift) |
| `ai_capex_efficiency.py` | Generates `AI_Capex_Efficiency.xlsx` (live-formula workbook) and runs the formula-twin check |
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
established what the memory lever rests on:

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
- Both measured cells ran against us: the transformer was at 95.6% GPU-busy; bAttention's 64-stream
  cell was 9.5% GPU-busy on an unoptimised decode path, a grid cap rather than a ceiling.
- H5 and H6 moved off this cell on 2026-10-01 to the per-layer decode receipt described above (256
  streams, kernels as built today); the 24-layer step cross-checks the 2026-08-14 transformer cell to
  within 5%.

Receipt file names for every measured constant are listed in the constants blocks of
`ai_capex_model.py`.

## Caveats

- Cash basis (capex not depreciated). Capitalization is a simple perpetuity (benefit ÷ discount rate;
  default 6%, ~the long bond).
- "AI spend" is property & equipment additions plus electricity; no labor. The servers and data centres
  sized by the GPUs follow the fleet by default; holding the data centres (leases contracted to FY33)
  delays, not removes, ~$258B of the FY2026 saving.
- No filing splits data-center spend into AI versus ordinary cloud; the accelerator share of servers
  covers part of that.
- AI revenue is the softest input: Microsoft and Amazon run-rates are disclosed; the rest are
  estimates.
- The equal-quality parameter ratio is a projection from measured models (47M–663M parameters) to
  frontier scale; since 2026-10-01 it multiplies inference as well as training.
- H4 (memory, ×2,032) is measured on our test model. For frontier-size models with grouped-query
  attention the frontier-geometry estimate is ~×208 at 262k, where memory would bind again (≈ −$2B
  on FY2026). H5 and H6 are measured per layer (both feed-forward blocks included; ours uncompiled, so a
  floor); the frontier-depth geometry they are quoted at is an anchored assumption.
- H3's step costs are measured at 2k and 8k and modelled beyond; the curriculum token shares that weight
  them are illustrative knobs, not any lab's recipe.
- The saving is spend no longer needed for the same AI output; firms will likely reinvest it.
- This is an analytical estimate, not investment advice.
