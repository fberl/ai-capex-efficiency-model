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
- **MEASURED — serving-throughput ratio at a chosen operating point.** Default is the **×2.19**
  serving blend at 64k context / 10:1 input:output. Other selectable points: ×8.74 at 262k
  context, ×4.02 long-context prefill at 1M tokens (2026-07-21), ×1 parity, ×0.28 granting the
  transformer an idealized KV÷8 stack, and **×0.15 — the single-GPU training step at
  2,048-token context, where we are ×6.46 *slower***.

At 1T scale with the serving blend the lever is ×9.24 → **~20.3× cost-weighted** →
**~$159B/yr** of FY2025 spend removed across the named firms. Memory stays at the deliberate
**÷100** cap even though the *measured* full-model serving-state ratio is **×508 at 64k context
and ×2,032 at 262k** (197 KB of transformer KV per token of context per stream, d2048/24 layers
bf16, against a constant 25.4 MB recurrent state) — past ~100× the memory term is 0.6% of
residual cost and stops moving the blend.

### The 2026-08-14 decode re-base — read this before quoting the compute lever

A full-model bf16 decode measurement on one GH200 (d2048/24 layers, both families, same session,
[`deck_decode_d2048_bf16_gh200_20260814.json`]) retired three numbers this model used to carry:

| Retired | Why | Replaced by |
|---|---|---|
| transformer decode 87.9 ms/token | a growing-KV re-planning artifact in the old harness inflated it | measured GPU-busy 1.69 ms at 2k → 14.92 ms at 262k, linear in context |
| bAttention state 0.15 MB/stream | h-recurrence carry only — no M memory in it | **25.4 MB/stream** full model (24.2 MiB measured), still context-flat |
| decode ×15.9, "×53 decode" | both stood on the two rows above | **memory-ceiling aggregate ratio**, ×1.1 at 32k, ×2.20 at 64k, ×8.8 at 262k |

**The decode lever is not a latency lever.** Per generated token at a single stream the transformer
is *faster than we are* at 64k context — ~4.9 ms of GPU-busy against our ~5.7 ms flat. What it
cannot do is hold many long-context streams on one card: it re-reads its whole KV cache for every
token it emits, so once the card is full of KV its aggregate throughput falls as 1/context, while
ours is context-flat. That is the lever, and it is measured at both ends:

| Context | Transformer | bAttention | Aggregate ratio |
|---|---|---|---|
| 32,768 | 8 streams, 511.8 tok/s/GPU (16 streams OOM) | 64 streams, 585.4 tok/s/GPU | **×1.14 measured** |
| 65,536 | 255.9 tok/s/GPU | 562.1 tok/s/GPU | ×2.20 derived |
| 262,144 | 1 stream, 64.0 tok/s/GPU (2 streams OOM) | 64 streams in 1.6 GB, 562.1 tok/s/GPU | **×8.79 measured** |

The 1/context law anchored on the 32k cell predicts 63.980 tok/s at 262k against the measured
63.976 — **0.006%** — so the 64k row is arithmetic, not a guess. The lever crosses 1 at
**~29,800 tokens**: below that context the transformer serves more tokens per GPU-second than we
do, and the model says so on every surface.

Both measured cells run *against* us. The transformer is at 95.6% GPU-busy at its ceiling — no
headroom left — while bAttention's 64-stream cell is 9.5% GPU-busy on 1.6 GB of a 96 GB card, a
grid cap rather than a ceiling, and its decode ran the unoptimised Triton per-step path (the
receipt records the CUDA twin declining with `mfwd: C != 64`). The measured ratio is a lower bound
on us and an upper bound on the transformer.

**What it cost the headline.** The compute lever fell ×50.33 → ×9.24, a factor of **5.4**. The FY25
headline moved **$165B → $159B, −3.6%**. That is the model's whole thesis arriving on schedule: the
cut is `accelerator capex × (1 − 1/reduction)` and it saturates, so a 5.4× haircut on the compute
lever is worth 3.6% of the money. The stated downside row — granting the transformer an idealized
KV÷8 mature stack, which drops the blend to ×0.28 and the compute lever to ×1.18 — takes the cut to
~$109B; the headline uses the measured cells, in which neither family's decode is optimised.

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
| **Quality-matched (2026-08-14)** — default | ×9.24 at 1T scale (×4.22 FIT × ×2.19 MEASURED) | ~20.3× | ~$159B/yr | sealed quality refit + 2026-08-14 full-model decode receipt |
| Measured (H100 2026-07-21) | ×4 | ~9.4× | ~$149B/yr | parameter-matched bf16 prefill at 1M tokens |
| Ceiling (kernels mature) | ×10 | ~21.7× | ~$159B/yr | architectural ceiling |
| Memory-only (compute parity) | ×1 | ~2.5× | ~$99B/yr | credits nothing to compute |

The FY25 cut spans only ~$99B–$159B across a 10× spread in the compute lever, because the cut is
`accelerator capex × (1 − 1/reduction)` and saturates. **Doubling the multiple is worth a few $B.**
That is the honest shape of the result and the reason the model refuses multiplicative headlines.

### Receipts for the 2026-08-14 scenario

Every constant in the `2026-08-14 SEALED SCALING RECEIPTS` block of `ai_capex_model.py` names its
source file, and `python ai_capex_model.py` re-derives the published parameter-matching curve from
the fit coefficients (asserted to <1e-6 pp), checks the 1/context decode law against **both** measured
aggregate cells, and asserts the published serving-blend constants equal what `serving_blend()` computes.

| What | Receipt |
|---|---|
| Quality fits, crossover, parameter-matching curve | `bdm/docs/deck/build/quality_fit_v4.json` (built by `docs/deck/build/build_slides_v4.py`) |
| Narrative receipt for the above | `bdm/docs/deck/README.md`, section `## v4` |
| **Full-model decode: per-token cost, serving state, concurrency ceilings** | `bdm/docs/deck_scaling/deck_decode_d2048_bf16_gh200_20260814.json` (schema `deck_decode_v1`), figure `context_decode_bf16_gh200_20260814.png` |
| Memory walls, 8-GPU training scaling, prefill lane | `bdm/docs/deck/build/derived.json` |
| Single-GPU training-step gap (the measurement against us) | `bdm/docs/deck/build/dtype_trio_v4.json` |
| Serving blend (×2.19) over those receipts | `serving_blend()` in `ai_capex_model.py`; `_selfcheck()` asserts the published constants equal what it computes |

## How it works

- **Engine:** cost-weighted reduction `= 1 / (mem_share/mem_factor + (1−mem_share)/flop_factor)` — ≈20.3× quality-matched at 1T (flop ×9.24), ≈9.4× at the 2026-07-21 measured lever (flop ×4), ≈22× ceiling (flop ×10).
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
