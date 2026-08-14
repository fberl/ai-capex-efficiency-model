"""Single source of truth for the AI-Capex-Efficiency model.

Both the Excel generator (ai_capex_efficiency.py) and the Streamlit app (app.py)
import their DEFAULTS and LOGIC from here, so the spreadsheet and the web app
never drift.

Engine: a GPU is ~60% memory / ~40% compute by cost. Cutting memory 100x and
FLOPs 10x gives a residual cost of ~0.6%+4% -> ~22x cost-weighted reduction,
floored by the less-reduced component (compute).

Scenario history (each one keeps working; the app defaults to the latest):
  2026-07-21  single-H100 parameter-matched prefill      -> flop x4
  2026-08-14  sealed quality refit + serving receipts    -> see SCENARIOS below

2026-08-14 DECODE RE-BASE. A full-model bf16 decode measurement (one GH200, both
families, same session) retired the transformer's 87.9 ms/token decode cost (a
growing-KV re-planning harness artifact) and the 0.15 MB bAttention serving state
(an h-carry-only proxy with no M memory). The decode lever is no longer a
per-token LATENCY ratio -- per token at one stream the transformer is FASTER at
64k -- it is a MEMORY-CEILING ratio: per-GPU aggregate serving throughput, set by
how many concurrent streams a card can hold. Compute lever x50.33 -> x9.24; the
FY25 headline moved $165B -> $159B, which is the Amdahl saturation this model
exists to show. See DECODE_THROUGHPUT_20260814 and SERVING_BLEND_20260814.
"""

import math

# ---- default global assumptions ------------------------------------------------
GLOBALS = {
    "mem_factor": 100,  # memory reduction (x)
    "flop_factor": 10,  # FLOPs reduction (x)
    "mem_share": 0.60,  # memory share of GPU cost (BOM)
    "opex_reduction_override": None,  # energy reduction: None = derive (= cost-weighted reduction); set a number to override
    "discount_rate": 0.10,  # perpetuity capitalization rate
    "gpu_cost": 50000,  # fully-loaded $ per GPU
    "wall_power_kw": 1.8,  # wall power per GPU (incl PUE + node overhead)
    "elec_rate": 0.08,  # $/kWh
    "cooling_overhead": 0.25,  # non-power running cost as fraction of electricity
    "fleet_life_yr": 4,  # AI-GPU depreciation life
    "spacex_mktcap": 1770,  # $B
    "dc_scale": 0.0,  # datacenter scaling factor: 0=accel-only, 1=whole DC scales
    "named_share_of_global": 0.80,  # ESTIMATE: named firms' share of GLOBAL AI capex
}

# ---- default companies ---------------------------------------------------------
# fy25/fy26 = (total_capex_$B, infra_share, server_share, accel_share)
# ai_rev = (FY25_$B, FY26_$B)
COMPANIES = [
    {
        "name": "Microsoft",
        "fy25": (88.0, 1.00, 0.50, 0.75),
        "fy26": (190, 1.00, 0.67, 0.75),
        "mcap": 3700,
        "ai_rev": (30, 50),
        "basis": "FY25 capex incl leases ~$88B [DISCLOSED proxy]; FY26 ~$190B CY26 guide. "
        "Short-lived 'half'->'two-thirds' (CFO). Accel ~75% [BOM]. AI rev $37B run-rate (Q3 FY26).",
        "sources": [
            (
                "FY25 capex >$88B incl leases; server life 2-6yr (FY25 10-K)",
                "https://www.sec.gov/Archives/edgar/data/789019/000095017025100235/msft-20250630.htm",
            ),
            (
                "FY26 ~$190B guide; 'two-thirds' short-lived = GPUs+CPUs (Q3 FY26 call)",
                "https://www.fool.com/earnings/call-transcripts/2026/04/29/microsoft-msft-q3-2026-earnings-transcript/",
            ),
            (
                "AI revenue $37B annual run-rate (Q3 FY26)",
                "https://news.alphastreet.com/microsoft-msft-q3-fy2026-azure-hits-40-growth-as-ai-business-reaches-37-billion-run-rate/",
            ),
            (
                "~485k Hopper GPUs / ~$31B in 2024 (Omdia) - cross-checks accel capex",
                "https://techcrunch.com/2024/12/18/microsoft-bought-nearly-500000-nvidia-hopper-chips-this-year/",
            ),
        ],
    },
    {
        "name": "Alphabet",
        "fy25": (91.4, 1.00, 0.60, 0.67),
        "fy26": (185, 1.00, 0.60, 0.67),
        "mcap": 2600,
        "ai_rev": (25, 40),
        "basis": "FY25 capex $91.4B [DISCLOSED]; FY26 $180-190B guide. 60% servers / 40% DC (CFO). "
        "Accel ~67% [BOM]. AI rev ESTIMATE (Cloud $70B run-rate, AI subset).",
        "sources": [
            (
                "FY25 capex $91.4B; CFO 60% servers / 40% DC+net (Q4'25 call)",
                "https://www.fool.com/earnings/call-transcripts/2026/02/04/alphabet-googl-q4-2025-earnings-call-transcript/",
            ),
            (
                "FY26 guide $180-190B (Q1'26)",
                "https://www.cnbc.com/2026/04/29/alphabet-googl-q1-2026-earnings.html",
            ),
            (
                "Server life 6yr; TPU at-cost TCO 20-50% below NVIDIA (SemiAnalysis)",
                "https://newsletter.semianalysis.com/p/tpuv7-google-takes-a-swing-at-the",
            ),
        ],
    },
    {
        "name": "Amazon",
        "fy25": (128.3, 0.68, 0.65, 0.67),
        "fy26": (200, 0.70, 0.65, 0.67),
        "mcap": 2400,
        "ai_rev": (12, 22),
        "basis": "FY25 cash capex $128.3B [DISCLOSED]; FY26 ~$200B. AWS ~68-70% -> strips fulfillment. "
        "Accel ~67% [BOM]. AI rev >$15B AWS run-rate (Q1 FY26, Jassy).",
        "sources": [
            (
                "FY25 cash capex $128.3B; AWS 67.8% of net P&E adds; server life 6->5yr (FY25 10-K)",
                "https://www.sec.gov/Archives/edgar/data/1018724/000101872426000004/amzn-20251231.htm",
            ),
            (
                "~$200B 2026 capex, 'predominantly AWS' (Q4'25 call)",
                "https://www.fool.com/earnings/call-transcripts/2026/02/05/amazon-amzn-q4-2025-earnings-call-transcript/",
            ),
            (
                "AWS AI revenue >$15B annual run-rate (Q1'26, Jassy)",
                "https://www.bnnbloomberg.ca/business/artificial-intelligence/2026/04/09/amazon-cloud-units-ai-revenue-run-rate-exceeds-us15-billion-in-first-quarter-ceo-says/",
            ),
        ],
    },
    {
        "name": "Meta",
        "fy25": (72.2, 1.00, 0.65, 0.74),
        "fy26": (135, 1.00, 0.65, 0.74),
        "mcap": 1900,
        "ai_rev": (4, 8),
        "basis": "FY25 capex incl leases $72.2B [DISCLOSED]; FY26 $125-145B. Accel ~74% [BOM]. "
        "AI rev DIRECT est ~$4B -- Meta's real AI payoff is INDIRECT ad-uplift (~$20B), not counted.",
        "sources": [
            (
                "FY25 capex $72.2B incl finance leases (Q4/FY25 release)",
                "https://investor.atmeta.com/investor-news/press-release-details/2026/Meta-Reports-Fourth-Quarter-and-Full-Year-2025-Results/default.aspx",
            ),
            (
                "FY26 $125-145B guide; raise driven by HBM pricing (Q1'26 call)",
                "https://www.fool.com/earnings/call-transcripts/2026/04/29/meta-meta-q1-2026-earnings-call-transcript/",
            ),
            (
                ">1.3M GPUs by end-2025 (Zuckerberg) - cross-checks accel capex",
                "https://techcrunch.com/2025/01/24/mark-zuckerberg-says-meta-will-have-1-3m-gpus-for-ai-by-year-end/",
            ),
            (
                "Server life 5.5yr (-$2.9B depreciation)",
                "https://www.thestack.technology/meta-extends-server-life-again-saving-it-2-9-billion/",
            ),
            (
                "~25% of capex to NVIDIA (Bloomberg) - accel-share basis",
                "https://finance.yahoo.com/news/big-techs-spending-drove-nvidias-rise-154027146.html",
            ),
        ],
    },
    {
        "name": "Oracle",
        "fy25": (21.2, 0.85, 0.75, 0.70),
        "fy26": (50, 0.90, 0.75, 0.70),
        "mcap": 700,
        "ai_rev": (8, 15),
        "basis": "FY25 capex $21.2B [DISCLOSED 10-K]; FY26 ~$50B guide (rev $67B). Capex ~all OCI/GPU "
        "data centers. Accel ~70% [BOM]. AI rev = OCI/GPU cloud (IaaS ~$12B r/r Q4 FY25; AI subset, EST).",
        "sources": [
            (
                "FY25 capex $21.2B; FY26 ~$50B guide on $67B rev (CIO Dive)",
                "https://www.ciodive.com/news/oracle-capex-spike-cloud-ai-data-center/807721/",
            ),
            (
                "OCI run-rate ~$27B exiting FY25; IaaS $3.0B Q4 +52%; RPO $138B (Q4 FY25)",
                "https://futurumgroup.com/insights/oracle-delivers-q4-fy-2025-results-with-27-cloud-growth-rpo-hits-138-billion/",
            ),
            (
                "FY26 OCI +77% to ~$18B; RPO $523B +433% on AI deals (Q2 FY26)",
                "https://futurumgroup.com/insights/oracle-q2-fy-2026-cloud-grows-capex-rises-for-ai-buildout/",
            ),
        ],
    },
    {
        "name": "SpaceX",
        "fy25": (12.7, 1.00, 0.79, 1.00),
        "fy26": (18, 1.00, 0.79, 1.00),
        "mcap": 1770,
        "ai_rev": (0.3, 1),
        "basis": "FY25 S-1 AI capex $12.7B [DISCLOSED]; FY26 ~$18B (ESTIMATE). ~all accelerator (greenfield). "
        "AI rev ~0 (nascent; Anthropic orbital compute is future).",
        "sources": [
            (
                "S-1 registration statement (SEC EDGAR)",
                "https://www.sec.gov/Archives/edgar/data/1181412/000162828026036936/spaceexplorationtechnologi.htm",
            ),
            (
                "S-1 financials & AI breakdown (Hargreaves Lansdown)",
                "https://www.hl.co.uk/news/inside-spacexs-ipo-filing-revenue-starlink-ai-and-key-financials",
            ),
            (
                "$12.7B AI capex = ~60% of capex; COLOSSUS buildout (Investing.com)",
                "https://www.investing.com/analysis/spacex-is-quietly-becoming-an-ai-compute-and-orbital-infrastructure-company-200681776",
            ),
            (
                "AI1 orbital data centers; Anthropic as compute customer (DCD)",
                "https://www.datacenterdynamics.com/en/news/spacex-ipo-musks-firm-set-to-launch-first-orbital-data-center-ai1-satellites-in-2027-will-put-compute-on-starlink-craft/",
            ),
        ],
    },
]


# ================================================================================
# 2026-08-14 SEALED SCALING RECEIPTS
# ================================================================================
# Receipt authority: /home/fberl/bdm/docs/deck/build/quality_fit_v4.json, written
# by docs/deck/build/build_slides_v4.py; the "## v4" section of
# /home/fberl/bdm/docs/deck/README.md is its narrative receipt.
#
# Naming: bAttention on reader-facing surfaces; the receipts use the internal
# name BDM.
#
# TWO KINDS OF NUMBER live in this block and every surface must keep them apart:
#
#   FIT-DERIVED   the parameter-matching curve. A projection of two OLS fits,
#                 each over 4 measured rungs spanning 47M-663M params. The
#                 receipt says it in its own words: "A projection of the two
#                 fits, not a measurement." Everything past ~1B params is
#                 extrapolation, and the quoted trillion-parameter point is
#                 three orders of magnitude beyond the top measured rung.
#   MEASURED      the serving-state arithmetic (KV bytes per token of context vs
#                 a context-flat decode carry) and the single-GPU training-step
#                 gap, which runs AGAINST bAttention and is carried here for
#                 exactly that reason.

# ---- quality fits: ln(bpb) = a + b*ln(params), OLS per family ------------------
QUALITY_FIT_20260814 = {
    "form": "ln(bpb) = a + b*ln(params), OLS per family over the sealed rope-convention ladder",
    "status": "FIT-DERIVED",
    "battn": {"a": 2.413386876447594, "b": -0.12534052380844748, "rungs": 4},
    "tf": {"a": 1.9582862071595213, "b": -0.1023400790322208, "rungs": 4},
    # the measured ladder the fits stand on (params at each rung)
    "battn_rung_params": (56056320, 141586176, 314340480, 579841536),
    "tf_rung_params": (47803392, 136641024, 337713408, 662676480),
    # where the two fits cross: below it the transformer is ahead, above it we are
    "crossover_params": 391933596.2,
    "crossover_ci95": (259177028.7, 592691198.8),
    # the top rung is a measured pair, not a fit: bAttention reaches within 0.51%
    # of the transformer's bits-per-byte on 12.5% fewer parameters
    "top_pair": {
        "battn_params": 579841536, "battn_bpb": 0.894186824913843,
        "tf_params": 662676480, "tf_bpb": 0.889691062993173,
        "param_pct": -12.5, "bpb_pct": +0.505,
    },
    "receipt": "bdm/docs/deck/build/quality_fit_v4.json",
}

# The parameter-matching curve as the receipt publishes it: the bAttention
# parameter count whose FITTED bpb equals the transformer fit's, as a percentage
# of the transformer's. Used as a cross-check on param_matching_fraction().
PARAM_MATCHING_PCT_20260814 = {
    391933596: 100.0,
    1_000_000_000: 84.20793235593338,
    10_000_000_000: 55.18859584465115,
    100_000_000_000: 36.16976484388941,
    1_000_000_000_000: 23.705112783532595,
    10_000_000_000_000: 15.535969738960972,
}

# The deployment scales the app exposes on its scale slider.
DEPLOYMENT_SCALES = (
    (391933596, "392M (fit crossover)"),
    (1_000_000_000, "1B"),
    (10_000_000_000, "10B"),
    (100_000_000_000, "100B"),
    (1_000_000_000_000, "1T"),
    (10_000_000_000_000, "10T"),
)
DEFAULT_DEPLOYMENT_SCALE = 1_000_000_000_000

# ---- serving state: MEASURED (FULL-MODEL scope, re-based 2026-08-14) -----------
# The transformer's serving state grows with every token of context; ours does
# not. Same session, same box, same dtype, both families the whole model.
#
# RE-BASE NOTE. The figure this model used to carry -- a 0.15 MB bAttention decode
# carry -- was an h-recurrence-carry PROXY that did not include the M memory. The
# full-model state is 25.4 MB/stream, ~167x larger, and it is still CONSTANT in
# context, which is the whole claim. Both scopes are kept below so no surface can
# quote the proxy as if it were the model.
SERVING_STATE_20260814 = {
    "status": "MEASURED",
    "config": "d2048 / 24 layers / bf16, 1 x GH200 96 GB (94.5 GiB usable)",
    # 51,539,607,552 B of KV for one 262,144-token stream = 196,608 B/token
    "tf_kv_mb_per_token_per_stream": 196608 / 1e6,  # = 0.196608 MB (196.6 KB) per token of context
    "battn_state_mb_per_stream": 25362432 / 1e6,  # = 25.362 MB FULL MODEL (h + M), CONSTANT in context
    "battn_state_mb_per_stream_h_carry_proxy": 0.15234375,  # RETIRED as a headline: h-carry only, no M memory
    "battn_state_scope": "FULL MODEL: the per-layer (h, M) state carry, 25.4 MB/stream (24.2 MiB measured), "
    "context-flat. The 0.15 MB figure this model used to quote was an h-carry-only proxy with no M memory "
    "and is retired as a headline",
    "tf_measured_point": "51.5 GB of KV for ONE 262,144-token stream (196.6 KB per token of context per "
    "stream); a 2nd stream OOMs the 96 GB card, and 16 streams OOM it at 32,768 tokens",
    "battn_measured_point": "64 concurrent 262,144-token streams in 1.62 GB of state",
    "receipt": "bdm/docs/deck_scaling/deck_decode_d2048_bf16_gh200_20260814.json (schema deck_decode_v1)",
}

# ---- single-GPU training step: MEASURED, AND IT RUNS AGAINST US ----------------
# The honest counterweight to everything above. At a short training context the
# transformer's step is still much cheaper on one card.
STEP_GAP_20260814 = {
    "status": "MEASURED",
    "config": "d1536 / 27 layers, batch 32 x block 2048 (65,536 tokens/step), fp16, 1 x GH200, 601-step protocol",
    "battn_ms_per_step": 8979.3,
    "tf_ms_per_step": 1389.0,
    "gap_against_battn": 8979.3 / 1389.0,  # = 6.46x, i.e. we are SLOWER
    "note": "a 2,048-token training configuration, not a serving workload; the kernel campaign is the line item against it",
    "receipt": "bdm/docs/deck/build/dtype_trio_v4.json",
}

# ---- multi-GPU training scaling: MEASURED --------------------------------------
TRAINING_SCALING_20260814 = {
    "status": "MEASURED",
    "battn_x8_best_load": 6.272772372602131,
    "tf_x8_best_load": 3.697557217680221,
    "note": "8 GPUs, each family against its OWN 1-GPU run at its own best measured load; component scope",
    "receipt": "bdm/docs/deck/build/derived.json (scaling_1to8_best, matched_load_curve)",
}

# ---- decode: MEASURED per-GPU SERVING THROUGHPUT -------------------------------
# THE 2026-08-14 RE-BASE. The decode lever used to be a per-token-LATENCY ratio
# resting on a transformer decode cost of 87.9 ms/token. That number was inflated
# by a growing-KV re-planning artifact in the old harness, and the "53x decode"
# and "decode x15.9" figures built on it are RETIRED.
#
# The honest per-token latency picture, from the new receipt's CUDA kernel traces
# (single stream, GPU-busy, d2048/24L bf16 on one GH200):
#
#     ctx        2,048     8,192    32,768   262,144
#     transformer 1.69 ms  1.99 ms   3.26 ms   14.92 ms   (linear: KV re-read)
#     bAttention  5.69 ms  5.75 ms   5.75 ms    5.76 ms   (flat)
#
# OLS over those four transformer rows gives gpu_busy_us = 1584.0 + 0.050874*T
# (worst residual 0.43%, exact at 262k), so at the scenario's 64k context the
# transformer costs ~4.92 ms/token against our 5.74 ms flat. PER-TOKEN AT ONE
# STREAM WE ARE BEHIND AT 64k -- the ratio is x0.86, in the transformer's favour.
#
# So the decode lever is NOT a latency lever. It is a MEMORY-CEILING lever, and
# the serving-relevant quantity is per-GPU AGGREGATE THROUGHPUT: how many
# concurrent streams fit, times how fast each is served.
#
#   * The transformer re-reads its whole KV cache every decoded token, so once
#     the card is full of KV its aggregate rate is bandwidth-floored at
#     BW / (kv_bytes_per_token * T) -- i.e. INVERSELY LINEAR IN CONTEXT.
#   * Ours is context-flat because the state is context-flat.
#
# Two independent measured cells pin that law:
#
#     ctx  32,768 :   8 streams -> 511.84 tok/s/GPU   (16 streams OOM)
#     ctx 262,144 :   1 stream  ->  63.98 tok/s/GPU   ( 2 streams OOM)
#
# At 262k the transformer is at its CEILING exactly -- 1 runs, 2 OOM, the integers
# are adjacent. At 32k it is not: 8 ran and 16 OOMed, 9..15 were never run, so the
# ceiling there is only bracketed (8 <= c < 16). See "ratio_32k_bounded" below --
# the 32k cell is a RANGE, near parity, and nothing here leans on it.
#
# 511.84 * 32768/262144 = 63.980 vs the measured 63.976 -- the 1/T law reproduces
# the far cell to 0.006%, so interpolating it to 64k is arithmetic, not a guess.
DECODE_THROUGHPUT_20260814 = {
    "status": "MEASURED",
    "config": "d2048 / 24 layers / bf16, 1 x GH200 96 GB; autoregressive decode, prefill untimed",
    # transformer: the anchor cell, at its KV memory ceiling
    "tf_anchor_ctx": 32768,
    "tf_anchor_tokens_per_s_per_gpu": 511.84161509663613,  # 8 streams; 16 OOM
    "tf_far_cell": {"ctx": 262144, "streams": 1, "tokens_per_s": 63.976137240450086, "next_streams_status": "oom"},
    "tf_law": "aggregate tok/s/GPU = anchor * (32768 / ctx): the KV budget is re-read once per decoded "
    "token, so filling the card with KV and serving at the bandwidth floor makes throughput 1/T",
    # bAttention: context-flat, measured at 64 concurrent streams
    "battn_tokens_per_s_per_gpu": 562.0835756046827,  # 64 streams @ 262,144 ctx -- the LOWER of the two cells
    "battn_cell_32k": 585.4143612476083,  # 64 streams @ 32,768 ctx
    "battn_flat_note": "context-flat (-4.0% from 32k to 262k); the lower 262k cell is used everywhere",
    # measured aggregate ratios -- the two cells the derived curve must reproduce
    "measured_ratio_262k": 562.0835756046827 / 63.976137240450086,  # = 8.786, ceiling EXACT
    "measured_ratio_32k": 585.4143612476083 / 511.84161509663613,  # = 1.144 AT 8 STREAMS
    # ---- the 32k cell is BOUNDED, not a point (adversarial audit, 2026-08-14:
    # bdm/docs/deck_scaling/decode_rebase_adversarial_audit_20260814.md, finding D5).
    # The receipts prove 8 <= ceiling < 16 at 32k; decode-only arithmetic admits ~15
    # streams and the measuring harness's own prefill activations ~11. The other end
    # of the cell is physics: a 32k decode step re-reads 6.44 GB of KV per stream, so
    # at the measured 3.87 TB/s KV streaming rate NO stream count lets the transformer
    # pass ~601 tok/s at that context. Both ends are therefore known and the truth is
    # between them: 0.97-1.14, i.e. NEAR PARITY at 32k. Only the upper end was measured.
    "tf_kv_streaming_bytes_per_s": 3.87e12,  # top of the audit's measured 3.84-3.87 TB/s window
    "tf_32k_aggregate_bandwidth_cap_tokens_per_s": 3.87e12 / (196608.0 * 32768.0),  # = 600.7
    "ratio_32k_bounded": (
        585.4143612476083 / (3.87e12 / (196608.0 * 32768.0)),  # = 0.975 -> 0.97
        585.4143612476083 / 511.84161509663613,  # = 1.144, the measured 8-stream end
    ),
    "ratio_32k_label": "0.97-1.14 bounded (near parity): the transformer's 511.8 tok/s was measured at 8 "
    "streams, which the receipts place at 8 <= ceiling < 16 (analytic ~11-15); at its true depth its 32k "
    "aggregate is bandwidth-capped near 601 tok/s against our 585.4. Only the 1.14 end was measured. The "
    "262k cell (x8.79) is UNAFFECTED -- there the ceiling is exactly 1 stream, with 2 OOM",
    "anchor_note": "the 1/T law is anchored on the MEASURED 8-stream 32k cell. If the transformer's true "
    "32k depth is 11-15 streams its aggregate there rises toward the ~601 tok/s bandwidth cap, so this "
    "anchor can be low by up to ~17% NEAR 32k -- a direction that flatters us. The law is pinned "
    "out-of-sample at 262k, where the ceiling IS exactly one stream and the prediction lands within "
    "0.006%, and every scenario context (64k, 262k) sits at or beyond that pin. Do not quote the ~32k "
    "end of this curve; quote the range",
    "conservatism": "BOTH sides of these cells run against us. The transformer is at 95.6% GPU-busy at its "
    "ceiling cell (it has no headroom left), while bAttention's 64-stream cell is only 9.5% GPU-busy and "
    "1.62 GB of state on a 96 GB card -- a grid cap, not a ceiling. Its decode also ran the unoptimised "
    "Triton per-step path: the receipt records the CUDA M-forward twin declining the one-token step with "
    "'mfwd: C != 64', and the compiled decode kernel is not reachable from the model. The measured "
    "aggregate ratio is therefore a LOWER bound on us and an upper bound on the transformer",
    "receipt": "bdm/docs/deck_scaling/deck_decode_d2048_bf16_gh200_20260814.json (schema deck_decode_v1)",
    "figure": "bdm/docs/deck_scaling/context_decode_bf16_gh200_20260814.png",
}


def decode_throughput_ratio(ctx_tokens, kv_compression=1.0, d=None):
    """DERIVED-FROM-MEASURED. Per-GPU serving throughput ratio at decode:
    bAttention aggregate tokens/s over the transformer's, at a context.

    This is the decode COST ratio -- if a GPU serves X tok/s one way and Y the
    other, the cost per token goes as 1/X vs 1/Y -- and it is memory-ceiling
    driven, not latency driven. It reproduces both cells: x8.79 at 262k against
    the x8.79 measured there (an EXACT ceiling: 1 stream runs, 2 OOM), and x1.10
    at 32k, inside that cell's bounded 0.97-1.14 range (see 'ratio_32k_label' --
    the 1.14 end is the transformer measured at 8 of an 8-15 ceiling). It crosses
    1 at ~29,800 tokens: BELOW ~30k context the transformer serves more tokens
    per GPU-second than we do, and the model says so.

    kv_compression grants the transformer a mature KV stack (paged attention +
    quantization). It divides our lever by the same factor -- see
    SERVING_BLEND_20260814['sensitivity_kv8'] for what that does to the headline.
    """
    d = d or DECODE_THROUGHPUT_20260814
    tf = d["tf_anchor_tokens_per_s_per_gpu"] * (d["tf_anchor_ctx"] / float(ctx_tokens)) * float(kv_compression)
    return d["battn_tokens_per_s_per_gpu"] / tf


# ---- prefill: MEASURED, UNCHANGED by the 2026-08-14 re-base --------------------
# Prefill throughput (tokens/s per GPU) at d2048 in the bf16 SATURATED lane, each
# family at its own saturating batch. Mirrors the deck's
# PREFILL_TOKENS_PER_S_D2048 table exactly so the two models cannot drift.
# (context, transformer tok/s, bAttention tok/s)
PREFILL_TOKENS_PER_S_D2048 = {
    2048: (4682606.88285999, 1185825.0),
    8192: (3450080.0, 1183690.0),
    32768: (1711802.0, 1186661.0),
    65536: (1007187.0, 1181622.0),
    131072: (541094.0, 978107.0),
    262144: (277474.0, 726142.0),
    524288: (140820.0, 478522.0),
}


def prefill_tokens_per_s(family, ctx_tokens):
    """MEASURED. Prefill throughput at a context, log-interpolated between the
    measured points of the bf16 saturated lane."""
    idx = 0 if family == "transformer" else 1
    pts = sorted(PREFILL_TOKENS_PER_S_D2048)
    if ctx_tokens <= pts[0]:
        return PREFILL_TOKENS_PER_S_D2048[pts[0]][idx]
    if ctx_tokens >= pts[-1]:
        return PREFILL_TOKENS_PER_S_D2048[pts[-1]][idx]
    for lo, hi in zip(pts, pts[1:]):
        if lo <= ctx_tokens <= hi:
            a = PREFILL_TOKENS_PER_S_D2048[lo][idx]
            b = PREFILL_TOKENS_PER_S_D2048[hi][idx]
            f = (math.log(ctx_tokens) - math.log(lo)) / (math.log(hi) - math.log(lo))
            return a * (b / a) ** f
    return PREFILL_TOKENS_PER_S_D2048[pts[-1]][idx]


def serving_blend(ctx_tokens=65536, in_out_ratio=10.0, kv_compression=1.0):
    """The serving-workload compute blend: transformer serving cost over ours.

    Cost is accumulated per GENERATED token -- `in_out_ratio` input tokens through
    prefill plus one token through decode -- in GPU-seconds, so it is a cost ratio
    and not an average of ratios. Prefill comes from the measured bf16 saturated
    lane; decode from decode_throughput_ratio(), the memory-ceiling model fitted
    to the 2026-08-14 cells. Returns each phase's ratio and the share of
    transformer cost it accounts for, so a reader can see which phase the number
    comes from."""
    tf_pf_s = in_out_ratio / prefill_tokens_per_s("transformer", ctx_tokens)
    ba_pf_s = in_out_ratio / prefill_tokens_per_s("bdm", ctx_tokens)
    d = DECODE_THROUGHPUT_20260814
    tf_dec_tps = d["tf_anchor_tokens_per_s_per_gpu"] * (d["tf_anchor_ctx"] / float(ctx_tokens)) * kv_compression
    tf_dec_s = 1.0 / tf_dec_tps
    ba_dec_s = 1.0 / d["battn_tokens_per_s_per_gpu"]
    tf_total, ba_total = tf_pf_s + tf_dec_s, ba_pf_s + ba_dec_s
    return {
        "blend": tf_total / ba_total,
        "prefill_ratio": tf_pf_s / ba_pf_s,
        "decode_ratio": tf_dec_s / ba_dec_s,
        "tf_decode_share_of_cost": tf_dec_s / tf_total,
        "context_tokens": ctx_tokens,
        "in_out_ratio": in_out_ratio,
        "kv_compression": kv_compression,
    }


# ---- serving-workload compute blend: MEASURED ----------------------------------
# Total transformer serving cost over a request divided by total bAttention cost,
# at the deck's default operating point. A cost ratio, not an average of ratios:
#     blend = 1 / ((1 - s)/prefill_ratio + s/decode_ratio),  s = decode's share
#                                                                of transformer cost
# The PREFILL half is unchanged -- same bf16 saturated prefill lane, same 64k
# point, same x1.1732 -- because the 2026-08-14 receipt does not time prefill
# (it states so explicitly: prefill only establishes the state). The DECODE half
# is re-derived above and is what moved.
SERVING_BLEND_20260814 = {
    "status": "MEASURED (decode) x MEASURED (prefill, 2026-08-07/08 lane)",
    "blend": 2.1914745310042303,
    "prefill_ratio": 1.1731902814472388,  # UNCHANGED lane (published 1.173206; same table, tighter arithmetic)
    "decode_ratio": 2.196318388447414,  # = decode_throughput_ratio(65536); was 15.911024
    "tf_decode_share_of_cost": 0.9974654937498,  # was 0.973364
    "operating_point": "10:1 input:output, 64k average context, training excluded",
    "supersedes": {
        "blend": 11.929698, "decode_ratio": 15.911024, "tf_decode_share_of_cost": 0.973364,
        "why": "the old decode component rested on a transformer decode cost of 87.9 ms/token that a "
        "growing-KV re-planning harness artifact had inflated, and on a bAttention decode throughput "
        "measured on the h-carry-only proxy rather than the full model",
    },
    "crossover_ctx": 29839,  # the blend passes 1 here; below it the transformer is ahead at decode
    "caveat": "MEMORY-CEILING, NOT LATENCY. At one stream and 64k context the transformer decodes a token "
    "in ~4.9 ms against our ~5.7 ms -- it is FASTER per token. The lever is that it cannot hold many "
    "64k streams on a card and we can. Below ~30k context the blend falls under 1 and we are behind",
    "sensitivity_kv8": {
        "blend": 0.27879433403451537, "decode_ratio": 0.2745397985559268,
        "note": "granting the transformer an idealized mature stack (paged attention + KV quantization, "
        "KV /8) divides the decode lever by 8 and puts the TRANSFORMER ahead at 64k. The measured cells "
        "grant it no such thing, and neither family's decode is optimised in this receipt -- ours ran the "
        "unoptimised Triton per-step path at 9.5% GPU-busy. The headline uses the measured cells and "
        "carries this row as the stated downside",
    },
    "receipt": "bdm/docs/deck_scaling/deck_decode_d2048_bf16_gh200_20260814.json; prefill lane from "
    "bdm/docs/deck/build/derived.json via the deck's serving module",
}

# Kernel operating points the compute lever can be evaluated at. The value is the
# measured transformer-cost / bAttention-cost ratio at that point: above 1 we are
# ahead, below 1 we are behind.
KERNEL_OPERATING_POINTS = {
    "Serving blend, 64k context (measured 2026-08-14)": SERVING_BLEND_20260814["blend"],
    "Serving blend, 262k context (measured 2026-08-14)": 8.737893443830517,
    "Long-context prefill, 1M tokens (measured 2026-07-21)": 4.016653993568674,
    "Kernels at parity (x1)": 1.0,
    "Serving blend, 64k, granting the transformer KV /8": SERVING_BLEND_20260814["sensitivity_kv8"]["blend"],
    "Single-GPU training step, 2k context (measured 2026-08-14)": 1.0 / STEP_GAP_20260814["gap_against_battn"],
}
DEFAULT_KERNEL_POINT = "Serving blend, 64k context (measured 2026-08-14)"


def param_matching_fraction(n_tf, fit=None):
    """FIT-DERIVED. bAttention parameters needed to reach the transformer fit's
    quality at `n_tf` transformer parameters, as a FRACTION of n_tf.

    ln N_b = (a_t + b_t*ln N_t - a_b) / b_b   then   fraction = N_b / N_t
    """
    f = fit or QUALITY_FIT_20260814
    a_b, b_b = f["battn"]["a"], f["battn"]["b"]
    a_t, b_t = f["tf"]["a"], f["tf"]["b"]
    ln_nb = (a_t + b_t * math.log(n_tf) - a_b) / b_b
    return math.exp(ln_nb) / n_tf


def param_matching_gain(n_tf, fit=None):
    """FIT-DERIVED. Compute-per-token multiplier at EQUAL QUALITY = 1 / fraction.

    Per-token cost is ~linear in parameters for both the prefill FLOP term and
    the weight-bandwidth-bound decode term, so needing 23.7% of the parameters at
    trillion scale is a x4.22 compute-per-token advantage at equal quality."""
    return 1.0 / param_matching_fraction(n_tf, fit)


def serving_state_ratio(ctx_tokens, kv_compression=1.0, s=None):
    """MEASURED. Transformer serving state / bAttention serving state at a context.

    Theirs is linear in context, ours is flat, so the ratio grows ~linearly. Set
    kv_compression to the transformer stack's assumed KV saving (the deck grants
    it 8x for paged attention + quantization)."""
    s = s or SERVING_STATE_20260814
    tf_mb = s["tf_kv_mb_per_token_per_stream"] * ctx_tokens / float(kv_compression)
    return tf_mb / s["battn_state_mb_per_stream"]


def compute_lever(scale_params=DEFAULT_DEPLOYMENT_SCALE, kernel_factor=None):
    """The 2026-08-14 scenario's FLOPs lever, as an explicit product of one
    FIT-DERIVED and one MEASURED factor:

        flop_factor = param_matching_gain(scale)   [FIT-DERIVED]
                    x kernel_factor                [MEASURED, operating-point]

    They compose because they are different quantities: the kernel factor is the
    per-token cost ratio at EQUAL SIZE, the parameter gain is the size ratio at
    EQUAL QUALITY. Multiplying them assumes bAttention's per-token cost falls
    ~linearly with its parameter count, which is a projection, not a receipt."""
    if kernel_factor is None:
        kernel_factor = KERNEL_OPERATING_POINTS[DEFAULT_KERNEL_POINT]
    return param_matching_gain(scale_params) * float(kernel_factor)


# ---- scenarios the app offers (latest first; the app defaults to the first) ----
SCENARIOS = {
    "Quality-matched (2026-08-14 sealed refit)": {
        "flop_factor": None,  # computed: compute_lever(scale, kernel_factor)
        "dynamic": True,
        "blurb": "Equal fitted quality needs FEWER bAttention parameters as scale grows "
        "(FIT-DERIVED, 4 rungs per family), multiplied by the MEASURED cost ratio at the "
        "chosen kernel operating point. That measured factor is a SERVING-THROUGHPUT ratio "
        "set by the memory ceiling -- how many concurrent streams a GPU can hold -- not a "
        "per-token latency ratio; per token at one stream the transformer is faster at 64k. "
        "Memory stays at the conservative /100 lever even though the measured serving-state "
        "ratio is far larger.",
    },
    "Measured (H100 2026-07-21: compute x4)": {
        "flop_factor": 4.0,
        "dynamic": False,
        "blurb": "Long-context prefill measured 4x on one H100 at 1M tokens, parameter-matched, bf16.",
    },
    "Ceiling (kernels mature: compute x10)": {
        "flop_factor": 10.0,
        "dynamic": False,
        "blurb": "The architectural ceiling as kernels mature.",
    },
    "Memory-only (compute parity x1)": {
        "flop_factor": 1.0,
        "dynamic": False,
        "blurb": "Credits only the serving-memory advantage; assumes nothing at all of compute.",
    },
}
DEFAULT_SCENARIO = next(iter(SCENARIOS))


def reduction_factor(g):
    """Amdahl cost-weighted reduction = 1 / (mem_share/mem_factor + compute_share/flop_factor)."""
    compute_share = 1 - g["mem_share"]
    residual = g["mem_share"] / g["mem_factor"] + compute_share / g["flop_factor"]
    return 1.0 / residual


def energy_reduction(g):
    """Operating-energy (opex) reduction. Energy splits between memory (HBM I/O + data
    movement) and compute the SAME way cost does, so by default it equals the cost-weighted
    reduction. Override by setting g['opex_reduction_override'] to a number (None = derive)."""
    ov = g.get("opex_reduction_override")
    return float(ov) if ov else reduction_factor(g)


def compute_company(comp, g, year):
    """Return all derived numbers for one company in one year ('fy25' or 'fy26'). All $B."""
    total, infra, server, accel_share = comp[year]
    R = reduction_factor(g)
    ai_capex = total * infra  # AI-infra capex (full)
    server_bucket = ai_capex * server
    accel = server_bucket * accel_share  # accelerator capex
    # fleet & operating cost (from accelerator capex)
    fleet = accel * 1e9 / g["gpu_cost"]  # GPU-equiv
    power_mw = fleet * g["wall_power_kw"] / 1000
    ai_opex = (
        power_mw * 24 * 365 * 1000 * g["elec_rate"] * (1 + g["cooling_overhead"]) / 1e9
    )
    # efficient version & value (dc_scale extends the cut to non-accelerator AI capex)
    capex_avoided = (accel + (ai_capex - accel) * g["dc_scale"]) * (1 - 1 / R)
    opex_saved = ai_opex * (1 - 1 / energy_reduction(g))
    spend_cut = capex_avoided + opex_saved
    capitalized = spend_cut / g["discount_rate"]
    # AI economics (cash basis)
    rev = comp["ai_rev"][0 if year == "fy25" else 1]
    net_now = rev - ai_capex - ai_opex
    net_arch = net_now + spend_cut
    return {
        "name": comp["name"],
        "total": total,
        "ai_capex": ai_capex,
        "accel": accel,
        "accel_pct": accel / total if total else 0.0,
        "ai_opex": ai_opex,
        "ai_rev": rev,
        "net_now": net_now,
        "capex_avoided": capex_avoided,
        "opex_saved": opex_saved,
        "spend_cut": spend_cut,
        "net_arch": net_arch,
        "capitalized": capitalized,
        "pct_cut": spend_cut / (ai_capex + ai_opex) if (ai_capex + ai_opex) else 0.0,
    }


def compute_year(g, companies, year):
    """Per-company rows + a 'TOTAL' dict for one year."""
    rows = [compute_company(c, g, year) for c in companies]
    keys_sum = [
        "total",
        "ai_capex",
        "accel",
        "ai_opex",
        "ai_rev",
        "net_now",
        "capex_avoided",
        "opex_saved",
        "spend_cut",
        "net_arch",
        "capitalized",
    ]
    total = {k: sum(r[k] for r in rows) for k in keys_sum}
    total["name"] = f"TOTAL ({len(companies)})"
    spend = total["ai_capex"] + total["ai_opex"]
    total["pct_cut"] = total["spend_cut"] / spend if spend else 0.0
    total["accel_pct"] = total["accel"] / total["total"] if total["total"] else 0.0
    return rows, total


def global_estimate(total, g):
    """Gross the named-firm TOTAL up to a worldwide ESTIMATE. The named firms are
    ~named_share_of_global of global AI capex; the remainder (other clouds, China
    [Alibaba/ByteDance], neoclouds [CoreWeave], xAI, sovereign & enterprise) is added
    pro-rata. Clearly an estimate -- adjust the share to taste."""
    share = g.get("named_share_of_global", 0.80)
    f = 1.0 / share
    out = {
        k: total[k] * f
        for k in (
            "ai_capex",
            "ai_opex",
            "ai_rev",
            "net_now",
            "spend_cut",
            "net_arch",
            "capitalized",
        )
    }
    out["name"] = "GLOBAL (est.)"
    out["named_share"] = share
    return out


def _selfcheck():
    """Every published constant must be re-derivable from the receipts.

    1. param_matching_fraction() reproduces the sealed receipt table.
    2. decode_throughput_ratio() reproduces both MEASURED aggregate cells.
    3. The published SERVING_BLEND_20260814 constants are what serving_blend()
       computes -- so the mirrored constant can never drift from its own math.
    """
    worst = 0.0
    for n_tf, pct in PARAM_MATCHING_PCT_20260814.items():
        got = param_matching_fraction(n_tf) * 100.0
        worst = max(worst, abs(got - pct))
    assert worst < 1e-6, f"parameter-matching curve drifted from the receipt by {worst}"

    d = DECODE_THROUGHPUT_20260814
    # the 1/T law is anchored at 32k, so 262k is the out-of-sample check
    assert abs(decode_throughput_ratio(262144) / d["measured_ratio_262k"] - 1) < 1e-4, (
        "the 1/T decode law no longer reproduces the measured 262k aggregate cell"
    )
    assert abs(decode_throughput_ratio(32768) / d["measured_ratio_32k"] - 1) < 0.05, (
        "the 1/T decode law drifted from the measured 32k aggregate cell by >5%"
    )

    b = SERVING_BLEND_20260814
    got = serving_blend()
    for k in ("blend", "prefill_ratio", "decode_ratio", "tf_decode_share_of_cost"):
        assert abs(got[k] - b[k]) < 1e-12, f"published {k} {b[k]} != derived {got[k]}"
    assert abs(serving_blend(kv_compression=8.0)["blend"] - b["sensitivity_kv8"]["blend"]) < 1e-12
    return worst


if __name__ == "__main__":
    print(f"receipt cross-check: parameter-matching curve max drift {_selfcheck():.2e} pp\n")
    print("== 2026-08-14 scenario: the two factors of the compute lever ==")
    for n_tf, label in DEPLOYMENT_SCALES:
        frac = param_matching_fraction(n_tf)
        lever = compute_lever(n_tf)
        g = dict(GLOBALS, flop_factor=lever)
        print(
            f"  {label:>20s}: equal quality at {frac:6.1%} of the params "
            f"(x{param_matching_gain(n_tf):5.2f} FIT) x {KERNEL_OPERATING_POINTS[DEFAULT_KERNEL_POINT]:5.2f} MEASURED "
            f"= flop x{lever:6.2f}  ->  cost-weighted x{reduction_factor(g):6.2f}"
        )
    print(
        f"  counterweight: at 2,048-token context on ONE GPU a training step costs "
        f"x{STEP_GAP_20260814['gap_against_battn']:.2f} MORE, not less "
        f"({STEP_GAP_20260814['battn_ms_per_step']:,.0f} vs {STEP_GAP_20260814['tf_ms_per_step']:,.0f} ms/step)"
    )
    print("== decode lever, MEASURED: per-GPU SERVING THROUGHPUT, not per-token latency ==")
    _d = DECODE_THROUGHPUT_20260814
    print(
        f"  per-token at ONE stream, 64k: transformer ~4.92 ms vs ours ~5.74 ms GPU-busy "
        f"-> x0.86, the TRANSFORMER is ahead"
    )
    print(
        f"  per-GPU aggregate at the memory ceiling (exact at 262k: 1 stream, 2 OOM; at 32k the "
        f"receipts bracket it, 8 ran and 16 OOMed):"
    )
    for ctx in (4096, 32768, 65536, 262144, 1048576):
        tf = _d["tf_anchor_tokens_per_s_per_gpu"] * (_d["tf_anchor_ctx"] / ctx)
        print(
            f"    ctx {ctx:>9,}: transformer {tf:>9,.1f} tok/s/GPU  vs  ours "
            f"{_d['battn_tokens_per_s_per_gpu']:>7,.1f} (flat)  ->  x{decode_throughput_ratio(ctx):>7.2f}"
        )
    print(
        f"    cells: x{_d['measured_ratio_262k']:.2f} at 262k (MEASURED, ceiling exact) and "
        f"x{_d['ratio_32k_bounded'][0]:.2f}-{_d['ratio_32k_bounded'][1]:.2f} at 32k (BOUNDED, near parity: "
        f"the transformer measured at 8 of an 8-15 ceiling, bandwidth-capped at "
        f"{_d['tf_32k_aggregate_bandwidth_cap_tokens_per_s']:.0f} tok/s); "
        f"the lever crosses 1 at ~{SERVING_BLEND_20260814['crossover_ctx']:,} tokens"
    )
    print("== serving state, MEASURED, FULL MODEL (transformer KV grows per token; ours does not) ==")
    _ss = SERVING_STATE_20260814
    print(
        f"  transformer {_ss['tf_kv_mb_per_token_per_stream'] * 1000:.1f} KB of KV per token of context per stream; "
        f"ours a constant {_ss['battn_state_mb_per_stream']:.1f} MB/stream "
        f"(the retired h-carry proxy said {_ss['battn_state_mb_per_stream_h_carry_proxy']:.2f} MB and left out the M memory)"
    )
    for ctx in (4096, 65536, 262144, 1048576):
        print(
            f"  ctx {ctx:>9,}: x{serving_state_ratio(ctx):>12,.0f} raw  |  "
            f"x{serving_state_ratio(ctx, kv_compression=8.0):>10,.0f} granting the transformer KV /8  "
            f"(the model's memory lever stays capped at /{GLOBALS['mem_factor']:.0f})"
        )
    _g = dict(GLOBALS, flop_factor=compute_lever())
    _rows, _t25 = compute_year(_g, COMPANIES, "fy25")
    _rows, _t26 = compute_year(_g, COMPANIES, "fy26")
    print(
        f"== {DEFAULT_SCENARIO} at {DEFAULT_DEPLOYMENT_SCALE:,.0f} params: "
        f"cost-weighted x{reduction_factor(_g):.1f} -> FY25 spend cut ${_t25['spend_cut']:.0f}B/yr "
        f"(${_t25['capitalized'] / 1000:.1f}T capitalized), FY26 ${_t26['spend_cut']:.0f}B/yr ==\n"
        "   (the tables below use GLOBALS, which still carry the x10 ceiling lever "
        "so the generated workbook is unchanged)"
    )
    _sup = SERVING_BLEND_20260814["supersedes"]
    _og = dict(GLOBALS, flop_factor=param_matching_gain(DEFAULT_DEPLOYMENT_SCALE) * _sup["blend"])
    _, _ot25 = compute_year(_og, COMPANIES, "fy25")
    _kg = dict(GLOBALS, flop_factor=param_matching_gain(DEFAULT_DEPLOYMENT_SCALE)
               * SERVING_BLEND_20260814["sensitivity_kv8"]["blend"])
    _, _kt25 = compute_year(_kg, COMPANIES, "fy25")
    print(
        f"   re-base: the retired blend x{_sup['blend']:.2f} gave flop x{_og['flop_factor']:.2f} -> "
        f"x{reduction_factor(_og):.1f} -> ${_ot25['spend_cut']:.0f}B/yr. The compute lever fell "
        f"{_og['flop_factor'] / _g['flop_factor']:.1f}x; the DOLLARS moved "
        f"{100 * (_t25['spend_cut'] / _ot25['spend_cut'] - 1):+.1f}% -- that is the Amdahl saturation the model exists to show.\n"
        f"   downside row: granting the transformer KV /8 gives blend "
        f"x{SERVING_BLEND_20260814['sensitivity_kv8']['blend']:.2f} -> flop x{_kg['flop_factor']:.2f} -> "
        f"x{reduction_factor(_kg):.1f} -> ${_kt25['spend_cut']:.0f}B/yr"
    )
    print()
    for yr in ("fy25", "fy26"):
        rows, tot = compute_year(GLOBALS, COMPANIES, yr)
        glob = global_estimate(tot, GLOBALS)
        print(f"\n{yr.upper()}  reduction={reduction_factor(GLOBALS):.1f}x")
        for r in rows + [tot]:
            print(
                f"  {r['name']:13s} rev {r['ai_rev']:6.1f}  AIcapex {r['ai_capex']:7.1f}  "
                f"netNOW {r['net_now']:8.1f}  cut {r['spend_cut']:6.1f}  netARCH {r['net_arch']:8.1f}  "
                f"cap {r.get('capitalized', 0):7.0f}"
            )
        print(
            f"  {glob['name']:13s} rev {glob['ai_rev']:6.1f}  AIcapex {glob['ai_capex']:7.1f}  "
            f"netNOW {glob['net_now']:8.1f}  cut {glob['spend_cut']:6.1f}  netARCH {glob['net_arch']:8.1f}  "
            f"cap {glob['capitalized']:7.0f}  (named {glob['named_share']:.0%} of world)"
        )
