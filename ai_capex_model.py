"""Single source of truth for the AI-Capex-Efficiency model.

Both the Excel generator (ai_capex_efficiency.py) and the Streamlit app (app.py)
import their DEFAULTS and LOGIC from here, so the spreadsheet and the web app
never drift.

Engine: a GPU is ~60% memory / ~40% compute by cost. The cost-weighted
reduction is 1 / (mem_share/mem_factor + compute_share/flop_factor), floored
by the less-reduced component (compute). Two scenarios (2026-08-17 ruling):
TODAY = memory /100 + the quality-matched compute lever (~x9.24 at 1T) ->
~20x; CEILING = memory /100 + the prefill-kernel-campaign lever
(x4.02 measured x 5.5 speedup = x22.1, CEILING_FLOP_LEVER) -> ~41x.

RE-BASED 2026-08-14 (decode). A full-model bf16 decode measurement (d2048/24L,
both families, one GH200, same session --
bdm/docs/deck_scaling/deck_decode_d2048_bf16_gh200_20260814.json) retired the
transformer's 87.9-per-token decode cost (a growing-KV re-planning artifact of the
old harness) and the 0.15 MB bAttention serving state (an h-carry-only proxy with
no M memory). The decode advantage is NOT a per-token latency advantage -- at one
stream and 64k context the transformer is ahead of us -- it is a MEMORY CEILING:
per-GPU aggregate serving throughput, x8.79 measured at 262k (an exact ceiling: 1
stream, 2 OOM) and a bounded x0.97-1.14 at 32k (near parity -- 8 streams ran, 16
OOMed, 9-15 were never run),
crossing 1 at ~29,800 tokens. Serving blend x11.93 -> x2.19, compute lever
x50.33 -> x9.24. The "x53 decode" and "512 streams at 0.15 MB" lines are RETIRED.
This file and /home/fberl/ai-capex-efficiency-model/ai_capex_model.py must agree
on the blend and the lever -- they are cross-checked, do not let them drift.

Measured basis, updated 2026-08-08: the MEASURED block below carries the
2026-08-07/08 multi-GPU receipts (8xH100 training scaling, 512-stream decode at
262k context, memory walls, quality-parity projections), each value keyed to
/home/fberl/bdm/docs/deck/build/derived.json. serving_economics() and
training_throughput_ratio() turn them into $/1M-token and GPU-hour numbers on
this file's existing $/GPU-hour ladder.

Compute lever: the prefill anchor is 4.02, from the bf16 parameter-matched
prefill sweep (matched_d512_h100.json, 1 x H100, 2026-07-21) at T=1,048,576.
Same lane: 3.83 at d1024/1M, 2.01 at d2048/262k. It also anchors the CEILING
scenario: CEILING_FLOP_LEVER = 4.02 x 5.5 kernel-campaign speedup = 22.09.

The comparison exists in two lanes:

  * bf16 (matched_d*_h100.json) — FlashAttention available to the transformer,
    parameter counts exactly equal on both arms; 1 layer, batch 16.
  * fp32 (deck_speed_scaling_d2048_h100_20260803.json) — full model, 24 layers,
    batch 1; 7.908x at T=262,144. There is no fp32 FlashAttention kernel, so
    that lane runs mem-efficient SDPA on the transformer (the receipt's
    sdpa_fp32_backend_probe records FLASH_ATTENTION and CUDNN_ATTENTION as
    "rejected: No available kernel") and the owned arm's route attestation
    reads io=fp32,bf_col=fp16. Params differ by 16.7%.

The lever comes from the bf16 lane, where both families run their production
attention kernels; the fp32 sweep is kept below as the full-model scope
reference. Full model in bf16 has not been run, and it cannot be derived from
these two: r_fp32 x (m_bdm/m_tf) cancels algebraically back to the bf16 ratio,
and the apparent per-arm multipliers are dominated by the lanes' batch
difference rather than dtype.

The ratio is context- and width-dependent and is always quoted with both. The
43-315x serving-state math from the same 2026-07-21 session underpins
mem_factor, which is unchanged.
"""

# ---- default global assumptions ------------------------------------------------
GLOBALS = {
    "mem_factor": 100,  # memory reduction (x) — conservative vs the 2026-08-14 full-model receipt (MEASURED at 262k: 64 streams on ONE GPU in 1.62 GB vs the transformer's 1 stream at 51.5 GB, a 2nd OOMs; per-stream state 25.4 MB full model vs 197 KB of KV per token of context = x2,032)
    "flop_factor": 9.24,  # FLOPs reduction (x) — the TODAY lever: measured workload blend x2.19 (10:1 in:out, 64k context) x fit-derived equal-quality parameter ratio x4.22 at 1T = COMPUTE_LEVER_20260814 (cross-checked at import). CEILING is CEILING_FLOP_LEVER = 22.09 (measured prefill x4.02 x 5.5 kernel-campaign speedup). Measured cluster TRAINING throughput is x1.39–x1.70 (see MEASURED)
    "mem_share": 0.60,  # memory share of GPU cost (BOM)
    "opex_reduction_override": None,  # energy reduction: None = derive (= cost-weighted reduction); set a number to override
    "discount_rate": 0.06,  # perpetuity capitalization rate (~long-bond yield; was 0.10 until 2026-08-17)
    "gpu_cost": 50000,  # fully-loaded $ per GPU
    "wall_power_kw": 1.8,  # wall power per GPU (incl PUE + node overhead)
    "elec_rate": 0.08,  # $/kWh
    "cooling_overhead": 0.25,  # non-power running cost as fraction of electricity
    "fleet_life_yr": 4,  # AI-GPU depreciation life
    "spacex_mktcap": 1840,  # $B — SPCX ~$1.84T Aug 2026 (IPO 2026-06-12 at ~$1.77T)
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
        "basis": "FY25 capex incl leases ~$88B [DISCLOSED proxy]; FY26 ~$190B CY26 guide (Jul'26 restated "
        "to ~$175B capex+finance-leases on an accounting change -- the gross guide is kept here). "
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
        "mcap": 4250,
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
        "fy26": (220, 0.70, 0.65, 0.67),
        "mcap": 2830,
        "ai_rev": (12, 22),
        "basis": "FY25 cash capex $128.3B [DISCLOSED]; FY26 raised to ~$220B (Q2'26, 2026-07-30). "
        "AWS ~68-70% -> strips fulfillment. Accel ~67% [BOM]. AI rev >$15B AWS run-rate (Q1 FY26, Jassy).",
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
        "fy26": (137.5, 1.00, 0.65, 0.74),
        "mcap": 1490,
        "ai_rev": (4, 8),
        "basis": "FY25 capex incl leases $72.2B [DISCLOSED]; FY26 $130-145B guide (mid 137.5). Accel ~74% [BOM]. "
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
        "fy26": (55.7, 0.90, 0.75, 0.70),
        "mcap": 433,
        "ai_rev": (8, 15),
        "basis": "FY25 capex $21.2B [DISCLOSED 10-K]; FY26 ACTUAL $55.7B (reported 2026-06-10; FY27 guide "
        "~$70B net / up to $95B gross). Capex ~all OCI/GPU data centers. Accel ~70% [BOM]. AI rev = "
        "OCI/GPU cloud (IaaS ~$12B r/r Q4 FY25; AI subset, EST).",
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
        "mcap": 1840,
        "ai_rev": (0.3, 1),
        "basis": "FY25 S-1 AI capex $12.7B [DISCLOSED]; FY26 ~$18B (ESTIMATE, likely conservative -- Q1'26 "
        "AI capex alone was $7.7B, ~$31B annualized). ~all accelerator (greenfield). "
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


# ---- measured multi-GPU systems receipts (2026-08-07/08) -----------------------
# Source of truth: /home/fberl/bdm/docs/deck/build/derived.json (the bAttention
# deck build — 2/4/8 x H100 sessions 2026-08-07/08; per-value provenance inside).
# Naming: bAttention on reader-facing surfaces; the receipts use its internal
# name BDM. Every value below states the derived.json key it came from — do not
# edit a number here without a new receipt.
#
# SCOPE DISCIPLINE (non-negotiable, matches the deck): the systems benches are
# COMPONENT-scope — bAttention times its d1536 fwd+bwd h-recurrence sub-block,
# the transformer a d2048 FORWARD-ONLY attention+FFN layer. Every speedup is
# each family against ITS OWN 1-GPU baseline in the same harness, so the scope
# cancels inside each ratio. Never quote cross-family ABSOLUTE per-token costs
# side by side without this label.
MEASURED = {
    # --- training, multi-GPU (8 x H100, matched load = 16 sequences in flight for BOTH families)
    "train_x8_battn_matched": 5.148575074557121,  # matched_load_curve.bdm / scaling_1to8_best.matched_load_16_in_flight.bdm_x8_K16 (fwd+bwd, layer pipeline)
    "train_x8_tf_matched": 3.697557217680221,  # matched_load_curve.tf — Ulysses at its own best measured B=16 (saturated), forward-only
    "train_x8_battn_deep": 6.272772372602131,  # scaling_1to8_best.bdm — 256 sequences in flight: the pipeline keeps filling; the transformer saturates by 16
    # --- decode / serving at 262,144-token context
    # RE-BASED 2026-08-14 onto the FULL-MODEL bf16 receipt
    # (bdm/docs/deck_scaling/deck_decode_d2048_bf16_gh200_20260814.json, d2048/24L,
    # both families, one GH200, same session). Everything in this sub-block used to
    # come from the 2026-08-07/08 h-carry-only decode proxy, whose state figure left
    # the M memory out entirely and whose transformer per-token cost was inflated by
    # a growing-KV re-planning artifact in that harness. Per-GPU cells are scaled by
    # gpus_per_box for the box rows -- a per-GPU measurement grossed up, not an
    # 8-GPU measurement. See DECODE_THROUGHPUT_20260814 for the full derivation.
    "serve_streams_per_gpu": 64,  # measured concurrent 262,144-token streams on ONE GPU, 1.62 GB of state
    "serve_streams_per_box": 512,  # 64 per GPU x 8 GPUs (unchanged headcount; the per-stream state behind it was 0.15 MB, is now 25.4 MB)
    "serve_tokens_per_s_per_gpu": 562.0835756046827,  # measured aggregate at 64 streams / 262,144 ctx
    "serve_tokens_per_s_box": 562.0835756046827 * 8,  # = 4,496.7 -- per-GPU cell x gpus_per_box
    "serve_state_mb_per_stream": 25.362432,  # FULL MODEL (h + M), 24.2 MiB measured, context-flat
    "serve_state_mb_per_stream_h_carry_proxy": 0.15234375,  # RETIRED as a headline: h-carry only, no M memory
    "tf_kv_gb_per_stream_262k": 51.54,  # 51,539,607,552 B for one 262,144-token stream = 196,608 B (197 KB) per token
    "tf_oom_streams_262k": 2,  # a SECOND 262k stream OOMs the 96 GB card; 1 fits at 51.5 GB
    "tf_oom_streams_32k": 16,  # 8 streams RAN at 32,768 tokens and 16 OOMed; 9-15 untested, so the
    # 32k ceiling is bracketed (8 <= c < 16, analytic ~11-15) -- unlike 262k, where 1 fits and 2 OOM
    "tf_ms_per_token_262k_1stream": 15.630828042048961,  # measured wall clock (was 88.5 -- that number is RETIRED)
    "decode_gap_262k_1stream": 14919.937999999996 / 5759.9,  # = 2.59, GPU-busy per token, 1 stream each side
    "decode_gap_262k_1stream_scope": "GPU-BUSY per token. On WALL CLOCK the same single-stream cell runs "
    "AGAINST us (15.6 ms vs 109.1 ms): our decode step is 95% host-launch-bound at one stream (2,731 kernel "
    "launches) and only 5% device work. Concurrency amortises that -- which is why the serving claim is the "
    "aggregate-throughput one, never the single-stream one. The retired x53 gap is gone",
    "decode_gap_262k_aggregate": 562.0835756046827 / 63.976137240450086,  # = 8.79, per-GPU aggregate at each side's ceiling
    # --- memory walls (component scope, see above; GiB labelled GB, the deck convention)
    "train_64k_battn_gb": 30.93900489807129,  # pipeline_feasibility.bdm_64k_gb — a 64k-token training sequence fits ONE GPU
    "train_64k_tf_attempted_gb": 78.35036659240723,  # pipeline_feasibility.tf_64k_attempted_gb — OOM (tf_64k_status = "oom") on the same 80 GB GPU
    "pipe_stage_battn_gb": 9.048385620117188,  # pipeline_memory_asymmetry.bdm — per-GPU pipeline peak, FLAT over 16..256 in flight
    "pipe_stage_tf_gpipe_gb": 43.253414154052734,  # pipeline_memory_asymmetry.tf — GPipe stage peak, flat over 16..64 (component-scope, width-unmatched; caveat carried in that key)
    # --- quality-side equivalences — PROJECTIONS from the measured ladder fits (label as such wherever quoted)
    "parity_70B_param_multiple": 4.10407288161176,  # quality_fit.ref_70B.param_multiple, 95% CI [2.20, 7.66] — PROJECTION
    "parity_70B_token_multiple": 1.675445365112522,  # quality_fit.ref_70B.token_multiple_beta028, CI [1.33, 2.10]; Chinchilla beta = 0.28 is an ASSUMPTION — PROJECTION
    "quality_crossover_params": 320787561.0,  # quality_fit.crossover.N_star — lands essentially AT the top measured rung (314.3M)
    # --- trainability
    "stepped_vs_scanned": 23.558062554766632,  # stepped_vs_scanned.ratio — a scanned training step vs stepping the recurrence token-by-token (same geometry, same box)
    #
    # --- cross-family prefill, bf16 FAIR-DTYPE lane — this sets flop_factor
    # Source: experiments/paper_figures/output/matched_d{512,1024,2048}_h100.json
    # benchmark_kind full_sequence_forward_prefill_not_decode; bfloat16; 1 x H100 80GB HBM3;
    # 2026-07-21; 1 layer (block scope), batch 16; parameter_matched=True with param counts
    # EXACTLY equal on both arms (e.g. 3,185,184 vs 3,185,184 at d512).
    # This lane sets the lever: bf16 gives both families their production attention kernels
    # (there is no fp32 FlashAttention kernel). Cost of the choice: 1 layer rather than 24.
    "prefill_bf16_tf_battn_d512_1m": 4.016653993568674,  # TF 7,339.483 ms vs bAttention 1,827.263 ms at d512, T=1,048,576 — THE measured flop_factor
    "prefill_bf16_tf_battn_d1024_1m": 3.834441155040915,  # d1024, T=1,048,576
    "prefill_bf16_tf_battn_d2048_262k": 2.0141390010169635,  # d2048, T=262,144 — the width/context-matched counterpart to the fp32 lane's 7.908x
    #
    # --- the same comparison in the fp32 lane — FULL MODEL, but NOT fair on dtype
    # Source: experiments/paper_figures/output/deck_speed_scaling_d2048_h100_20260803.json
    # schema deck_speed_scaling_v2; precision_lane fp32_both_families_no_autocast; 1 x H100
    # (GPU UUID, GPU snapshot, script sha256 in the receipt); d_model 2048 / 24 layers /
    # batch 1; every cell status "ok"; transformer 1,040,781,312 params vs bAttention
    # 867,108,864 (a 16.7% gap, unlike the bf16 lane's exact match).
    # flop_factor is sourced from the bf16 lane above, not from these. This receipt's
    # sdpa_fp32_backend_probe reads FLASH_ATTENTION "rejected: No available kernel" (and
    # CUDNN_ATTENTION likewise), and the owned arm's last_route_attestation reads dtype_mix
    # "io=fp32,bf_col=fp16". At the same width and context the two lanes read 7.908 vs 2.014.
    # Kept as the full-model scope reference and for the crossover shape.
    "prefill_fp32_d2048_tf_battn_262k": 7.90781287273641,  # TF 149,366.343 ms vs bAttention 18,888.452 ms at T=262,144
    "prefill_fp32_d2048_tf_battn_131k": 3.9960243388830623,  # T=131,072
    "prefill_fp32_d2048_tf_battn_65k": 2.0470555605257617,  # T=65,536
    "prefill_fp32_d2048_crossover_tokens": 31000,  # approximate T where the fp32-lane ratio crosses 1.0 — measured 0.5833 at 16,384 and 1.0674 at 32,768
    #
    # --- 16-bit IO adoption gate — SAME-ARCH own-baseline, GH200; NOT a cross-family ratio
    # Source: experiments/paper_figures/output/receipts/io16_gate_20260809.md section 4
    # (copied from /home/fberl/bdm/docs/io16_gate_20260809.md, which is untracked there).
    # 1 x GH200 480GB; bAttention d1536 / 27 layers / block 2048 / batch 32 / seed 1; GPU
    # verified idle before each arm; steady state only, warmup discarded. ONLY the io dtype
    # varies between arms. THROUGHPUT is d1536/L27; the QUALITY deltas come from the gate's
    # smaller vehicle (width 512, 15 layers) — different geometry, keep them labelled.
    # 0 skips on all arms; verdict PASS_io16_SHIPS.
    # This is a training step-time result on a different device. It is not an input to
    # flop_factor, which is measured in the bf16 lane where both arms already have their
    # 16-bit kernels.
    "io16_fp32_s_per_step": 10.460,
    "io16_fp16_s_per_step": 8.828,
    "io16_bf16_s_per_step": 8.931,
    "io16_fp16_speedup": 1.185,
    "io16_bf16_speedup": 1.171,
    "io16_fp32_peak_mb": 72366,
    "io16_fp16_peak_mb": 59667,
    "io16_bf16_peak_mb": 57452,
    "io16_fp16_mem_ratio": 0.825,  # -17.5% peak allocated (72.4 -> 59.7 GB)
    "io16_bf16_mem_ratio": 0.794,  # -20.6% peak allocated (72.4 -> 57.5 GB)
}

# ---- the CEILING scenario (2026-08-17 ruling) ----------------------------------
# The FLOPs lever if the prefill kernel campaign lands its ~5-6x speedup over our
# CURRENT prefill, anchored on the measured bf16 parameter-matched prefill lever
# (x4.02 at d512/1M). Replaces the old flat x10 "kernels mature" ceiling.
CEILING_PREFILL_SPEEDUP = 5.5  # ASSUMPTION: mid of the 5-6x kernel-campaign target
CEILING_FLOP_LEVER = MEASURED["prefill_bf16_tf_battn_d512_1m"] * CEILING_PREFILL_SPEEDUP  # = 22.09

# ---- serving & training economics on the measured receipts ---------------------
# Rates come from this file's own cost ladder (CostLadder tab / costladder_tab):
# "Rent NVIDIA — neocloud / committed" $2.00–3.50/hr -> mid 2.50. Utilization
# matches the ladder's owned-TCO cross-check. These are the SAME rate assumptions
# the model always carried — not new levers.
SERVING = {
    "gpu_hr": 2.50,  # $/H100-hr — mid of the ladder's neocloud/committed rent row (2.00–3.50); swap for own-silicon 0.90–1.40 or on-demand 3.00–7.00
    "utilization": 0.85,  # same value as the ladder's owned-TCO cross-check
    "gpus_per_box": 8,  # 8 x H100 80 GB, the measured box
    "hbm_gb": 80.0,  # H100 80 GB
    "hbm_usable_gb": 72.0,  # ASSUMPTION: 8 GB reserved for weights + workspace when counting KV streams
    "hbm_tbps": 3.35,  # H100 SXM HBM bandwidth (same profile as experiments/architecture_costs.py)
    "tf_kv_compression": 1.0,  # 2026-08-14: the headline now runs on the MEASURED cells, where the transformer is granted nothing and neither family's decode is optimised (ours ran the unoptimised Triton per-step path at 9.5% GPU-busy). Set 8.0 to grant it the idealized mature stack -- paged attention + KV quantization -- which divides our decode lever by 8 and puts the TRANSFORMER ahead at 64k. That row is reported, not hidden
}

KV_MB_PER_TOKEN_PER_STREAM = 39936.0 / 262144  # = 0.15234375 MB — measured linear at 4k/32k/262k (derived.json decode rows)

# The two prefill lanes at d2048, so every surface can show WHY the lever comes
# from bf16 and not from the fp32 sweep. (seq_len, transformer_ms, battn_ms),
# rounded to 3 decimals for legibility; every ratio they produce matches the
# full-precision receipt at display precision. Ratio is transformer / bAttention:
# below 1 the transformer is ahead.
#
# bf16, FlashAttention available to the transformer, params exactly matched,
# 1 layer / batch 16 (matched_d2048_h100.json). This is the fair-dtype lane.
PREFILL_BF16_D2048_SWEEP = (
    (2048, 111.229, 663.113),
    (4096, 125.091, 662.751),
    (8192, 153.085, 663.327),
    (16384, 204.919, 663.202),
    (32768, 309.886, 662.618),
    (65536, 520.986, 663.199),
    (131072, 969.790, 753.457),
    (262144, 1889.616, 938.175),
    (524288, 3718.717, 1308.837),
)

# fp32, FlashAttention unavailable to the transformer (no fp32 kernel) and fp16
# internals on the owned arm, params 16.7% apart, 24 layers / batch 1
# (deck_speed_scaling_d2048_h100_20260803.json). Full-model scope, unfair dtype.
PREFILL_FP32_D2048_SWEEP = (
    (512, 6.578, 61.216),
    (1024, 12.611, 86.660),
    (2048, 27.856, 160.738),
    (4096, 70.339, 305.622),
    (8192, 207.758, 595.671),
    (16384, 686.813, 1177.361),
    (32768, 2518.562, 2359.638),
    (65536, 9629.459, 4704.054),
    (131072, 37690.932, 9432.108),
    (262144, 149366.343, 18888.452),
)

# Same session and lane as the fp32 sweep, the smaller pair (d_model 1152, 24
# layers): transformer 337,713,408 params vs bAttention 314,340,480 (-6.9%).
# Carries the same fp32/no-FlashAttention caveat as its d2048 sibling.
PREFILL_FP32_D1152_TF_BATTN_262K = 83580.588 / 15575.729  # = 5.366


# ---- workload-mix compute lever ------------------------------------------------
# A single flop_factor cannot represent training, prefill and decode at once: the
# three point in OPPOSITE directions at today's kernel maturity.
#
#   training  transformer is FASTER (5.3x at d384, 9.6x at d1152, T=2048 blocks).
#             FLOPs are near-matched (1.07-1.19x); the gap is achieved FLOP/s
#             (200.9 TF/s vs 21.7). Crosses in our favour near T ~ 88,000 at
#             d1152, where the transformer's causal-attention term finally
#             dominates its core. Source: bdm/docs/bdm_tf_speed_gap_20260729.md
#   prefill   we win only past the crossover (~65k in the bf16 saturated lane).
#   decode    RE-BASED 2026-08-14: we win only past ~30k context, and only on
#             per-GPU AGGREGATE throughput (how many streams the memory ceiling
#             allows), never on per-token cost -- at 64k and one stream the
#             transformer is ahead of us. x8.79 measured at 262k (exact ceiling);
#             x0.97-1.14 bounded at 32k (near parity, ceiling only bracketed).
#             The old "1.9x at 4k, 7.5x at 32k, 53x at 262k" line is RETIRED.
#
# The blend below is a COST ratio, not an average of ratios: total transformer
# time over a request divided by total owned time. Weighting is set by the
# workload, not by which phase flatters the measurement.
#
# Aggregation note: our decode cost is context-FLAT while the transformer's is
# linear in context, so the decode advantage is linear in E[context] and the
# ratio of averages equals the average of ratios. Mixing context lengths
# therefore carries no Jensen penalty, and E[T] is an honest single knob.
WORKLOAD = {
    "in_out_ratio": 10.0,  # input tokens per output token. ~10:1 = the deck's wedge (code, reasoning, agent traces). RAG/doc-QA is 50-100:1; chat ~10:1; reasoning can invert.
    "context_tokens": 65536,  # E[context] over the workload. 32k-128k is the agentic/code band.
    "train_share": 0.0,  # fraction of accelerator cost spent TRAINING. Default 0: this is a SERVING-cost claim (deck page 5), and training is a loss at short sequence today. Set >0 to see it.
}

# Decode, 1 stream per side, GPU-BUSY per generated token from a CUDA kernel trace,
# bf16, full model, one GH200 (deck_decode_d2048_bf16_gh200_20260814.json).
# RE-BASED 2026-08-14: this replaces an fp16 8xH100 series whose transformer row
# read 87.88 at 262k. That figure was inflated by a growing-KV re-planning artifact
# in the old harness; the trace-based series below is 5.9x smaller at the same
# point, and the "x53 decode" line built on the old series is RETIRED.
# Transformer cost is linear in context (it re-reads its whole KV per token); ours
# is flat. OLS over the four measured transformer points has a worst residual of
# 0.43% and is exact at 262k.
DECODE_MEASURED = {
    2048: (1.691854, 5.691737),  # (transformer, owned) GPU-busy ms per generated token
    8192: (1.992146, 5.754300),
    32768: (3.256546, 5.747400),
    262144: (14.919938, 5.759900),
}
DECODE_TF_MS_INTERCEPT = 1.58399  # ms/token at ctx -> 0 (OLS over the four points)
DECODE_TF_MS_PER_TOKEN_CTX = 5.08737e-5  # ms/token added per token of context
DECODE_OWN_MS = 5.738334  # mean of the four measured points; flat in context (spread 1.2%)
# The reading a reader must not miss: at the deck's 64k operating point this puts
# the TRANSFORMER ahead per token -- 4.92 ms against our 5.74 ms. The decode
# advantage is a memory ceiling (how many streams fit), not a per-token win.

# Prefill throughput (tokens/s) at d2048 in the bf16 SATURATED lane, each arm at
# its own saturating batch (sat_d2048_h100.json). Ours is flat until the batch
# gets too small to fill the GPU; the transformer's falls as attention goes
# quadratic. Used by log-interpolation in prefill_tokens_per_s().
PREFILL_TOKENS_PER_S_D2048 = {
    2048: (4682606.88285999, 1185825.0),
    8192: (3450080.0, 1183690.0),
    32768: (1711802.0, 1186661.0),
    65536: (1007187.0, 1181622.0),
    131072: (541094.0, 978107.0),
    262144: (277474.0, 726142.0),
    524288: (140820.0, 478522.0),
}

# Training wall-clock advantage (owned / transformer). Below 1 means we are
# slower. Measured at T=2048 blocks; the crossover is near T ~ 88,000.
TRAIN_ADVANTAGE_SHORT_SEQ = 1.0 / 9.6  # d1152, TF 9.6x faster (bdm_tf_speed_gap_20260729.md)
TRAIN_CROSSOVER_TOKENS = 88000  # where owned training cost meets the transformer's at d1152


def decode_ms_per_token(family, ctx):
    """Measured decode cost per generated token at a given context."""
    if family == "transformer":
        return DECODE_TF_MS_INTERCEPT + DECODE_TF_MS_PER_TOKEN_CTX * ctx
    return DECODE_OWN_MS


def prefill_tokens_per_s(family, ctx):
    """Prefill throughput at a context, log-interpolated between measured points."""
    import math

    idx = 0 if family == "transformer" else 1
    pts = sorted(PREFILL_TOKENS_PER_S_D2048)
    if ctx <= pts[0]:
        return PREFILL_TOKENS_PER_S_D2048[pts[0]][idx]
    if ctx >= pts[-1]:
        return PREFILL_TOKENS_PER_S_D2048[pts[-1]][idx]
    for lo, hi in zip(pts, pts[1:]):
        if lo <= ctx <= hi:
            a = PREFILL_TOKENS_PER_S_D2048[lo][idx]
            b = PREFILL_TOKENS_PER_S_D2048[hi][idx]
            f = (math.log(ctx) - math.log(lo)) / (math.log(hi) - math.log(lo))
            return a * (b / a) ** f
    return PREFILL_TOKENS_PER_S_D2048[pts[-1]][idx]


# ---- decode: MEASURED per-GPU SERVING THROUGHPUT (the 2026-08-14 re-base) ------
# The serving-relevant decode quantity is not latency, it is how many concurrent
# streams a GPU can hold times how fast it serves them. The transformer re-reads
# its whole KV cache for every token it emits, so once the card is full of KV its
# aggregate rate is bandwidth-floored at BW / (kv_bytes_per_token * T) -- INVERSELY
# LINEAR IN CONTEXT. Ours is context-flat because the state is context-flat.
#
# Two measured cells pin the law:
#
#     ctx  32,768 :  8 streams -> 511.84 tok/s/GPU   (16 streams OOM)
#     ctx 262,144 :  1 stream  ->  63.98 tok/s/GPU   ( 2 streams OOM)
#
# At 262k the transformer is at its CEILING exactly -- 1 runs, 2 OOM, the integers
# are adjacent. At 32k it is not: 8 ran and 16 OOMed, 9..15 were never run, so the
# ceiling there is only bracketed (8 <= c < 16). See "ratio_32k_bounded" below --
# the 32k cell is a RANGE, near parity, and nothing here leans on it.
#
# 511.84 * 32768/262144 = 63.980 against the measured 63.976 -- the law anchored on
# the near cell reproduces the far one to 0.006%, so the 64k row is arithmetic.
DECODE_THROUGHPUT_20260814 = {
    "status": "MEASURED",
    "config": "d2048 / 24 layers / bf16, 1 x GH200 96 GB; autoregressive decode, prefill untimed",
    "tf_anchor_ctx": 32768,
    "tf_anchor_tokens_per_s_per_gpu": 511.84161509663613,
    "tf_far_cell": {"ctx": 262144, "streams": 1, "tokens_per_s": 63.976137240450086, "next_streams_status": "oom"},
    "battn_tokens_per_s_per_gpu": 562.0835756046827,  # 64 streams @ 262,144 ctx -- the LOWER of the two cells
    "battn_cell_32k": 585.4143612476083,  # 64 streams @ 32,768 ctx
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
    "crossover_ctx": 29839,  # the decode lever passes 1 here; below it the transformer is ahead
    "conservatism": "BOTH cells run against us. The transformer sits at 95.6% GPU-busy at its ceiling -- no "
    "headroom left -- while our 64-stream cell is 9.5% GPU-busy on 1.62 GB of a 96 GB card, a grid cap and "
    "not a ceiling, running the unoptimised Triton per-step decode path (the receipt records the CUDA twin "
    "declining with 'mfwd: C != 64'). The measured ratio is a LOWER bound on us",
    "receipt": "bdm/docs/deck_scaling/deck_decode_d2048_bf16_gh200_20260814.json (schema deck_decode_v1)",
    "figure": "bdm/docs/deck_scaling/context_decode_bf16_gh200_20260814.png",
}


def decode_tokens_per_s_per_gpu(family, ctx_tokens, kv_compression=1.0):
    """DERIVED-FROM-MEASURED. Per-GPU aggregate decode throughput at a context,
    each family at its own measured concurrency ceiling."""
    d = DECODE_THROUGHPUT_20260814
    if family == "transformer":
        return d["tf_anchor_tokens_per_s_per_gpu"] * (d["tf_anchor_ctx"] / float(ctx_tokens)) * float(kv_compression)
    return d["battn_tokens_per_s_per_gpu"]


def decode_throughput_ratio(ctx_tokens, kv_compression=1.0):
    """DERIVED-FROM-MEASURED. Our aggregate decode throughput over the
    transformer's -- which is also the decode COST ratio, since cost per token
    goes as 1/throughput. Crosses 1 at ~29,800 tokens."""
    return decode_tokens_per_s_per_gpu("bdm", ctx_tokens) / decode_tokens_per_s_per_gpu(
        "transformer", ctx_tokens, kv_compression
    )


def workload_compute_advantage(w=None):
    """Blended compute advantage (transformer cost / owned cost) for a workload.

    Cost is accumulated per GENERATED token: `in_out_ratio` input tokens through
    prefill plus one token through decode. Returns the blend, each phase's own
    ratio, and the share of transformer cost each phase accounts for -- so a
    reader can see WHICH phase the number is coming from."""
    w = dict(WORKLOAD, **(w or {}))
    ctx = float(w["context_tokens"])
    r = float(w["in_out_ratio"])
    gpus = SERVING["gpus_per_box"]

    # Decode comes from serving_economics(), which models the production regime:
    # streams batched per GPU up to the KV memory wall, transformer granted the
    # IDEALIZED mature stack (paged attention + KV quantization, bandwidth-floor
    # serving). Using single-stream decode latency instead would overstate the
    # transformer's decode cost by orders of magnitude, because prefill is
    # measured at a saturating batch and unbatched decode is its worst case.
    econ = serving_economics(ctx)
    tf_dec_tps = econ["tf_tokens_per_s_box"]
    own_dec_tps = econ["battn_tokens_per_s_box"]

    # Prefill throughput scales with the box (data-parallel over requests).
    tf_pf_tps = prefill_tokens_per_s("transformer", ctx) * gpus
    own_pf_tps = prefill_tokens_per_s("bdm", ctx) * gpus

    # Seconds of box time per GENERATED token: r input tokens through prefill,
    # one token through decode.
    tf_pf_s = r / tf_pf_tps if tf_pf_tps else float("inf")
    own_pf_s = r / own_pf_tps if own_pf_tps else float("inf")
    tf_dec_s = 1.0 / tf_dec_tps if tf_dec_tps else float("inf")
    own_dec_s = 1.0 / own_dec_tps if own_dec_tps else float("inf")

    tf_total = tf_pf_s + tf_dec_s
    own_total = own_pf_s + own_dec_s
    serving = tf_total / own_total

    train_share = float(w["train_share"])
    if train_share > 0:
        # Harmonic (cost-weighted) blend across training and serving.
        blended = 1.0 / (
            train_share / TRAIN_ADVANTAGE_SHORT_SEQ + (1.0 - train_share) / serving
        )
    else:
        blended = serving

    return {
        "blended": blended,
        "serving": serving,
        "prefill_ratio": tf_pf_s / own_pf_s if own_pf_s else float("inf"),
        "decode_ratio": tf_dec_s / own_dec_s if own_dec_s else float("inf"),
        "tf_prefill_share": tf_pf_s / tf_total if tf_total else 0.0,
        "tf_decode_share": tf_dec_s / tf_total if tf_total else 0.0,
        "context_tokens": ctx,
        "in_out_ratio": r,
        "train_share": train_share,
    }


# ---- 2026-08-14 sealed quality refit — the QUALITY-side compute lever ----------
# Receipt authority: /home/fberl/bdm/docs/deck/build/quality_fit_v4.json (written
# by docs/deck/build/build_slides_v4.py); narrative receipt is the "## v4" section
# of /home/fberl/bdm/docs/deck/README.md.
#
# Everything above this block compares the two families at MATCHED SIZE. This
# block adds the missing axis: how big each family has to be to reach the SAME
# QUALITY. Both families now carry four sealed ladder rungs (47M-663M params)
# fitted as ln(bpb) = a + b*ln(params). The fits cross at 392M; above the
# crossing bAttention reaches the transformer fit's quality on fewer parameters.
#
# STATUS: FIT-DERIVED, and the receipt says so in its own words — "A projection
# of the two fits, not a measurement." The rungs span 47M-663M, so the
# trillion-parameter point quoted on the slide is three orders of magnitude
# beyond the top measured rung. Label it as a projection wherever it is shown.
QUALITY_FIT_20260814 = {
    "status": "FIT-DERIVED",
    "form": "ln(bpb) = a + b*ln(params), OLS per family",
    "battn": {"a": 2.413386876447594, "b": -0.12534052380844748, "rungs": 4},
    "tf": {"a": 1.9582862071595213, "b": -0.1023400790322208, "rungs": 4},
    "battn_rung_params": (56056320, 141586176, 314340480, 579841536),
    "tf_rung_params": (47803392, 136641024, 337713408, 662676480),
    "crossover_params": 391933596.2,
    "crossover_ci95": (259177028.7, 592691198.8),
    # the top rung is a MEASURED pair, not a fit
    "top_pair": {
        "battn_params": 579841536, "battn_bpb": 0.894186824913843,
        "tf_params": 662676480, "tf_bpb": 0.889691062993173,
        "param_pct": -12.5, "bpb_pct": +0.505,
    },
    "receipt": "bdm/docs/deck/build/quality_fit_v4.json",
}

# The curve exactly as the receipt publishes it — bAttention params for equal
# fitted quality, as a percentage of the transformer's. Cross-checked against
# param_matching_fraction() at import.
PARAM_MATCHING_PCT_20260814 = {
    391933596: 100.0,
    1_000_000_000: 84.20793235593338,
    10_000_000_000: 55.18859584465115,
    100_000_000_000: 36.16976484388941,
    1_000_000_000_000: 23.705112783532595,
    10_000_000_000_000: 15.535969738960972,
}

# The scale the deck's economics slide is quoted at.
DECK_DEPLOYMENT_SCALE = 1_000_000_000_000

# ---- the honest counterweight: MEASURED, and it runs AGAINST us ---------------
# Source: /home/fberl/bdm/docs/deck/build/dtype_trio_v4.json (601-step io16
# protocol, 1 x GH200, d1536 / 27 layers, batch 32 x block 2048 = 65,536
# tokens/step; the transformer runs 16 x grad-accum 2 for the same tokens/step
# because it cannot fit batch 32x1 at d1536 in any dtype).
STEP_GAP_20260814 = {
    "status": "MEASURED",
    "battn_ms_per_step": 8979.3,   # fp16
    "tf_ms_per_step": 1389.0,      # fp16
    "gap_against_battn": 8979.3 / 1389.0,  # = 6.46 — we are SLOWER
    "context_tokens": 2048,
    "note": "a short-context training configuration, not a serving workload; the kernel campaign is the line item against it",
    "receipt": "bdm/docs/deck/build/dtype_trio_v4.json",
}


def param_matching_fraction(n_tf, fit=None):
    """FIT-DERIVED. bAttention parameters needed to match the transformer fit's
    quality at n_tf transformer parameters, as a FRACTION of n_tf.

        ln N_b = (a_t + b_t*ln N_t - a_b) / b_b   then   fraction = N_b / N_t
    """
    import math

    f = fit or QUALITY_FIT_20260814
    a_b, b_b = f["battn"]["a"], f["battn"]["b"]
    a_t, b_t = f["tf"]["a"], f["tf"]["b"]
    return math.exp((a_t + b_t * math.log(n_tf) - a_b) / b_b) / n_tf


def param_matching_gain(n_tf, fit=None):
    """FIT-DERIVED. Compute-per-token multiplier at EQUAL QUALITY = 1 / fraction."""
    return 1.0 / param_matching_fraction(n_tf, fit)


def quality_matched_compute_lever(n_tf=DECK_DEPLOYMENT_SCALE, w=None):
    """The compute lever with BOTH axes in it, as an explicit product:

        flop_factor = workload_compute_advantage()['blended']   [MEASURED]
                    x param_matching_gain(n_tf)                 [FIT-DERIVED]

    They compose because they are different quantities: the blend is the
    per-token serving cost ratio at EQUAL SIZE, the parameter gain is the size
    ratio at EQUAL QUALITY. Multiplying them assumes bAttention's per-token cost
    falls ~linearly with its parameter count — true to first order for the
    prefill FLOP term and the weight-bandwidth-bound decode term, but a
    PROJECTION, not a receipt.

    The counterweight is STEP_GAP_20260814: at 2,048-token context on one GPU a
    training step still costs x6.46 MORE, which the equal-quality parameter
    ratio reduces to x1.53 but does not erase. The economics slide excludes
    training for exactly that reason."""
    return workload_compute_advantage(w)["blended"] * param_matching_gain(n_tf)


def _check_param_matching_curve():
    """param_matching_fraction() must reproduce the sealed receipt table."""
    worst = max(
        abs(param_matching_fraction(n) * 100.0 - pct)
        for n, pct in PARAM_MATCHING_PCT_20260814.items()
    )
    assert worst < 1e-6, f"parameter-matching curve drifted from the receipt by {worst}"
    return worst


def kv_gb_per_stream(ctx_tokens):
    """Transformer KV cache per stream at a context length, GB (measured-linear)."""
    return KV_MB_PER_TOKEN_PER_STREAM * ctx_tokens / 1024.0


def training_throughput_ratio(deep=False):
    """Measured cluster-throughput multiple at 8 GPUs: same box, more steps/s.

    Matched load (16 sequences in flight for BOTH families): x5.15 / x3.70 =
    x1.39. deep=True uses bAttention's 256-in-flight best (the pipeline keeps
    filling; the transformer is saturated by 16): x6.27 / x3.70 = x1.70.
    Component scope; each family vs its own 1-GPU baseline (see MEASURED)."""
    b = MEASURED["train_x8_battn_deep" if deep else "train_x8_battn_matched"]
    return b / MEASURED["train_x8_tf_matched"]


def serving_economics(ctx_tokens, s=None, tf_mode="mature"):
    """Cost per 1M generated tokens on one 8xH100 box at a context length.

    RE-BASED 2026-08-14 onto the full-model decode receipt.

    bAttention: the MEASURED operating point — 64 concurrent 262,144-token
    streams on ONE GPU in 1.62 GB of state, 562 tok/s/GPU aggregate, measured
    context-flat 32k..262k (rows beyond 262k extrapolate that flatness and say
    so). That 64-stream cell is a grid cap and not a ceiling — 9.5% GPU-busy —
    so it is a LOWER bound on us.

    Transformer, tf_mode='mature' (default): the MEASURED 1/context law at its
    own KV ceiling. It re-reads its whole KV cache per emitted token, so once
    the card is full its aggregate rate is 511.84 tok/s/GPU at 32,768 tokens
    (8 streams ran; 16 OOMed, so that ceiling is bracketed at 8-15 and the 32k
    cell is a RANGE, x0.97-1.14 -- see 'ratio_32k_label') falling as 1/context to
    63.98 at 262,144 (1 stream; 2 OOM, an exact ceiling). s['tf_kv_compression'] multiplies that rate: at the default 1.0 the
    transformer is granted nothing, at 8.0 it gets an idealized paged-attention
    + KV-quantization stack and goes AHEAD of us at 64k.
    tf_mode='bandwidth_floor' is the pre-re-base analytic model;
    tf_mode='as_measured' is single-stream wall clock times whatever fits.

    Short context is honest: the ratio crosses 1 at ~29,800 tokens, and BELOW
    that the transformer serves more tokens per GPU-second than we do. So does
    per-token latency at any context we measured — this is a memory-ceiling
    result, not a per-token one."""
    s = dict(SERVING, **(s or {}))
    box_hr = s["gpu_hr"] * s["gpus_per_box"]
    kv_raw = kv_gb_per_stream(ctx_tokens)
    kv_eff = kv_raw / s["tf_kv_compression"]
    # RE-BASED 2026-08-14: the bAttention rate is the FULL-MODEL measured cell
    # (64 streams on one GPU at 262k), context-flat, grossed up by the box.
    battn_toks = decode_tokens_per_s_per_gpu("bdm", ctx_tokens) * s["gpus_per_box"]
    if tf_mode == "mature":
        # measured 1/T law at the transformer's own KV ceiling
        tf_streams_gpu = int(s["hbm_usable_gb"] // kv_eff) if kv_eff > 0 else 0
        tf_toks_gpu = (
            decode_tokens_per_s_per_gpu("transformer", ctx_tokens, s["tf_kv_compression"])
            if tf_streams_gpu >= 1
            else 0.0
        )
    elif tf_mode == "bandwidth_floor":  # the pre-re-base analytic model, kept for comparison
        tf_streams_gpu = int(s["hbm_usable_gb"] // kv_eff) if kv_eff > 0 else 0
        tf_toks_gpu = (s["hbm_tbps"] * 1e3) / kv_eff if tf_streams_gpu >= 1 else 0.0
    else:  # as_measured -- single-stream wall clock x whatever fits
        tf_streams_gpu = int(s["hbm_usable_gb"] // kv_raw)
        tf_toks_gpu = (
            tf_streams_gpu * 1e3 / MEASURED["tf_ms_per_token_262k_1stream"]
            if tf_streams_gpu >= 1
            else 0.0
        )
    tf_toks = tf_toks_gpu * s["gpus_per_box"]
    cost = lambda tps: (box_hr / (tps * 3600.0)) * 1e6 if tps > 0 else float("inf")
    return {
        "ctx": ctx_tokens,
        "battn_tokens_per_s_box": battn_toks,
        "battn_usd_per_mtok": cost(battn_toks),
        "battn_extrapolated": ctx_tokens > 262144,
        "tf_streams_per_gpu": tf_streams_gpu,
        "tf_tokens_per_s_box": tf_toks,
        "tf_usd_per_mtok": cost(tf_toks),
        "tf_kv_gb_per_stream_eff": kv_eff if tf_mode == "mature" else kv_raw,
        "cost_ratio": cost(tf_toks) / cost(battn_toks) if battn_toks else float("inf"),
        "tf_fits": tf_streams_gpu >= 1,
        "tf_mode": tf_mode,
    }


def serving_cost_curve(ctxs=(4096, 16384, 32768, 65536, 131072, 262144, 1048576), s=None):
    """serving_economics over a context sweep — the KV wall moves linearly with
    context, so the cost ratio grows ~linearly past the ~4k crossover."""
    return [serving_economics(c, s=s) for c in ctxs]


# The economics slide's compute lever, resolved once at import so the chart, the
# slide copy and the app can never quote different numbers. Defined here (not up
# with the constants) because it needs serving_economics().
_check_param_matching_curve()
COMPUTE_LEVER_20260814 = quality_matched_compute_lever()
# GLOBALS carries the TODAY lever as its default; it must stay pinned to the
# computed quality-matched value so no surface quotes a stale number.
assert abs(GLOBALS["flop_factor"] - COMPUTE_LEVER_20260814) < 0.05, (
    f"GLOBALS['flop_factor']={GLOBALS['flop_factor']} drifted from the computed "
    f"quality-matched lever {COMPUTE_LEVER_20260814:.4f}"
)


# One row per assumption, visible on every surface that renders the serving /
# training numbers (app tab, workbook sheet, __main__ print). status is one of
# MEASURED / PROJECTION / ASSUMPTION.
SERVING_TRAINING_ASSUMPTIONS = [
    ("Compute lever (economics slide)", f"x{COMPUTE_LEVER_20260814:.2f} = the x{workload_compute_advantage()['blended']:.2f} MEASURED serving blend x the x{param_matching_gain(DECK_DEPLOYMENT_SCALE):.2f} FIT-DERIVED equal-quality parameter ratio at trillion scale", "MEASURED x PROJECTION", "workload_compute_advantage() x quality_matched_compute_lever(); receipts derived.json + quality_fit_v4.json"),
    ("Equal-quality parameter matching", "sealed 2026-08-14 refit, 4 rungs per family (47M-663M params): the fits cross at 392M and bAttention then matches the transformer fit's quality on 84.2% of the params at 1B, 55.2% at 10B, 36.2% at 100B, 23.7% at 1T, 15.5% at 10T. The receipt's own words: 'a projection of the two fits, not a measurement'", "PROJECTION", "quality_fit_v4.json param_matching; fits over the sealed rope-convention ladder"),
    ("Single-GPU training step (AGAINST us)", "at 2,048-token context on one GPU a bAttention step costs x6.46 MORE (8,979 vs 1,389 ms/step, d1536/27L, fp16, 601-step protocol). At equal quality the same step costs x1.53 more, not x6.46 - the parameter ratio shrinks the gap but does not erase it. Training is excluded from the serving claim for this reason", "MEASURED", "dtype_trio_v4.json rows[fp16]"),
    ("$/GPU-hour", "$2.50 (H100 rent, neocloud/committed mid; ladder 2.00-3.50)", "ASSUMPTION", "this file's CostLadder row — unchanged rate basis"),
    ("Utilization", "85%", "ASSUMPTION", "same value as the ladder's owned-TCO cross-check"),
    ("Context-length mix", "headline quoted at 262k (a measured decode cell); table sweeps 4k-1M. The decode lever crosses 1 at ~29,800 tokens and BELOW that the transformer serves more tokens per GPU-second than we do — stated on every surface", "ASSUMPTION", "serving_cost_curve, decode_throughput_ratio"),
    ("Decode lever is MEMORY-CEILING, not latency (READ FIRST)", "per generated token at ONE stream the transformer is AHEAD of us at 64k context — 4.92 vs 5.74 GPU-busy per token. The lever is that it cannot hold many long-context streams: it re-reads its whole KV cache per emitted token, so its aggregate throughput falls as 1/context while ours is flat. Measured: 1 stream at 262k (2 OOM — an exact ceiling) and 8 streams at 32k (16 OOM, so that ceiling is only bracketed at 8-15 and the 32k cell is a bounded x0.97-1.14, near parity) against 64 of ours in 1.62 GB", "MEASURED", "deck_decode_d2048_bf16_gh200_20260814.json; DECODE_THROUGHPUT_20260814"),
    ("Transformer-stack maturity", "NOT GRANTED by default since the 2026-08-14 re-base: the headline runs on the measured cells, in which neither family's decode is optimised (ours ran the unoptimised Triton per-step path at 9.5% GPU-busy while the transformer sat at 95.6%). Setting tf_kv_compression=8.0 grants it paged attention + KV quantization, divides our decode lever by 8 and puts the TRANSFORMER ahead at 64k — reported, not hidden", "ASSUMPTION", "SERVING['tf_kv_compression']"),
    ("bAttention serving throughput", "562 tok/s/GPU (4,497/box): 64 concurrent 262,144-token streams on one GPU, context-flat 32k-262k (-4.0%). The 64-stream cell is a grid cap, not a ceiling — 1.62 GB of state on a 96 GB card at 9.5% GPU-busy — so it is a LOWER bound. Supersedes a 349,880 tok/s/box figure measured on the h-carry-only proxy", "MEASURED", "deck_decode_d2048_bf16_gh200_20260814.json rows[bdm, 64 streams]"),
    ("bAttention per-stream state", "25.4 MB/stream, the FULL recurrent cell (h + M memory), 24.2 MiB measured, context-flat. The 0.15 MB figure previously quoted was an h-carry-only proxy with no M memory and is retired as a headline", "MEASURED", "deck_decode_d2048_bf16_gh200_20260814.json state_bytes_per_stream"),
    ("Transformer KV wall", "197 KB per token of context per stream — 51.5 GB for one 262,144-token stream, a 2nd stream OOMs a 96 GB card, and 16 streams OOM it at 32,768 tokens. Its retired 88.5 per-token decode cost was inflated by a growing-KV re-planning harness artifact; the traced figure is 14.92 GPU-busy at 262k", "MEASURED", "deck_decode_d2048_bf16_gh200_20260814.json rows[transformer]"),
    ("Training throughput at 8 GPUs", "x5.15 (bAttention, fwd+bwd) vs x3.70 (transformer Ulysses best, forward-only) at matched 16-in-flight load -> x1.39; x6.27 at 256 in flight -> x1.70", "MEASURED", "derived.json matched_load_curve, scaling_1to8_best"),
    ("Memory walls (component scope)", "64k-token sequence: 30.9 GB (fits 1 GPU) vs OOM at 78.4 GB attempted; pipeline stage 9.05 GB flat vs GPipe 43.3 GB", "MEASURED", "derived.json pipeline_feasibility, pipeline_memory_asymmetry"),
    ("Compute lever (flop_factor)", "cross-family prefill, bf16, 1xH100, parameter counts exactly matched, block scope (1 layer, batch 16): TF/bAttention 4.02 at d512/1M, 3.83 at d1024/1M, 2.01 at d2048/262k. Short context favours the transformer and is stated so", "MEASURED", "matched_d{512,1024,2048}_h100.json (2026-07-21)"),
    ("Compute, fp32 lane (scope reference)", "the d2048/24-layer fp32 sweep gives 7.91 at 262k at full-model scope. No fp32 FlashAttention kernel exists, so that lane runs mem-efficient attention on the transformer and the owned arm is fp16 internally; params 16.7% apart. Full model in bf16 has not been run", "LABEL", "deck_speed_scaling_d2048_h100_20260803.json sdpa_fp32_backend_probe + last_route_attestation"),
    ("16-bit IO (training)", "same-arch GH200 gate at d1536: fp16 x1.185 step time and -17.5% peak memory, bf16 x1.171 and -20.6%; quality deltas +0.001 to +0.004 from the gate's smaller vehicle. Reported on its own; not an input to the compute lever", "MEASURED", "receipts/io16_gate_20260809.md (2026-08-09)"),
    ("Quality parity at 70B", "transformer needs x4.1 params [2.2-7.7] or x1.7 tokens [1.33-2.10, beta=0.28 assumption]; crossover N* ~321M = the top measured rung", "PROJECTION", "derived.json quality_fit.ref_70B, parity_cost"),
    ("Scope", "systems numbers are component-scope (bAttention d1536 fwd+bwd h-recurrence sub-block vs transformer d2048 forward-only layer); every speedup is vs the family's OWN 1-GPU baseline", "LABEL", "derived.json scope_reconciliation; deck README"),
]


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


if __name__ == "__main__":
    print("== Serving at long context — $/1M generated tokens, one 8xH100 box ==")
    print("   (component scope; decode from the 2026-08-14 MEASURED per-GPU cells, transformer granted")
    print("    no KV compression by default — see SERVING_TRAINING_ASSUMPTIONS)")
    for r in serving_cost_curve():
        tag = " [battn rate extrapolated past 262k]" if r["battn_extrapolated"] else ""
        if not r["tf_fits"]:
            print(
                f"  ctx {r['ctx']:>9,}: bAttention ${r['battn_usd_per_mtok']:.4f}/Mtok "
                f"vs transformer: ONE stream does not fit a GPU without KV compression or sharding{tag}"
            )
            continue
        print(
            f"  ctx {r['ctx']:>9,}: bAttention ${r['battn_usd_per_mtok']:.4f}/Mtok "
            f"vs transformer ${r['tf_usd_per_mtok']:.2f}/Mtok "
            f"({r['tf_streams_per_gpu']} streams/GPU)  ->  x{r['cost_ratio']:.1f}{tag}"
        )
    print("== decode lever, MEASURED: per-GPU SERVING THROUGHPUT, not per-token latency ==")
    print("  per token at ONE stream, 64k: transformer 4.92 vs ours 5.74 GPU-busy -> x0.86, the TRANSFORMER is ahead")
    for c in (4096, 32768, 65536, 262144, 1048576):
        print(
            f"    ctx {c:>9,}: transformer {decode_tokens_per_s_per_gpu('transformer', c):>9,.1f} tok/s/GPU  vs  "
            f"ours {decode_tokens_per_s_per_gpu('bdm', c):>7,.1f} (flat)  ->  x{decode_throughput_ratio(c):>7.2f}"
        )
    _d = DECODE_THROUGHPUT_20260814
    print(
        f"    cells: x{_d['measured_ratio_262k']:.2f} at 262k (1 stream vs 64, MEASURED, ceiling exact) and "
        f"x{_d['ratio_32k_bounded'][0]:.2f}-{_d['ratio_32k_bounded'][1]:.2f} at 32k (BOUNDED, near parity: 8 of an "
        f"8-15 ceiling, bandwidth-capped at {_d['tf_32k_aggregate_bandwidth_cap_tokens_per_s']:.0f} tok/s); "
        f"crosses 1 at ~{_d['crossover_ctx']:,} tokens"
    )
    print(
        f"    granting the transformer KV /8: decode lever at 64k becomes "
        f"x{decode_throughput_ratio(65536, kv_compression=8.0):.2f} — the transformer ahead. Reported, not hidden"
    )
    print(
        f"== Training at 8 GPUs: x{training_throughput_ratio():.2f} cluster throughput at matched load "
        f"(x{MEASURED['train_x8_battn_matched']:.2f} vs x{MEASURED['train_x8_tf_matched']:.2f}), "
        f"x{training_throughput_ratio(deep=True):.2f} at 256 in flight; "
        f"PLUS quality-parity PROJECTION at 70B: x{MEASURED['parity_70B_param_multiple']:.1f} params "
        f"or x{MEASURED['parity_70B_token_multiple']:.2f} tokens (beta=0.28 assumption) =="
    )
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
