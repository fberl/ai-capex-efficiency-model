"""Single source of truth for the AI-Capex-Efficiency model.

Both the Excel generator (ai_capex_efficiency.py) and the Streamlit app (app.py)
import their DEFAULTS and LOGIC from here, so the spreadsheet and the web app
never drift.

Engine: a GPU is ~60% memory / ~40% compute by cost. The cost-weighted
reduction is 1 / (mem_share/mem_factor + compute_share/flop_factor), floored
by the less-reduced component (compute). Two scenarios (2026-08-17 ruling):
TODAY = memory /100 + the quality-matched compute lever (~x9.24 at 1T) ->
~20x; CEILING = memory /100 + the prefill-kernel-campaign lever
(x4.02 measured x 7.03 campaign speedup = x28.2, CEILING_FLOP_LEVER) -> ~50x
(RE-BASED 2026-08-29/31 onto the review-verdicts position of record, 2k clean-wall
101.737 vs 59.268 ms; before that 2026-08-25 -> ~45x, 2026-08-24 -> ~39x).

RE-BASED 2026-08-24 (kernels). The CEILING's campaign speedup used to be a flat
x5.5 ASSUMPTION ("mid of the 5-6x target"). It is now MEASURED-realized x
TARGET-remaining: the 2026-08-24 GH200 megakernel measurement banked x2.856 on
our own arm in one day (400.4 -> 140.216 ms/step, single layer, fwd+bwd, bf16)
and the remaining funded gap to transformer parity is x1.786 at 8k context /
x2.226 at 2k, so the full-campaign speedup brackets x5.10-x6.36 and the
conservative (8k) end is taken. Same measurement moved the training step-time
gap from x6.116 to x2.226 at T=2048 and x1.786 at T=8192, and training peak
memory from x2.829 to x1.579. See KERNEL_CAMPAIGN_20260824 -- it also retires
the "transformer is 5.3-9.6x faster at T=2048, crossover near T~88,000" line.

RE-BASED AGAIN 2026-08-25 (3), on TWO fronts.
(a) The 2k cell itself moved: a newer receipt (tests/_w1_r0_gh200.jsonl rows
21-24, cited in ~/bdm-prefill/docs/exec_projection_20260825.md) reads BDM
115.9910 ms / TF 59.966 ms at B16/T2048 -- battn_ms_2k and tf_ms_2k are
RE-BASED to this (was 140.216 / 62.984). step_ratio_2k moved 2.226 -> 1.934,
TRAIN_ADVANTAGE_SHORT_SEQ 0.449 -> 0.517, realized_speedup 2.856 -> 3.452.
[SUPERSEDED 2026-08-29/31: the review-verdicts position of record moves the 2k
cell again, to clean-wall 101.737 +/- 0.609 vs TF 59.268 +/- 0.381 (n=8) ->
step_ratio_2k 1.717, TRAIN_ADVANTAGE_SHORT_SEQ 0.583, realized x3.936,
CEILING_PREFILL_SPEEDUP 7.03, CEILING_FLOP_LEVER 28.24, ceiling ~50x; the
dollar ceiling barely moves ($163.7B) because it is memory-bound-dominated at
these magnitudes, not flop-bound.] The 8k cell (battn_ms_8k,
step_ratio_8k, target_8k_gap_remaining) is UNCHANGED -- still the 2026-08-24
number, no newer receipt for it yet.
(b) ctx_growth_per_4x_battn (the +5.8%-per-4x slope derived from the SAME
2026-08-24 2k/8k pair) is corrected to FLAT (1.0): per
~/bdm-prefill/docs/ledger_frame_20260825.md, that slope is a B4-BATCH
OCCUPANCY ARTIFACT (the BH-parallel grid underfills the SMs at B4, and the
sequential chunk chain is 4x deeper with 4x less batch to hide it), not a
real per-token FLOP cost -- per-token FLOPs are exactly flat (no attention
term); at production batch the flat model applies. The B4/T32,768 "~parity"
cell shares the same artifact and is retired as the crossover anchor;
TRAIN_CROSSOVER_TOKENS moves from the MEASURED 32,768 to a PROJECTED ~17k
(now ~13k on the 2026-08-29 anchor, and CORROBORATED by a measured d4096
crossover: 21-24k @ B4 / ~16k @ B16 -- see TRAIN_CROSSOVER_TOKENS)
(flat model, new 2k anchor). The proper bucket-attribution fix (isolating the
artifact from the B4/T8192 cell, which would also correct step_ratio_8k) is
OWED upstream and has not landed -- both of these are the flat/re-based model
applied ahead of that receipt, not a re-measurement of the 8k cell.

The "$159B/yr FY25" headline family was re-checked against that re-base on
2026-08-24 and did NOT move: it is a pure SERVING claim (WORKLOAD train_share
defaults to 0) and both serving levers are untouched. Only the CEILING arm
moved, $163.04B -> $162.80B, which still reads "~$163B". See headline_family()
at the bottom of this file -- it now guards every quoted literal at import.

TRAINING RE-BASED AGAIN 2026-08-24 (2), on CONTEXT MIX. The training term used
to be r at ONE context (2,048 tokens) applied to all of training -- the
pessimistic corner, and never a stated assumption. Two things replace it:
r(T) as a per-token cost model (transformer base + attn*T, ours ~flat, fitted to
the 2k/8k cells and VALIDATED on the 32k parity cell it never saw), and
training_advantage_mix(), which integrates COST over a curriculum of contexts
rather than averaging ratios. Cost-weighted, the training term flips from a DRAG
of x0.449 to a CONTRIBUTOR of x1.33 (modern mix) to x6.40 (frontier + long RL
rollouts), because our per-token cost is flat in context and the transformer's
grows -- 10% of tokens at 262k is 45% of its training bill. The headline itself
is a higher bar and moves much less: training only RAISES it once it beats the
SERVING advantage x2.19, so FY25 reads $158B at a modern mix and $160B at a
long-context one against $159B serving-only. Curriculum shares are ILLUSTRATIVE
knobs; this repo carries no citation for any lab's recipe and none is invented.

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
scenario: CEILING_FLOP_LEVER = 4.02 x 7.03 campaign speedup = 28.24 (RE-BASED
2026-08-29/31; was 24.77 on the 08-25 cell, 20.49 on the 08-24 one). Note that
the 4.02 was measured on the PRE-CAMPAIGN kernels and has not been re-run since
2026-08-24, so the TODAY lever is conservative by construction; the realized
x3.936 lives in the CEILING factor, not in the TODAY one.

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
    "flop_factor": 9.24,  # FLOPs reduction (x) — the TODAY lever: measured workload blend x2.19 (10:1 in:out, 64k context) x fit-derived equal-quality parameter ratio x4.22 at 1T = COMPUTE_LEVER_20260814 (cross-checked at import). CEILING is CEILING_FLOP_LEVER = 28.24 (measured prefill x4.02 x 7.03 campaign speedup: x3.936 MEASURED-realized, 2k cell re-based 2026-08-29, x x1.786 TARGET-remaining to 8k parity). Measured cluster TRAINING throughput is x1.39–x1.70 (see MEASURED)
    "mem_share": 0.60,  # memory share of GPU cost (BOM)
    "opex_reduction_override": None,  # energy reduction: None = derive (= cost-weighted reduction); set a number to override
    "discount_rate": 0.06,  # perpetuity capitalization rate (~long-bond yield; was 0.10 until 2026-08-17)
    "gpu_cost": 50000,  # fully-loaded $ per GPU
    "wall_power_kw": 1.8,  # wall power per GPU (incl PUE + node overhead)
    "elec_rate": 0.08,  # $/kWh
    "cooling_overhead": 0.25,  # non-power running cost as fraction of electricity
    "fleet_life_yr": 4,  # AI-GPU depreciation life
    "spacex_mktcap": 1950,  # $B — SPCX ~$1.95T 2026-08-17 (IPO 2026-06-12 at ~$1.77T)
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
        "fy27": (298, 1.00, 0.67, 0.75),  # STREET EST: FY26 guide x1.57 (Morgan Stanley +57% 2027 path)
        "mcap": 3700,
        "ai_rev": (30, 50, 75),  # fy27 = fy26 x1.5 EST
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
        "fy27": (250, 1.00, 0.60, 0.67),  # STREET EST: analyst (Morgan Stanley) up-to-$250B 2027; company flagged a further increase
        "mcap": 4250,
        "ai_rev": (25, 40, 60),  # fy27 = fy26 x1.5 EST
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
        "fy27": (345, 0.70, 0.65, 0.67),  # STREET EST: FY26 guide x1.57 (Morgan Stanley +57% 2027 path)
        "mcap": 2830,
        "ai_rev": (12, 22, 33),  # fy27 = fy26 x1.5 EST
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
        "fy27": (216, 1.00, 0.65, 0.74),  # STREET EST: FY26 guide mid x1.57 (Morgan Stanley +57% 2027 path)
        "mcap": 1490,
        "ai_rev": (4, 8, 12),  # fy27 = fy26 x1.5 EST
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
        "fy27": (70, 0.90, 0.75, 0.70),  # DISCLOSED GUIDE: ~$70B net (up to $95B gross) -- the only real FY27 guide in the set
        "mcap": 433,
        "ai_rev": (8, 15, 22),  # fy27 = fy26 x1.5 EST
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
        "fy26": (55, 1.00, 0.79, 1.00),
        "fy27": (83, 1.00, 0.79, 1.00),  # EST: no guidance; x1.5 placeholder (xAI/Google compute deals ramp from 2027)
        "mcap": 1950,
        "ai_rev": (0.3, 1, 1.5),  # fy27 EST
        "basis": "FY25 S-1 AI capex $12.7B [DISCLOSED]; FY26 ~$55B EST from H1'26 DISCLOSED $23.5B "
        "(Q1 $7.7B, Q2 $15.8B; 82.7% of total capex) + company guiding capex roughly flat Q3/Q4. "
        "~all accelerator (greenfield COLOSSUS I/II). AI rev ~0 today; Anthropic (up to $1.25B/mo) "
        "and Google (up to $920M/mo) compute deals ramp from 2027.",
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

# ---- 2026-08-24 megakernel campaign — the TRAINING-SIDE step-time and memory ---
# receipts. MEASURED on one GH200 (sm_90a), quiet card, route-attested, bf16,
# gradient checkpointing OFF on BOTH arms. Scope: ONE layer of bAttention at
# d3072 against ONE layer of a MODERN transformer reference (24 query heads /
# 4 KV heads, head_dim 256, RoPE 64, gated attention) -- i.e. the transformer is
# given its current best block, not a 2023 one. fwd+bwd together, which is what
# training and prefill-with-backward actually pay.
#
# This block supersedes, as a HEADLINE, both older training-gap lines:
#   * STEP_GAP_20260814's x6.46 (d1536/27L, fp16, 601-step protocol) and
#   * the "transformer is 5.3-9.6x faster at T=2048 blocks, crossover near
#     T~88,000" line from bdm/docs/bdm_tf_speed_gap_20260729.md.
# Both were measured before the megakernel campaign. They are kept where they are
# as dated receipts; nothing on a slide should quote them as current.
#
# The distinction that must survive every restatement: everything under
# "measured" below is a receipt; everything under "target" is a funded PROGRAM
# GOAL with no receipt behind it yet.
KERNEL_CAMPAIGN_20260824 = {
    "status": "MEASURED",
    "config": "1 layer, bAttention d3072 vs modern TF ref (24Q/4KV, head_dim 256, RoPE 64, "
              "gated attention); fwd+bwd; bf16; checkpointing OFF both arms; 1 x GH200 sm_90a; "
              "quiet card; route-attested",
    # --- step time, ms/step (BDM / TF), matched tokens per step (32,768)
    # RE-BASED 2026-08-25 (2k cell only): superseded by a newer receipt --
    # tests/_w1_r0_gh200.jsonl rows 21-24 (ledger of record cited in
    # ~/bdm-prefill/docs/exec_projection_20260825.md), BDM 115.9910 ms / TF
    # 59.966 ms at B16/T2048. The 8k cell (below) has NOT been re-based --
    # it is still the 2026-08-24 number, separately flagged as needing its
    # own bucket-attribution correction (see ctx_growth_per_4x_battn comment).
    # RE-BASED AGAIN 2026-08-29 (position of record, review-verdicts corpus):
    # clean-wall BDM 101.737 +/- 0.609 ms (2SE, n=8) vs TF 59.268 +/- 0.381,
    # GH200, T2048/B16, ckpt-off, S=4 (reanchor_k2fzero_review_pass1.md; docs/
    # reanchor_20260829.md:18-32, re-verified in ctxscale receipt audit A1).
    # Supersedes the 2026-08-25 115.9910/59.966 ledger cell AND the interim
    # 103.540/1.743 row.
    "battn_ms_2k": 101.737,          # B16 / T2048 (2026-08-29 position of record)
    "tf_ms_2k": 59.268,
    "step_ratio_2k": 101.737 / 59.268,   # = 1.717 — we are still SLOWER
    "battn_ms_8k": 148.407,          # B4 / T8192 -- 2026-08-24, NOT re-based; bucket attribution owed.
    # NOTE 2026-08-31: the 8k-win gate bar in the box1meas frame is banked at
    # ~73.6 ms (memory_parity_analysis.md), below this file's 83.076 TF cell --
    # the frames differ (TF clean-wall moved 62.98 -> 59.27 at 2k); reconcile
    # when the 8k cell is re-measured.
    "tf_ms_8k": 83.076,
    "step_ratio_8k": 148.407 / 83.076,   # = 1.786 -- same caveat as battn_ms_8k above
    # --- what the campaign moved, same day, same frame
    "battn_ms_2k_precampaign": 400.4,
    "step_ratio_2k_precampaign": 6.116,
    "realized_speedup": 400.4 / 101.737,  # = 3.936 — our arm vs our arm, at the 2026-08-29 2k position of record
    # --- fwd / bwd split of the 2k cell
    "fwd_ratio": 1.82,
    "bwd_ratio": 2.35,
    # --- training PEAK memory, MiB (BDM / TF)
    "battn_peak_mib_2k": 17087.5,
    "tf_peak_mib_2k": 10823.6,
    "mem_ratio_2k": 17087.5 / 10823.6,   # = 1.579
    "mem_ratio_8k": 1.567,
    "mem_ratio_2k_precampaign": 2.829,
    # NEWER BANKED POSITIONS 2026-08-29/30 (d3072 single layer, B16/T2048, bf16,
    # ckpt-off, GH200; review-verdicts corpus): allocated-peak ratio 1.552
    # (14,111.5 vs 9,095.6 MiB; reproduces at d4096/box-3 to 1.554), and the
    # deadtape lever BANKED -1,736.0 MiB (14,111.5 -> 12,375.5, bit-equal,
    # thrice-corroborated) -> ratio 1.361 at +1.5-2.0% step time. Memory is FLAT
    # in context at matched tokens on BOTH sides (17,087.5 -> 16,970.0 / 9,095.6
    # -> 9,098.6 across B16/T2048 -> B4/T8192). RISK: a pending fairness ruling
    # may move the TF bar 9,095.6 -> 7,943.6 (lean-bwd) or 7,403.6 (both-sides
    # fold), taking the ratio to ~1.776 -- do not quote memory parity as done.
    "mem_ratio_2k_20260830_banked": 12375.5 / 9095.6,  # = 1.361 (deadtape lever landed, S=4 slab)
    # --- context scaling at MATCHED tokens per step, 2k -> 8k (4x context)
    # Our per-token cost is FLAT in context (no attention term); the transformer's
    # grows. RE-BASED 2026-08-25: the previously-used +5.8% per 4x on our side
    # (rows 2->8, 140.216->148.407) is a B4-BATCH OCCUPANCY ARTIFACT, not a real
    # per-token cost -- at B4 the BH-parallel grid underfills the SMs (124 < 132)
    # and the sequential chunk chain is 4x deeper with 4x less batch to hide it.
    # Per-token FLOPs are exactly flat; at production batch the flat model applies.
    # Correction source: ~/bdm-prefill/docs/ledger_frame_20260825.md. The B4
    # artifact is quantified: 90.5% an h-grid occupancy effect (B*n_blocks =
    # 124 < 132 SMs at B4 -- a B effect, not a T cost; memory_parity_analysis).
    # MEASURED 2026-08-29/31 (tsweep, box-3 H100 d4096, n=25): BDM BWD per-token
    # is FLAT in T (3.27-3.31 us/tok, spread 1.4%, slope -2.4e-6) across 8x
    # context while TF bwd rises 1.80 -> 2.99 us/tok (+66%); BDM fwd fits
    # a + k/T (~1.34 us/tok + 6.4 ms/step fixed at B=4 -- amortization, not a
    # negative context cost). Decode is likewise flat: 108.22 -> 109.12 ms/tok
    # over 2k -> 262k (+0.83% over 128x). The flat model is now receipt-backed.
    "ctx_growth_per_4x_battn": 1.0,
    "ctx_growth_per_4x_tf": 1.319,
    "ratio_decay_per_4x": 1.0 / 1.319,  # = 0.758
    "equal_token_parity_ctx": 32768,
    "equal_token_parity_note": "an EQUAL-TOKEN cell at B4/T32768 read ~parity, but that cell shares "
                               "the same B4 occupancy artifact as the growth-rate pair above -- it is "
                               "not independent ground truth. At the flat per-token model the fitted "
                               "crossover moves to ~25,700 (TRAIN_CROSSOVER_TOKENS), i.e. we should "
                               "already be ahead of parity below 32k at production batch. Holding the "
                               "old ratio_decay_per_4x=0.802 constant instead gives x1.43 at 32k and a "
                               "crossover near T~310,000 -- that constant-decay extrapolation remains "
                               "the CONSERVATIVE bound, because the transformer's attention term is "
                               "quadratic so its per-4x growth rises with context while ours stays flat",
    # --- the funded program: TARGETS, not receipts. Label them as such wherever shown.
    "target_status": "TARGET (funded program, no receipt yet)",
    "target_8k_win_gate_ms": 83.076,     # T8192 step time that would beat the transformer
    "target_8k_gap_remaining": 148.407 / 83.076,  # = 1.786, i.e. 44% of the step still to remove
    "target_levers": (
        "recompute removal ~-12.8 ms; backward-fold + occupancy redesign (~3x headroom "
        "in the dominant fused kernel); copies elimination (24.8% of the step)"
    ),
    "target_mem_ratio": 0.88,            # program projection: at or below transformer peak memory
    "receipt": "2026-08-24 GH200 megakernel measurement (bdm campaign, single-layer frame)",
}

# ---- the CEILING scenario (2026-08-17 ruling, re-based 2026-08-24) -------------
# The FLOPs lever if the kernel campaign lands, anchored on the measured bf16
# parameter-matched prefill lever (x4.02 at d512/1M). Replaces the old flat x10
# "kernels mature" ceiling.
#
# RE-BASED 2026-08-24. This used to be a flat x5.5 ASSUMPTION ("mid of the 5-6x
# kernel-campaign target"). It is now an explicit product of a MEASURED factor and
# a TARGET factor, so the two halves can never be quoted as one measurement:
#
#   MEASURED  x3.936 already realized on our own arm (400.4 -> 101.737 ms/step
#             at B16/T2048, RE-BASED 2026-08-29 onto the review-verdicts position
#             of record; was x3.452 -> 115.991 on 08-25, x2.856 -> 140.216 on 08-24)
#   TARGET    x1.786 remaining to transformer parity at 8k context -- the funded
#             8k-win gate (T8192 step <= 83.076 ms). At 2k the remaining gap is
#             now x1.717 (2026-08-29 position of record), so the full-campaign speedup brackets shift; the
#             CONSERVATIVE (8k) end is taken here.
KERNEL_SPEEDUP_REALIZED_20260824 = KERNEL_CAMPAIGN_20260824["realized_speedup"]  # = 2.856 MEASURED
KERNEL_SPEEDUP_REMAINING_TARGET = KERNEL_CAMPAIGN_20260824["target_8k_gap_remaining"]  # = 1.786 TARGET
CEILING_PREFILL_SPEEDUP = (
    KERNEL_SPEEDUP_REALIZED_20260824 * KERNEL_SPEEDUP_REMAINING_TARGET
)  # = 5.10 (MEASURED x TARGET); the 2k lane's counterpart is 6.36
CEILING_FLOP_LEVER = MEASURED["prefill_bf16_tf_battn_d512_1m"] * CEILING_PREFILL_SPEEDUP  # = 24.77 (RE-BASED AGAIN 2026-08-25 (3); was 20.49)

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
#   training  transformer is still FASTER, but by much less than it was.
#             RE-BASED 2026-08-24 (KERNEL_CAMPAIGN_20260824, 1 x GH200, single
#             layer, fwd+bwd, bf16, checkpointing off both arms, against a
#             MODERN transformer block): x2.226 at T=2,048 and x1.786 at
#             T=8,192, from x6.116 at T=2,048 the same morning. RE-BASED AGAIN
#             2026-08-25/29: the 2k cell is now x1.717 (2026-08-29 position of record),
#             our per-token cost is FLAT in context (the x1.058-per-4x growth and
#             the B4/T32,768 ~parity cell are RETIRED as a B4 occupancy
#             artifact), and the crossover is a PROJECTED T ~ 16,900
#             (TRAIN_CROSSOVER_TOKENS), from T ~ 88,000 pre-campaign.
#             RETIRED: "5.3x at d384, 9.6x at d1152, T=2048 blocks; achieved
#             FLOP/s 200.9 vs 21.7; crossover near T ~ 88,000"
#             (bdm/docs/bdm_tf_speed_gap_20260729.md) — pre-campaign kernels.
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
    "train_share": 0.0,  # fraction of accelerator cost spent TRAINING. Default 0: this is a SERVING-cost claim (deck page 5). It is NOT a claim that training is a loss -- that depends entirely on the context mix (see train_curriculum). Set >0 to see it.
    "train_curriculum": "legacy_short",  # the context mix training runs at, a key of TRAINING_CURRICULA or a [(ctx, token_share), ...] list. Default legacy_short (100% at 2,048) ONLY because it reproduces the model's historical TRAIN_ADVANTAGE_SHORT_SEQ constant exactly; it is the pessimistic case and NOT what a modern run looks like. modern_standard / long_context / frontier_rl flip the training term from a drag (x0.517) to a contributor (x1.91-x10.31) (RE-BASED AGAIN 2026-08-25 (3); was x0.449 / x1.33-x6.40).
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
# slower. RE-BASED 2026-08-24 onto KERNEL_CAMPAIGN_20260824 (1 x GH200, single
# layer, fwd+bwd, bf16, checkpointing off both arms, modern transformer block).
# It read 1/9.6 with a crossover near T ~ 88,000 until then, from the
# pre-megakernel d1152 sweep (bdm/docs/bdm_tf_speed_gap_20260729.md).
TRAIN_ADVANTAGE_SHORT_SEQ = 1.0 / KERNEL_CAMPAIGN_20260824["step_ratio_2k"]  # = 0.583 at T=2,048 (2026-08-29 position of record)
# RE-BASED 2026-08-25: the equal-token B4/T32,768 cell reading ~parity was the
# same B4-batch occupancy artifact as ctx_growth_per_4x_battn above (see that
# comment) -- not independent ground truth, so it is retired as the crossover
# anchor. At the flat per-token model (our real behavior at production batch),
# fed by the newer 2026-08-25 2k receipt (battn_ms_2k=115.9910), the fitted
# crossover is T~13,008 (2026-08-29 anchor; was ~16,943 on the 08-25 cell): we
# should already be meaningfully ahead of parity
# well before 32k. This is PROJECTED, not measured -- a corrected production-
# batch receipt is owed (~/bdm-prefill/docs/ledger_frame_20260825.md) and has
# not landed. A naive log-linear extrapolation of the 2k->8k ratio decay
# (x0.802 per 4x, the pre-2026-08-25 constant) would put the crossover near
# T ~ 310,000 instead: that remains the CONSERVATIVE bound, because the
# transformer's attention term is quadratic so its growth factor rises with
# context while ours stays flat.
# CORROBORATED 2026-08-31: an independent measured crossover exists on box-3
# H100 d4096 (tsweep, n=25): step parity ~21-24k tokens at B=4, bwd parity
# ~19.7-19.8k, and the fixed-B16 convention reads T* ~ 16k (memory_parity
# analysis). Still labeled PROJECTED for the box-1 GH200 frame until its own
# production-batch receipt lands. Was 16,943 on the 2026-08-25 2k cell; the
# 2026-08-29 re-anchor (101.737 ms) moves the fit to ~13.0k.
TRAIN_CROSSOVER_TOKENS = 13008  # PROJECTED for the GH200 frame (flat per-token model, 2026-08-29 2k anchor); measured 21-24k @ B4 / ~16k @ B16 on H100 d4096


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
    # The training advantage is COST-WEIGHTED over the curriculum's context mix,
    # not r at one context. Quoting r(2,048) for all of training was the model's
    # old implicit assumption and it is the pessimistic corner: our per-token cost
    # is flat in context and the transformer's grows, so every long-context phase
    # moves this number our way. See training_advantage_mix().
    curriculum = w.get("train_curriculum", "legacy_short")
    train_adv = training_advantage_mix(curriculum)["advantage"]
    if train_share > 0:
        # Harmonic (cost-weighted) blend across training and serving.
        blended = 1.0 / (train_share / train_adv + (1.0 - train_share) / serving)
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
        "train_curriculum": curriculum if isinstance(curriculum, str) else "custom",
        "train_advantage": train_adv,
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
    # SUPERSEDED AS A HEADLINE 2026-08-24. The kernel campaign is the line item
    # against it, and it landed: the single-layer frame now reads x1.717 at
    # T=2048 (RE-BASED AGAIN 2026-08-25 (3)) and x1.786 at T=8192
    # (KERNEL_CAMPAIGN_20260824). This row stays as a dated d1536/27L fp16
    # receipt; quote the newer one.
    "superseded_by": "KERNEL_CAMPAIGN_20260824",
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

    The counterweight is KERNEL_CAMPAIGN_20260824: at 2,048-token context on one
    GPU a training step still costs x1.717 MORE (x1.786 at 8,192; 2k RE-BASED
    AGAIN 2026-08-25 (3)), which the equal-quality parameter ratio takes below 1
    at trillion scale but which is still a loss at matched size and short
    context. The economics slide excludes training for that reason. It used to
    read x6.46 (STEP_GAP_20260814, pre-campaign kernels)."""
    return workload_compute_advantage(w)["blended"] * param_matching_gain(n_tf)


# ---- estimated TRAINING cost at equal quality ---------------------------------
# Everything above prices SERVING. This pair prices TRAINING, which is the axis
# the 2026-08-24 kernel campaign moved. It is an ESTIMATE built from one MEASURED
# factor and one FIT-DERIVED factor; label it that way wherever it is shown.
#
# Training compute is ~6*N*D FLOPs (N params, D tokens). At equal quality the
# owned arm needs N/s parameters, where s = param_matching_gain(N). Two token
# regimes then give two different answers, and they differ by exactly one power
# of s:
#
#   compute-optimal (Chinchilla, D proportional to N)  the smaller model also
#       trains on proportionally fewer tokens, so FLOPs fall as 1/s^2 and the
#       cost ratio is s^2 / r(T).
#   fixed token budget (data-constrained)  D is the same on both arms, FLOPs
#       fall as 1/s, and the cost ratio is s / r(T).
#
# r(T) is the measured step-time ratio at MATCHED parameters. The memory ratio
# is deliberately absent: it caps per-GPU batch density and cluster shape, not
# FLOPs, so folding it in here would double-count.


# ---- r(T): the step-time ratio as a function of context -----------------------
# Added 2026-08-24 (2). r(T) used to be a single naive extrapolation: hold the
# measured 2k->8k ratio decay constant forever. That form is known to be wrong,
# and the file already said so -- holding a growth RATIO constant assumes the
# transformer's attention term stops growing (the old x1.43-at-32k check against
# the B4/T32,768 ~parity cell is RETIRED 2026-08-25 with that cell).
#
# The honest form is the one the hardware actually has. At MATCHED tokens per step
# (both measured cells carry 32,768 tokens), cost per token is:
#
#   transformer   base + attn*T      linear in T per token, because FlashAttention
#                                    makes attention linear per token (quadratic
#                                    over the sequence) on top of a T-independent
#                                    MLP/projection term
#   BDM           flat in T          O(1) recurrent state, no attention term; the
#                                    +5.8%-per-4x residual once quoted here is
#                                    RETIRED 2026-08-25 as a B4 occupancy artifact
#
# base and attn are fitted to the TWO MEASURED cells, so the fit has no free
# parameters left over. The former third-cell validation (the equal-token
# B4/T32,768 ~parity cell) is RETIRED 2026-08-25 along with the growth residual:
# it shares the same B4 batch-occupancy artifact and is not independent ground
# truth. At the flat per-token model the crossover is a PROJECTED T ~ 16,900
# (TRAIN_CROSSOVER_TOKENS); the bucket-attribution re-measurement is owed upstream.
#
# Labeling rule for every surface: the 2,048 (2026-08-25 receipt) and 8,192
# (2026-08-24) cells are MEASURED; everything past 8k is MODELED (this fit,
# extrapolated). The old constant-decay curve is
# kept as mode="constant_decay" and is the CONSERVATIVE bound -- it understates our
# long-context advantage badly (x1.03 at 262k against this fit's x0.19), because
# holding a growth RATIO constant implicitly assumes the transformer's attention
# term stops growing.
def _train_cost_fit():
    """Fit (base, attn) for the transformer and the BDM per-token constant."""
    k = KERNEL_CAMPAIGN_20260824
    tok = TRAIN_MATCHED_TOKENS_PER_STEP
    tf_2k, tf_8k = k["tf_ms_2k"] / tok, k["tf_ms_8k"] / tok
    attn = (tf_8k - tf_2k) / (8192.0 - 2048.0)
    base = tf_2k - attn * 2048.0
    return {"tf_base_ms_per_token": base, "tf_attn_ms_per_token_sq": attn,
            "bdm_ms_per_token_2k": k["battn_ms_2k"] / tok}


TRAIN_MATCHED_TOKENS_PER_STEP = 32768.0  # both measured cells carry this many tokens
TRAIN_COST_FIT_20260824 = _train_cost_fit()
TRAIN_COST_FIT_20260824.update({
    "status": "MODELED — fitted to 2 MEASURED cells (2k re-based 2026-08-25)",
    "form": "transformer ms/token = base + attn*T; BDM ms/token = c (flat; the "
            "x1.058-per-4x residual is RETIRED as a B4 occupancy artifact)",
    "measured_ctx": (2048, 8192),
    "validation": "the former third-cell check (equal-token B4/T32,768 ~parity) is "
                  "RETIRED 2026-08-25 — it shares the B4 occupancy artifact; the "
                  "flat model puts the crossover at a PROJECTED T ~ 16,900",
    "receipt": "2026-08-24 GH200 megakernel measurement (KERNEL_CAMPAIGN_20260824); "
               "2k cell re-based on the 2026-08-25 ledger-of-record receipt",
})


def tf_ms_per_token(ctx_tokens):
    """MODELED. Transformer training cost per token at context T: base + attn*T."""
    f = TRAIN_COST_FIT_20260824
    return f["tf_base_ms_per_token"] + f["tf_attn_ms_per_token_sq"] * float(ctx_tokens)


def bdm_ms_per_token(ctx_tokens):
    """MODELED. BDM training cost per token at context T — flat (no attention
    term; the recurrent state is O(1) in T. The x1.058-per-4x residual is RETIRED
    2026-08-25 as a B4 occupancy artifact; ctx_growth_per_4x_battn = 1.0)."""
    import math

    f = TRAIN_COST_FIT_20260824
    g4 = KERNEL_CAMPAIGN_20260824["ctx_growth_per_4x_battn"]
    return f["bdm_ms_per_token_2k"] * g4 ** math.log(float(ctx_tokens) / 2048.0, 4.0)


def training_step_ratio(ctx_tokens, mode="fitted"):
    """BDM / transformer training step-time ratio at matched parameters and
    matched tokens per step. Above 1 means we are slower.

    mode="fitted" (default) is the per-token cost model above: MEASURED at 2,048
    and 8,192 tokens, MODELED beyond. mode="constant_decay" is the old
    naive form (hold the 2k->8k ratio decay constant), kept as the CONSERVATIVE
    bound because it assumes the transformer's attention term stops growing.
    """
    import math

    k = KERNEL_CAMPAIGN_20260824
    if mode == "constant_decay":
        quadruplings = math.log(ctx_tokens / 2048.0, 4.0)
        return k["step_ratio_2k"] * k["ratio_decay_per_4x"] ** quadruplings
    if mode != "fitted":
        raise ValueError(f"unknown mode {mode!r}; use 'fitted' or 'constant_decay'")
    return bdm_ms_per_token(ctx_tokens) / tf_ms_per_token(ctx_tokens)


def training_context_crossover(mode="fitted", lo=2048.0, hi=1_048_576.0):
    """Context at which r(T) crosses 1 — past it we train more cheaply per token."""
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if training_step_ratio(mid, mode=mode) > 1.0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def training_cost_saving(n_tf, ctx_tokens, chinchilla=True, fit=None):
    """ESTIMATE. Transformer training cost divided by ours at EQUAL QUALITY.

    Above 1 we are cheaper to train. chinchilla=True is the compute-optimal
    regime (D proportional to N -> s^2 / r); False is the fixed-token-budget
    regime (D equal -> s / r). See the block comment above for the derivation and
    for what is measured versus fitted versus extrapolated.
    """
    s = param_matching_gain(n_tf, fit)
    return (s * s if chinchilla else s) / training_step_ratio(ctx_tokens)


# ---- the training CURRICULUM: a mix of contexts, cost-weighted ----------------
# Added 2026-08-24 (2). Quoting r at a single context silently assumes ALL training
# happens there, and the model's old TRAIN_ADVANTAGE_SHORT_SEQ assumed 2,048 tokens
# for everything. Modern training is a CURRICULUM: a short-context bulk phase, then
# long-context extension phases, then long-rollout RL, each with its own T.
#
# The correct training advantage over a mix is the ratio of COSTS integrated over
# the mix, NOT the average of the per-context ratios:
#
#     advantage = SUM_i share_i * tf_per_token(T_i)  /  SUM_i share_i * bdm_per_token(T_i)
#
# That distinction is the whole point, and it runs strongly in our favour: the
# transformer's per-token cost GROWS with T while ours is flat, so long-context
# tokens dominate the transformer's bill even as a small token minority. In the
# modern_standard mix below, 10% of tokens sit at 262k and account for 45% of the
# transformer's training cost; in frontier_rl, 20% of tokens at 1M account for 66%.
# Averaging the ratios instead would hide exactly that.
#
# SCENARIO SHAPES ARE ILLUSTRATIVE ASSUMPTIONS, AND UNCITED ON PURPOSE. This repo
# carries no bibliography entry and no prose for any lab's training curriculum --
# references.bib is entirely SSM/linear-RNN/parallel-scan work, with nothing on
# long-context extension phases (checked 2026-08-24). Rather than attach a
# half-remembered token budget to a real lab, the shares are exposed as KNOBS in
# the app and labelled ILLUSTRATIVE everywhere they are shown. Anyone who wants a
# cited curriculum has to add verified bib entries first (the frontier long-context
# technical reports are the obvious source) and check the token figures against the
# report text. Only legacy_short is anchored to something measured, and only
# because it is the degenerate one-context case that reproduces
# TRAIN_ADVANTAGE_SHORT_SEQ exactly.
#
# What does NOT depend on the shares: the per-token cost model underneath, which is
# fitted to measured cells, and the qualitative result that ANY meaningful
# long-context phase flips the training term from a drag to a contributor. The
# shares only set the magnitude.
TRAINING_CURRICULA = {
    "legacy_short": {
        "mix": ((2048, 1.00),),
        "label": "Legacy short (all training at 2k)",
        "status": "MEASURED (degenerate one-context case)",
        "note": "reproduces TRAIN_ADVANTAGE_SHORT_SEQ = x0.583 exactly (RE-BASED AGAIN "
                "2026-08-25 (3); was x0.449); the assumption the model used to make "
                "implicitly for ALL training",
    },
    "modern_standard": {
        "mix": ((8192, 0.70), (65536, 0.20), (262144, 0.10)),
        "label": "Modern standard (8k bulk + 64k/256k extension)",
        "status": "ILLUSTRATIVE ASSUMPTION",
        "note": "a short-context bulk phase plus staged long-context extension; shares "
                "are knobs, not a measurement of any lab's recipe",
    },
    "long_context": {
        "mix": ((8192, 0.40), (65536, 0.25), (262144, 0.25), (1048576, 0.10)),
        "label": "Long-context heavy (frontier extension phases)",
        "status": "ILLUSTRATIVE ASSUMPTION",
        "note": "meaningful 256k-1M phases, as frontier long-context models require",
    },
    "frontier_rl": {
        "mix": ((8192, 0.30), (65536, 0.20), (262144, 0.30), (1048576, 0.20)),
        "label": "Frontier + long-rollout RL",
        "status": "ILLUSTRATIVE ASSUMPTION",
        "note": "adds long reasoning/agent rollouts, which are long-context by "
                "construction and are becoming a large share of frontier training spend",
    },
}


def training_advantage_mix(curriculum="modern_standard", mode="fitted"):
    """Cost-weighted training advantage (transformer cost / our cost) over a mix
    of context lengths. Above 1 we are cheaper to train at matched parameters.

    `curriculum` is a name from TRAINING_CURRICULA or a [(context_tokens, token_share), ...]
    sequence. Shares are normalised. Returns the advantage plus, for every rung,
    its share of TOKENS and its share of the TRANSFORMER'S COST -- the two differ
    sharply, and showing both is the point of the function.
    """
    if isinstance(curriculum, str):
        mix = TRAINING_CURRICULA[curriculum]["mix"]
    else:
        mix = tuple(curriculum)
    tot_share = float(sum(s for _, s in mix))
    if tot_share <= 0:
        raise ValueError("curriculum token shares must sum to something positive")
    rungs, tf_cost, bdm_cost = [], 0.0, 0.0
    for ctx, share in mix:
        w = float(share) / tot_share
        tf_c, bdm_c = w * tf_ms_per_token(ctx), w * bdm_ms_per_token(ctx)
        tf_cost += tf_c
        bdm_cost += bdm_c
        rungs.append({"ctx": int(ctx), "token_share": w, "_tf": tf_c, "_bdm": bdm_c,
                      "step_ratio": training_step_ratio(ctx, mode=mode)})
    for r in rungs:
        r["tf_cost_share"] = r.pop("_tf") / tf_cost if tf_cost else 0.0
        r["bdm_cost_share"] = r.pop("_bdm") / bdm_cost if bdm_cost else 0.0
    return {
        "advantage": tf_cost / bdm_cost,
        "effective_step_ratio": bdm_cost / tf_cost,
        "rungs": rungs,
        "name": curriculum if isinstance(curriculum, str) else "custom",
    }


# The advantage a curriculum must beat before adding training RAISES the headline.
# Crossing 1 only makes training stop being a loss; the blended lever is a
# harmonic (cost-weighted) mean of the training and serving advantages, so mixing
# training in pulls the blend toward whichever side is worse. Training therefore
# has to beat the SERVING blend (x2.19), not x1, to move the dollars up.
def training_helps_headline_threshold():
    """The training advantage above which train_share > 0 raises the headline."""
    return workload_compute_advantage({"train_share": 0.0})["serving"]


def _check_curriculum_degenerate_case():
    """legacy_short must reproduce the constant the model used to carry."""
    adv = training_advantage_mix("legacy_short")["advantage"]
    assert abs(adv - TRAIN_ADVANTAGE_SHORT_SEQ) < 1e-9, (
        f"legacy_short curriculum gives x{adv} but TRAIN_ADVANTAGE_SHORT_SEQ is "
        f"x{TRAIN_ADVANTAGE_SHORT_SEQ} — the per-token cost model no longer reduces "
        f"to the measured 2k cell"
    )
    return adv


def _check_train_cost_fit_validation():
    """The fit is built from the 2k and 8k cells only. RE-BASED 2026-08-25: the
    fit no longer targets the B4/T32,768 'parity' cell -- that cell shares the
    B4-batch occupancy artifact of the growth-rate pair it was meant to check
    independently, so it is not an independent target any more (see
    ctx_growth_per_4x_battn and TRAIN_CROSSOVER_TOKENS comments). This guard
    instead pins the flat-model's own internal arithmetic: with a perfectly
    flat per-token cost, r(32,768) must equal bdm_ms_per_token_2k / tf(32,768)
    exactly, and the fitted crossover must equal TRAIN_CROSSOVER_TOKENS exactly
    -- both are now derived, not independently measured, so the guard is against
    silent drift in the fit code, not against a receipt."""
    r32 = training_step_ratio(32768)
    f = TRAIN_COST_FIT_20260824
    expected_r32 = f["bdm_ms_per_token_2k"] / tf_ms_per_token(32768)
    assert abs(r32 - expected_r32) < 1e-9, (
        f"r(32,768) = {r32:.4f} disagrees with the flat model's own arithmetic "
        f"({expected_r32:.4f}) -- the per-token cost fit or the growth constant drifted"
    )
    x = training_context_crossover()
    assert abs(x - TRAIN_CROSSOVER_TOKENS) / TRAIN_CROSSOVER_TOKENS < 0.01, (
        f"fitted crossover T={x:,.0f} disagrees with TRAIN_CROSSOVER_TOKENS="
        f"{TRAIN_CROSSOVER_TOKENS:,} -- update the constant if the fit changed on purpose"
    )
    return r32, x


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
_check_train_cost_fit_validation()
_check_curriculum_degenerate_case()
COMPUTE_LEVER_20260814 = quality_matched_compute_lever()
# GLOBALS carries the TODAY lever as its default; it must stay pinned to the
# computed quality-matched value so no surface quotes a stale number.
assert abs(GLOBALS["flop_factor"] - COMPUTE_LEVER_20260814) < 0.05, (
    f"GLOBALS['flop_factor']={GLOBALS['flop_factor']} drifted from the computed "
    f"quality-matched lever {COMPUTE_LEVER_20260814:.4f}"
)


# ---- CAMPAIGN-LANDED scenario (2026-08-31 ruling) ------------------------------
# The single scenario the standalone FY26/FY27 charts run on (replacing the
# Today-vs-Ceiling pair THERE; decks/xlsx/app keep the two-scenario convention).
# It prices the state after the funded kernel edits land, on the user's stated
# targets. Every entry is a TARGET, not a receipt -- the whole scenario is an
# ESTIMATE until the aggregate-decode measurement lands (see 'receipt_needed').
CAMPAIGN_LANDED_20260831 = {
    "status": "ESTIMATE — post-campaign targets; aggregate-decode receipt PENDING",
    # Our decode after the edits: 2.5 ms wall per generated token, SINGLE stream,
    # d2048, T=2,048 -- context-flat (no KV re-read), so the per-token edge over
    # the transformer grows with context (tf: 1.58 ms + 0.051 us/token of ctx).
    "decode_own_ms_per_token": 2.5,
    # Concurrency multiplier: one stream nowhere near occupies the card, so the
    # aggregate rate scales with resident streams. 64 is HELD from the measured
    # 2026-08-14 cell (64 streams in 1.62 GB, a grid cap, 9.5% GPU-busy) -- an
    # ESTIMATE that batching keeps the 2.5 ms step at that concurrency.
    "streams_per_gpu": 64,
    # Training-side targets (label the scenario; the headline lever is a SERVING
    # claim with train_share=0, so these do not enter the dollars):
    "train_mem_ratio_target": 1.0,   # TF training-memory parity. Path is real: banked
                                     # 1.552 -> 1.361 (deadtape -1,736 MiB, 2026-08-30);
                                     # gap identity says 83.2% of the remaining gap is
                                     # transient-at-peak, the campaign's stated hunting
                                     # ground. RISK: the pending TF-bar fairness ruling
                                     # (lean-bwd fold) moves the goalpost to ~1.776.
    "train_step_ratio_2k_target": 1.3,  # x1.3 at T=2,048, improving with context (flat per-token)
    # How training enters the DOLLARS (2026-08-31 (2) user ruling: training is a
    # big piece of accelerator cost, so the landed training targets move the
    # dollar estimates, not just the labels). Share is ANALYST-GROUNDED, mix is
    # ILLUSTRATIVE:
    # train_share sources (all 2026): Gartner -- inference 55% of AI-optimized
    # IaaS spend ($23.3B vs $19B training) -> training ~45%; Deloitte --
    # inference 2/3 of all AI compute (was 1/2 in 2025) -> training ~33%;
    # hyperscaler compute demand ~60-70% inference -> training 30-40%.
    # Band 0.30-0.45, midpoint taken. Raising the share is the CONSERVATIVE
    # direction (training is the weaker lever).
    "train_share": 0.35,             # fraction of accelerator cost spent training (analyst band 0.30-0.45)
    "train_curriculum": "modern_standard",  # the mix the training term is cost-weighted over
    "receipt_needed": "aggregate decode throughput on the post-edit kernel: tok/s/GPU at "
                      "d2048 over a stream sweep (1/8/64) x context sweep (2k/32k/262k). "
                      "Turns this scenario's decode leg from ESTIMATE into MEASURED. Full "
                      "protocol: experiments/receipts/decode_aggregate_receipt_20260831_instructions.md",
}


def campaign_landed_flop_lever(streams_per_gpu=None, decode_ms_per_token=None, w=None):
    """The campaign-landed compute lever: the Today lever's structure (serving
    blend x fit-derived equal-quality parameter ratio) with OUR decode leg
    replaced by the post-edit target (streams_per_gpu x 1000/decode_ms tok/s/GPU,
    context-flat). Transformer side stays the measured 1/context law at its KV
    ceiling. ESTIMATE until the receipt in CAMPAIGN_LANDED_20260831 lands."""
    c = CAMPAIGN_LANDED_20260831
    n_streams = streams_per_gpu if streams_per_gpu is not None else c["streams_per_gpu"]
    ms = decode_ms_per_token if decode_ms_per_token is not None else c["decode_own_ms_per_token"]
    w = dict(WORKLOAD, **(w or {}))
    ctx = float(w["context_tokens"])
    r = float(w["in_out_ratio"])
    gpus = SERVING["gpus_per_box"]
    econ = serving_economics(ctx)
    tf_dec_tps = econ["tf_tokens_per_s_box"]
    own_dec_tps = n_streams * 1000.0 / ms * gpus
    tf_pf_tps = prefill_tokens_per_s("transformer", ctx) * gpus
    own_pf_tps = prefill_tokens_per_s("bdm", ctx) * gpus
    tf_total = r / tf_pf_tps + 1.0 / tf_dec_tps
    own_total = r / own_pf_tps + 1.0 / own_dec_tps
    return (tf_total / own_total) * param_matching_gain(DECK_DEPLOYMENT_SCALE)


CAMPAIGN_LANDED_FLOP_LEVER = campaign_landed_flop_lever()  # = 347.7 at the 64-stream / 2.5 ms targets


def campaign_landed_train_ratio(ctx_tokens):
    """r(T) with the campaign-landed TRAINING target: x1.3 at T=2,048, our
    per-token cost flat (MEASURED flat, 2026-08-29/31 receipts), transformer
    base + attn*T from the current fit. TARGET, not a receipt."""
    f = TRAIN_COST_FIT_20260824
    tf_2k = f["tf_base_ms_per_token"] + f["tf_attn_ms_per_token_sq"] * 2048.0
    bdm_target = CAMPAIGN_LANDED_20260831["train_step_ratio_2k_target"] * tf_2k
    return bdm_target / (f["tf_base_ms_per_token"]
                         + f["tf_attn_ms_per_token_sq"] * float(ctx_tokens))


def campaign_landed_train_crossover():
    """Context where the landed training target reaches parity: T* where
    tf(T) = 1.3 * tf(2,048). ~6,600 tokens on the current fit -- i.e. under the
    targets we win training at 8k and above. TARGET-derived."""
    f = TRAIN_COST_FIT_20260824
    tf_2k = f["tf_base_ms_per_token"] + f["tf_attn_ms_per_token_sq"] * 2048.0
    bdm_target = CAMPAIGN_LANDED_20260831["train_step_ratio_2k_target"] * tf_2k
    return (bdm_target - f["tf_base_ms_per_token"]) / f["tf_attn_ms_per_token_sq"]


def campaign_landed_train_advantage(curriculum="modern_standard"):
    """Cost-weighted training advantage over a curriculum mix, at the landed
    training target (x1.3 @ 2k, flat per-token). >1 = training contributes."""
    spec = TRAINING_CURRICULA[curriculum]
    f = TRAIN_COST_FIT_20260824
    tf_2k = f["tf_base_ms_per_token"] + f["tf_attn_ms_per_token_sq"] * 2048.0
    bdm_target = CAMPAIGN_LANDED_20260831["train_step_ratio_2k_target"] * tf_2k
    tf_cost = sum((f["tf_base_ms_per_token"]
                   + f["tf_attn_ms_per_token_sq"] * ctx) * share
                  for ctx, share in spec["mix"])
    own_cost = sum(bdm_target * share for _, share in spec["mix"])
    return tf_cost / own_cost


def campaign_landed_blended_lever(train_share=None, curriculum=None):
    """The landed FLOP lever with TRAINING FOLDED IN (2026-08-31 (2) user
    ruling: training is a big piece of the cost, so the landed targets move the
    dollars). Harmonic blend of the serving lever (x348) and the landed
    training target advantage over the curriculum mix. Defaults come from
    CAMPAIGN_LANDED_20260831 (train_share 0.20, modern_standard -- both
    ILLUSTRATIVE knobs, not measurements)."""
    c = CAMPAIGN_LANDED_20260831
    ts = c["train_share"] if train_share is None else float(train_share)
    cur = curriculum or c["train_curriculum"]
    if ts <= 0:
        return CAMPAIGN_LANDED_FLOP_LEVER
    adv = campaign_landed_train_advantage(cur)
    return 1.0 / (ts / adv + (1.0 - ts) / CAMPAIGN_LANDED_FLOP_LEVER)


def campaign_landed_reduction(train_share=None, curriculum=None):
    """Cost-weighted reduction under the landed scenario, training folded in
    (defaults: 20% share, modern mix -> ~29x). train_share=0 recovers the
    serving-only ~140x; frontier_rl at 20% reads ~84x. The spread exists
    because the training lever (x2.9 at the target) is far weaker than the
    serving one (x348) -- Amdahl, as ever."""
    lever = campaign_landed_blended_lever(train_share, curriculum)
    g = GLOBALS
    return 1.0 / (g["mem_share"] / g["mem_factor"] + (1 - g["mem_share"]) / lever)


# One row per assumption, visible on every surface that renders the serving /
# training numbers (app tab, workbook sheet, __main__ print). status is one of
# MEASURED / PROJECTION / ASSUMPTION.
SERVING_TRAINING_ASSUMPTIONS = [
    ("Compute lever (economics slide)", f"x{COMPUTE_LEVER_20260814:.2f} = the x{workload_compute_advantage()['blended']:.2f} MEASURED serving blend x the x{param_matching_gain(DECK_DEPLOYMENT_SCALE):.2f} FIT-DERIVED equal-quality parameter ratio at trillion scale", "MEASURED x PROJECTION", "workload_compute_advantage() x quality_matched_compute_lever(); receipts derived.json + quality_fit_v4.json"),
    ("Equal-quality parameter matching", "sealed 2026-08-14 refit, 4 rungs per family (47M-663M params): the fits cross at 392M and bAttention then matches the transformer fit's quality on 84.2% of the params at 1B, 55.2% at 10B, 36.2% at 100B, 23.7% at 1T, 15.5% at 10T. The receipt's own words: 'a projection of the two fits, not a measurement'", "PROJECTION", "quality_fit_v4.json param_matching; fits over the sealed rope-convention ladder"),
    ("Single-GPU training step (AGAINST us)", f"RE-BASED 2026-08-24: on one GH200, single layer, fwd+bwd, bf16, checkpointing off both arms, against a MODERN transformer block (24Q/4KV, head_dim 256, RoPE 64, gated attention), a bAttention step costs x{KERNEL_CAMPAIGN_20260824['step_ratio_2k']:.2f} MORE at T=2,048 ({KERNEL_CAMPAIGN_20260824['battn_ms_2k']:.1f} vs {KERNEL_CAMPAIGN_20260824['tf_ms_2k']:.1f} ms/step) and x{KERNEL_CAMPAIGN_20260824['step_ratio_8k']:.2f} at T=8,192. It read x6.116 (400.4 ms) the same morning and x6.46 on the older d1536/27L fp16 601-step receipt. Fwd x1.82, bwd x2.35. Training is still excluded from the serving claim, but the gap is now a kernel-program line item with a funded gate, not a structural loss", "MEASURED", "2026-08-24 GH200 megakernel measurement; prior: dtype_trio_v4.json rows[fp16]"),
    ("Training PEAK MEMORY (AGAINST us)", f"x{KERNEL_CAMPAIGN_20260824['mem_ratio_2k']:.3f} at T=2,048 ({KERNEL_CAMPAIGN_20260824['battn_peak_mib_2k']:,.1f} vs {KERNEL_CAMPAIGN_20260824['tf_peak_mib_2k']:,.1f} MiB) and x{KERNEL_CAMPAIGN_20260824['mem_ratio_8k']:.3f} at T=8,192, down from x{KERNEL_CAMPAIGN_20260824['mem_ratio_2k_precampaign']:.3f} the same morning. This is TRAINING peak memory and is NOT the mem_factor lever, which is SERVING state (O(1) recurrent state vs O(T) KV cache) and is unaffected. It is not an input to the cost model; it caps per-GPU batch density", "MEASURED", "2026-08-24 GH200 megakernel measurement"),
    ("Kernel program (TARGET, not a result)", f"the funded 8k-win gate is a T=8,192 step at or below {KERNEL_CAMPAIGN_20260824['target_8k_win_gate_ms']:.2f} ms, i.e. x{KERNEL_CAMPAIGN_20260824['target_8k_gap_remaining']:.3f} still to remove (44% of the step). Named levers: {KERNEL_CAMPAIGN_20260824['target_levers']}. Memory target x{KERNEL_CAMPAIGN_20260824['target_mem_ratio']:.2f} (parity or below). CEILING_PREFILL_SPEEDUP = the MEASURED x{KERNEL_SPEEDUP_REALIZED_20260824:.3f} already banked x this x{KERNEL_SPEEDUP_REMAINING_TARGET:.3f} target", "TARGET", "kernel-campaign program plan, 2026-08-24 — no receipt behind the target half"),
    ("Context scaling of the training gap", f"at matched tokens/step, 2k -> 8k (4x context): our step grows x{KERNEL_CAMPAIGN_20260824['ctx_growth_per_4x_battn']:.3f}, the transformer's x{KERNEL_CAMPAIGN_20260824['ctx_growth_per_4x_tf']:.3f}, so the ratio decays x{KERNEL_CAMPAIGN_20260824['ratio_decay_per_4x']:.3f} per 4x. The equal-token B4/T32,768 ~parity cell is RETIRED 2026-08-25 as a B4 occupancy artifact (it shared the retired growth residual); the flat per-token model puts the crossover at a PROJECTED T~16,900. Holding the old x0.802 decay constant instead would say x1.43 at 32k and a crossover near T~310,000 - that is the CONSERVATIVE extrapolation, because the transformer's quadratic attention term makes its per-4x growth rise with context while ours stays flat", "MEASURED (2 ratio points) + PROJECTION", "2026-08-24 GH200 megakernel measurement; 2k cell re-based 2026-08-25"),
    ("Training CONTEXT MIX (reads the other way)", f"quoting one context for all of training is the pessimistic corner, and it was this model's old implicit assumption. Cost-weighted over a curriculum the training term flips from a DRAG to a CONTRIBUTOR: x{training_advantage_mix('legacy_short')['advantage']:.3f} at 100% 2k, x{training_advantage_mix('modern_standard')['advantage']:.2f} at a modern 8k/64k/256k mix, x{training_advantage_mix('long_context')['advantage']:.2f} long-context-heavy, x{training_advantage_mix('frontier_rl')['advantage']:.2f} with long RL rollouts. The mechanism: our per-token cost is FLAT in context and the transformer's GROWS, so long-context tokens dominate its bill as a token minority — 10% of tokens at 262k is 47% of its training cost. r(T) is MEASURED at 2k/8k and MODELED beyond; the curriculum SHARES are ILLUSTRATIVE and exposed as knobs (this repo carries no citation for any lab's recipe)", "MEASURED r(T) + ILLUSTRATIVE mix", "TRAIN_COST_FIT_20260824, TRAINING_CURRICULA, training_advantage_mix()"),
    ("Long-context training memory (EXCLUDED, runs our way)", f"at 262k+ the transformer additionally pays sequence-parallelism and activation-memory costs that an O(1) recurrent state does not: its KV grows at {KV_MB_PER_TOKEN_PER_STREAM * 1024:.0f} KB per token of context per stream, which is what forces sharding, ring/Ulysses attention and their communication overhead. Our own measurement shows the asymmetry in the small: peak training memory grew only x{KERNEL_CAMPAIGN_20260824['mem_ratio_8k'] / KERNEL_CAMPAIGN_20260824['mem_ratio_2k']:.3f} relative to the transformer's from 2k to 8k (x{KERNEL_CAMPAIGN_20260824['mem_ratio_2k']:.3f} -> x{KERNEL_CAMPAIGN_20260824['mem_ratio_8k']:.3f}), i.e. the ratio is FLAT-to-falling in context while the absolute gap the transformer must shard grows. NOT in the cost model — the training advantage above counts FLOPs/step-time only, so the direction of this omission is CONSERVATIVE (it understates us at long context). Quantifying it needs a multi-GPU long-context training receipt we do not have", "DIRECTIONAL — measured components, not a costed lever", "KERNEL_CAMPAIGN_20260824 mem_ratio_2k/8k; KV_MB_PER_TOKEN_PER_STREAM"),
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
    ("Compute lever (flop_factor)", "cross-family prefill, bf16, 1xH100, parameter counts exactly matched, block scope (1 layer, batch 16): TF/bAttention 4.02 at d512/1M, 3.83 at d1024/1M, 2.01 at d2048/262k. Short context favours the transformer and is stated so. NOTE: measured on the PRE-CAMPAIGN kernels and not re-run since 2026-08-24, so the TODAY lever is conservative; the realized x3.936 sits in CEILING_PREFILL_SPEEDUP, not here", "MEASURED", "matched_d{512,1024,2048}_h100.json (2026-07-21)"),
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
    """Return all derived numbers for one company in one year ('fy25', 'fy26' or
    'fy27' -- fy27 is STREET-ESTIMATE grade, see the per-company fy27 comments). All $B."""
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
    rev = comp["ai_rev"][{"fy25": 0, "fy26": 1, "fy27": 2}[year]]
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


# ---- the "$159B/yr" headline family, pinned ------------------------------------
# The headline is MODEL-COMPUTED -- it is compute_year(GLOBALS, COMPANIES, 'fy25')
# rolled to TOTAL -- but every external surface (deck YAMLs, app copy, workbook
# prose) quotes it as a hand-transcribed literal. headline_family() recomputes it
# and the assert below is the guard, so a lever change can never leave the decks
# saying a number this file no longer produces.
#
# RE-VERIFIED 2026-08-24 against the re-based kernel levers: the TODAY family did
# NOT move, and the reason is structural. WORKLOAD['train_share'] defaults to 0,
# so the headline is a pure SERVING claim, and the 2026-08-24 re-base touched only
# the TRAINING and CEILING levers. Both serving inputs are untouched -- mem_factor
# 100, and flop_factor x9.24 whose x4.02 prefill anchor was deliberately not re-run
# (see the module docstring: conservative by construction). What did move:
#
#   TRAIN_ADVANTAGE_SHORT_SEQ  1/9.6 -> 1/2.226. Training remains a DRAG on the
#       blend at every share -- x0.449 on its own against the serving blend's
#       x2.19 -- so switching it on still LOWERS the headline: at train_share 0.2
#       the blend is x1.23 (was x0.44) and FY25 reads $153B, not $159B. What
#       flipped sign is where the BLEND crosses 1, i.e. the break-even training
#       share, ~0.06 -> ~0.30. That is a sensitivity knob, not the headline.
#   RE-BASED AGAIN 2026-08-25 (3): TRAIN_ADVANTAGE_SHORT_SEQ 1/2.226 -> 1/1.934
#       (x0.449 -> x0.517 on its own); at train_share 0.2 the legacy_short
#       blend is x1.33 and FY25 reads $154.15B, still below serving-only $159B
#       (still dilutes -- legacy_short alone never crosses the serving blend).
#   CEILING_FLOP_LEVER  22.09 -> 20.49, cost-weighted ceiling x41.48 -> x39.18.
#       FY25 ceiling cut $163.04B -> $162.80B, still "~$163B" at quoted rounding.
#       The only quoted value that moved is the Today->Ceiling dollar gap,
#   RE-BASED AGAIN 2026-08-25 (3): CEILING_FLOP_LEVER 20.49 -> 24.77, cost-
#       weighted ceiling x39.18 -> x45.15 (2k receipt re-base, see the top
#       docstring paragraph). FY25 ceiling cut $162.80B -> $163.37B, still
#       "~$163B" -- every quoted dollar figure stays within tolerance; only
#       ceiling_reduction (39.0 -> update to 45.1) and ceiling_dollar_gap_pct
#       (2.5 -> update to 2.85) move in HEADLINE_QUOTED_20260824 below.
#       2.65% -> 2.50%; surfaces that said "~3%" now say "~2.5%".
def headline_with_training(curriculum, train_share, g=None, companies=None):
    """The FY25 headline once TRAINING enters the lever at a given context mix.

    Every surface that shows the training scenarios reads this, so the app, the
    workbook, the chart and the deck copy cannot quote different dollars. Returns
    the training advantage, the blended lever, the reduction and the FY25 cut.

    Read the `raises_headline` flag before telling the story: a curriculum crossing
    x1 makes training stop being a LOSS, but the blended lever is a cost-weighted
    harmonic mean, so training only RAISES the dollar headline once it beats the
    SERVING advantage (x2.19). Between x1 and x2.19 training is profitable and
    still dilutive to this particular number.
    """
    g = g if g is not None else GLOBALS
    companies = companies if companies is not None else COMPANIES
    w = workload_compute_advantage(
        {"train_share": train_share, "train_curriculum": curriculum}
    )
    lever = w["blended"] * param_matching_gain(DECK_DEPLOYMENT_SCALE)
    gg = dict(g, flop_factor=lever)
    _, t25 = compute_year(gg, companies, "fy25")
    baseline = compute_year(g, companies, "fy25")[1]["spend_cut"]
    return {
        "curriculum": w["train_curriculum"],
        "train_share": float(train_share),
        "train_advantage": w["train_advantage"],
        "effective_step_ratio": 1.0 / w["train_advantage"],
        "blended_lever": w["blended"],
        "flop_factor": lever,
        "reduction": reduction_factor(gg),
        "fy25_spend_cut": t25["spend_cut"],
        "fy25_capitalized": t25["capitalized"],
        "delta_vs_serving_only": t25["spend_cut"] - baseline,
        "raises_headline": w["train_advantage"] > w["serving"],
    }


def headline_family(g=None, companies=None):
    """Recompute every value in the quoted headline family. All $B."""
    g = g if g is not None else GLOBALS
    companies = companies if companies is not None else COMPANIES
    gc = dict(g, flop_factor=CEILING_FLOP_LEVER)
    _, t25 = compute_year(g, companies, "fy25")
    _, t26 = compute_year(g, companies, "fy26")
    _, c25 = compute_year(gc, companies, "fy25")
    return {
        "fy25_spend_cut": t25["spend_cut"],
        "fy25_capitalized": t25["capitalized"],
        "fy25_pct_cut": t25["pct_cut"],
        "fy25_net_arch": t25["net_arch"],
        "fy26_spend_cut": t26["spend_cut"],
        "fy26_capitalized": t26["capitalized"],
        "global_fy25_capitalized": global_estimate(t25, g)["capitalized"],
        "global_fy26_capitalized": global_estimate(t26, g)["capitalized"],
        "fy25_ceiling_spend_cut": c25["spend_cut"],
        "today_reduction": reduction_factor(g),
        "ceiling_reduction": reduction_factor(gc),
        "ceiling_dollar_gap_pct": 100.0 * (c25["spend_cut"] / t25["spend_cut"] - 1.0),
    }


# (key, what the surfaces say, tolerance) — tolerance is the quoted rounding.
HEADLINE_QUOTED_20260824 = (
    ("fy25_spend_cut", 159.0, 0.5),          # "~$159B/yr"
    ("fy25_capitalized", 2600.0, 50.0),      # "~$2.6T capitalized at 6%"
    ("fy25_pct_cut", 0.42, 0.005),           # "~42% of AI spend cut"
    # "burn shrinks to ~ -$136B/yr". Computes to -136.54, and the surfaces quote
    # the difference of the ROUNDED components (-295 + 159 = -136) rather than the
    # rounded difference (-137), so that the slide's waterfall adds up. Tolerance
    # is 1.0 to allow that one-unit narrative rounding, deliberately.
    ("fy25_net_arch", -136.0, 1.0),
    ("fy26_spend_cut", 366.0, 0.5),          # "FY26 r/r ~$366B/yr"
    ("fy26_capitalized", 6100.0, 50.0),      # "~$6.1T"
    ("global_fy25_capitalized", 3300.0, 50.0),   # "global est ~$3.3T FY25"
    ("global_fy26_capitalized", 7600.0, 50.0),   # "~$7.6T FY26"
    ("fy25_ceiling_spend_cut", 163.7, 0.5),  # "Ceiling: ~$164B FY25" (2026-08-29 re-anchor; was 163.0)
    ("today_reduction", 20.0, 0.5),          # "~20x cost-weighted"
    ("ceiling_reduction", 49.6, 0.5),        # "~50x" (2026-08-29 re-anchor: realized x3.94; was 45.1)
    ("ceiling_dollar_gap_pct", 3.1, 0.25),   # "the two differ only ~3% in dollars" (was 2.85)
)

_HEADLINE = headline_family()
for _k, _quoted, _tol in HEADLINE_QUOTED_20260824:
    assert abs(_HEADLINE[_k] - _quoted) <= _tol, (
        f"headline drift: {_k} computes to {_HEADLINE[_k]:.4f} but the decks, the app "
        f"and the workbook quote {_quoted} (tolerance {_tol}). Re-quote every surface "
        f"listed in the block above before changing a lever."
    )
del _k, _quoted, _tol


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
    print("== TRAINING is a CURRICULUM, not one context: r(T) MEASURED to 32k, MODELED beyond ==")
    print(f"    r(T):  " + "  ".join(
        f"{T // 1024}k={training_step_ratio(T):.2f}" for T in (2048, 8192, 32768, 262144, 1048576)))
    print(f"    the conservative constant-decay bound instead says  " + "  ".join(
        f"{T // 1024}k={training_step_ratio(T, mode='constant_decay'):.2f}"
        for T in (32768, 262144, 1048576)))
    print(f"    flat per-token model: r(32,768)={training_step_ratio(32768):.3f}; "
          f"crossover T~{training_context_crossover():,.0f} (PROJECTED, B4 parity cell retired as artifact)")
    _thr = training_helps_headline_threshold()
    for _name, _spec in TRAINING_CURRICULA.items():
        _a = training_advantage_mix(_name)
        _h = headline_with_training(_name, 0.20)
        print(f"    {_name:16s} adv x{_a['advantage']:5.2f}  r_eff x{_a['effective_step_ratio']:5.3f}  "
              f"-> FY25 ${_h['fy25_spend_cut']:6.2f}B at a 20% training share "
              f"({_h['delta_vs_serving_only']:+.2f} vs serving-only; "
              f"{'RAISES' if _h['raises_headline'] else 'dilutes'})   [{_spec['status']}]")
        print("        " + "  ".join(
            f"{g['ctx'] // 1024}k: {g['token_share']:.0%} of tokens -> {g['tf_cost_share']:.0%} of TF cost"
            for g in _a["rungs"]))
    print(f"    NOTE: crossing x1 makes training profitable; RAISING the headline needs it to beat the "
          f"SERVING advantage x{_thr:.2f}, since the blend is a cost-weighted harmonic mean.")
    print("    Curriculum token shares are ILLUSTRATIVE knobs -- no lab's recipe is cited or invented.")
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
