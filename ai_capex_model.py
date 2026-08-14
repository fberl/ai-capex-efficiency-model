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

# ---- serving state: MEASURED ---------------------------------------------------
# The transformer's serving state grows with every token of context; ours does
# not. Same session, same box, same dtype.
SERVING_STATE_20260814 = {
    "status": "MEASURED",
    "config": "d1536 / 26 layers / fp16, 8 x H100 80GB",
    "tf_kv_mb_per_token_per_stream": 39936.0 / 262144,  # = 0.15234375 MB per token of context
    "battn_state_mb_per_stream": 0.15234375,  # h-recurrence carry, CONSTANT in context
    "battn_state_scope": "h-recurrence carry; the full recurrent cell (h + M memory) is ~21 MB/stream, also context-flat",
    "tf_measured_point": "39,936 MB of KV for ONE 262,144-token stream; 8 streams OOM an 80 GB card",
    "receipt": "bdm/docs/deck/build/derived.json (decode_multigpu, slide_decode_262k)",
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

# ---- serving-workload compute blend: MEASURED ----------------------------------
# Total transformer serving cost over a request divided by total bAttention cost,
# at the deck's default operating point. A cost ratio, not an average of ratios.
# Computed by the serving module of the deck build (epsilon-rnn
# ai_capex_model.workload_compute_advantage) over the 2026-08-07/08 receipts;
# mirrored here as a constant so this model and the deck quote one number.
SERVING_BLEND_20260808 = {
    "status": "MEASURED",
    "blend": 11.929698,
    "prefill_ratio": 1.173206,
    "decode_ratio": 15.911024,
    "tf_decode_share_of_cost": 0.973364,
    "operating_point": "10:1 input:output, 64k average context, training excluded",
    "caveat": "the transformer is granted an idealized mature stack (paged attention, KV /8, bandwidth-floor serving); "
    "below ~4k context with input-heavy traffic the blend falls under 1 and we are behind",
    "receipt": "bdm/docs/deck/build/derived.json (decode_multigpu, memory_wall_curves) via the deck's serving module",
}

# Kernel operating points the compute lever can be evaluated at. The value is the
# measured transformer-cost / bAttention-cost ratio at that point: above 1 we are
# ahead, below 1 we are behind.
KERNEL_OPERATING_POINTS = {
    "Serving blend, 64k context (measured 2026-08-07/08)": SERVING_BLEND_20260808["blend"],
    "Long-context prefill, 1M tokens (measured 2026-07-21)": 4.016653993568674,
    "Kernels at parity (x1)": 1.0,
    "Single-GPU training step, 2k context (measured 2026-08-14)": 1.0 / STEP_GAP_20260814["gap_against_battn"],
}
DEFAULT_KERNEL_POINT = "Serving blend, 64k context (measured 2026-08-07/08)"


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
        "chosen kernel operating point. Memory stays at the conservative /100 lever even "
        "though the measured serving-state ratio is far larger.",
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
    """param_matching_fraction() must reproduce the sealed receipt table."""
    worst = 0.0
    for n_tf, pct in PARAM_MATCHING_PCT_20260814.items():
        got = param_matching_fraction(n_tf) * 100.0
        worst = max(worst, abs(got - pct))
    assert worst < 1e-6, f"parameter-matching curve drifted from the receipt by {worst}"
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
    print("== serving state, MEASURED (transformer KV grows per token; ours does not) ==")
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
