"""AI_Capex_Efficiency — $ value of the Helarctos architecture for the six biggest AI spenders.

Layout:
  AUDIENCE LAYER (front, plain language):
  - Summary      : headline (FY2026 / FY2025), where the saving comes from (Training vs
                   Inference) with the Helarctos levers behind each fleet, by company, legend
  - Value Bridge : step by step from AI spend to the saving; levers one at a time; cross-check
                   against Totals; per-company engine tables (Python twin: value_bridge())
  - Levers       : scenario switch (C4); the six Helarctos levers H1-H6; how they combine;
                   the market & company data the front tabs use; per-company training shares
  - What Matters : one-at-a-time sensitivity, Helarctos levers vs market data, tornado chart
  - one tab per company: capex build from filings (FY2025 actual + FY2026)
  TECHNICAL APPENDIX: Totals, Inputs, Sensitivity, CostLadder, ServingTraining, Evidence,
  Methodology (the older memory/compute cost-split engine, ~x724 cost-weighted).

Every output is a LIVE FORMULA. Colours: teal + H1-H6 = Helarctos lever; yellow = market /
modelling assumption; green = disclosed data; blue = formula; ◆ purple = differs by scenario;
★ orange border = high-impact input. fit_rows() sizes every row so wrapped text is not clipped.
Run:  uv run --with openpyxl python ai_capex_efficiency.py
"""

from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side

from ai_capex_model import (  # single source of truth for defaults
    GLOBALS, COMPANIES, INFRA_SHARE_BASIS, TRAIN_SPEED_SAME_SIZE,
    CEILING_PREFILL_SPEEDUP,
    TODAY_FLOP_LEVER_20260901, CAMPAIGN_LANDED_FLOP_LEVER,
    CAMPAIGN_LANDED_20260831, DECK_DEPLOYMENT_SCALE,
    KERNEL_CAMPAIGN_20260824,
    KERNEL_SPEEDUP_REALIZED_20260824, KERNEL_SPEEDUP_REMAINING_TARGET,
    param_matching_gain, training_cost_saving,
    training_step_ratio, training_context_crossover, training_advantage_mix,
    training_helps_headline_threshold, headline_with_training,
    TRAINING_CURRICULA, KV_MB_PER_TOKEN_PER_STREAM,
    serving_context_sensitivity, compute_year,
    H5_CONVERSATIONS, H6_DECODE, PF_CURRENT, PF_OPTIMIZED, WORKLOAD,
    decode_tokens_per_s_per_gpu, prefill_tokens_per_s, fleet_memory_lever,
)

INPUT_FILL = PatternFill(
    "solid", fgColor="FFF2CC"
)  # yellow = ASSUMPTION (a lever / app slider)
DATA_FILL = PatternFill(
    "solid", fgColor="E2EFDA"
)  # green  = disclosed filing / market data
CALC_FILL = PatternFill("solid", fgColor="DDEBF7")  # blue   = derived formula
SCEN_FILL = PatternFill("solid", fgColor="E4DFEC")  # purple = differs by scenario (◆)
HEAD_FILL = PatternFill("solid", fgColor="1F4E78")
SUB_FILL = PatternFill("solid", fgColor="BDD7EE")
HLEV_FILL = PatternFill("solid", fgColor="A8E6D2")  # teal = a Helarctos lever (H1-H6)
HLEV_FONT = Font(bold=True, color="0B5E48")
BOLD = Font(bold=True)
WHITE_BOLD = Font(bold=True, color="FFFFFF")
THIN = Side(style="thin", color="BBBBBB")
BORDER = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)


def ctx_label(tokens):
    """Context length as people say it: 262,144 -> '262k', 8,192 -> '8k', 131,072 -> '128k'."""
    t = int(tokens)
    return "262k" if t == 262144 else f"{t // 1024}k"


def safe_text(v):
    """Notes that DESCRIBE a formula start with '= ' (equals+space); strip the
    prefix so Excel stores them as text. Real formulas start with '='+non-space
    (e.g. '=B2*B3') and pass through untouched."""
    if isinstance(v, str) and v.startswith("= "):
        return v[2:]
    return v


def header(ws, row, text, span=6):
    ws.cell(row=row, column=1, value=safe_text(scrub_external(text))).font = WHITE_BOLD
    for col in range(1, span + 1):
        ws.cell(row=row, column=col).fill = HEAD_FILL


# ---- external-content scrub (2026-09-01 pre-send review) ------------------------
# This workbook goes to outsiders. Content rule: the owned models are described
# only as "nonlinear internal dynamics with attractor states" — no mechanism
# terms (recurrent/RNN/scan), no kernel-stack internals, no internal cell names,
# no file paths or module/function names. Applied to EVERY string cell at write
# time so model-sourced provenance text cannot leak through.
import re as _re

_EXTERNAL_SCRUB = [
    (_re.compile(r"(?:/home/)?[\w~.-]+(?:/[\w~.()\[\]-]+)+\.(?:json|md|png|py|csv)"),
     "internal measurement archive"),
    (_re.compile(r"\b[\w-]+\.(?:json|md|png|py|csv)\b"), "internal measurement archive"),
    (_re.compile(r"\b(?:paper|experiments|bdm|src)/[\w./-]+"), "internal measurement archive"),
    (_re.compile(r"rows\[bdm"), "rows[bAttention"),
    (_re.compile(r"\bai_capex_model\.\w+"), "the model"),
    (_re.compile(r"\bSERVING\['\w+'\]"), "the model"),
    (_re.compile(r"\b\w+\(\)"), "the model"),
    (_re.compile(r"\b[A-Z0-9]+_[A-Z0-9_]+\b"), "the model"),  # internal constant names
    (_re.compile(r"\bTriton\s+"), ""),
    (_re.compile(r"\bstepped_vs_scanned\b"), "token-by-token ratio"),
    (_re.compile(r"\brecurrence\b", _re.I), "internal dynamics"),
    (_re.compile(r"\brecurrent\b", _re.I), "internal"),
    (_re.compile(r"\bRNN\b"), "our architecture"),
]


def scrub_external(s: str) -> str:
    for pat, rep in _EXTERNAL_SCRUB:
        s = pat.sub(rep, s)
    return s


def put(ws, r, c, val, fmt=None, fill=None, border=False, bold=False, wrap=False):
    if isinstance(val, str) and not val.startswith("="):
        val = scrub_external(val)
    cell = ws.cell(
        row=r, column=c, value=safe_text(val) if isinstance(val, str) else val
    )
    if fmt:
        cell.number_format = fmt
    if fill:
        cell.fill = fill
    if border:
        cell.border = BORDER
    if bold:
        cell.font = BOLD
    if wrap:
        cell.alignment = Alignment(wrap_text=True, vertical="top")
    return cell


def widths(ws, spec):
    for col, w in spec.items():
        ws.column_dimensions[col].width = w


# global Inputs addresses (fixed by the input order below)
MEMFAC, FLOPFAC, MEMSHARE = "Inputs!$B$2", "Inputs!$B$3", "Inputs!$B$4"
OPXRED, DISC, GPUCOST = "Inputs!$B$5", "Inputs!$B$6", "Inputs!$B$7"
PWR, ELEC, OH, LIFE = "Inputs!$B$8", "Inputs!$B$9", "Inputs!$B$10", "Inputs!$B$11"
MCAP, DCSCALE, RED = "Inputs!$B$12", "Inputs!$B$13", "Inputs!$B$20"

# uniform key cells on every company tab (col B = FY2025, col C = FY2026)
K_TOTAL25, K_TOTAL26 = "$B$3", "$C$3"
K_ACCEL25, K_ACCEL26 = "$B$12", "$C$12"
K_PCT25 = "$B$13"
K_AVOID25, K_AVOID26 = "$B$25", "$C$25"
K_CAP25, K_CAP26 = "$B$28", "$C$28"
K_AIREV, K_AICAPEX, K_AIOPEX = "$B$32", "$B$33", "$B$34"
K_AINOW, K_AICUT, K_AIARCH, K_AIPCT = "$B$35", "$B$36", "$B$37", "$B$38"


def build_inputs(inp):
    widths(inp, {"A": 50, "B": 13, "C": 7, "D": 22, "E": 74})
    header(
        inp,
        1,
        "INPUTS — teal = Helarctos lever (H1–H6) · yellow = market / modelling assumption · "
        "green = disclosed or market data · blue = formula",
        span=5,
    )

    G = GLOBALS  # values single-sourced from ai_capex_model; order fixes cell addresses B2..B14 (named-share = B14, used by the Totals global row)
    inputs = [
        (
            "Memory reduction factor",
            G["mem_factor"],
            "x",
            "MEASURED x2,032 at 262k on our test model (2026-08-14): 197 KB of transformer KV per token x 262,144 "
            "tokens = 51.6 GB vs a constant 25.4 MB Helarctos state. Caveat: for frontier-size models with "
            "grouped-query attention the frontier-geometry estimate is ~x208 at 262k, where memory would bind "
            "again (about -$0.5B on FY2026).",
            INPUT_FILL,
        ),
        (
            "FLOPs reduction factor",
            # LIVE: inference throughput per GPU from H5 x H6 and prompt speed (B27)
            "=$B$27",
            "x",
            "= B27 = inference throughput per GPU at a 262k-token average context, built live from H5 (B28), "
            "H6 (B29) and the active scenario's prompt-processing speed (B26). Inference cost does not scale "
            "with model size, so the equal-quality ratio (B23) applies to training only. Overtype this cell to "
            "pin the lever. See ServingTraining for the measured background.",
            CALC_FILL,
        ),
        (
            "Memory share of GPU cost (BOM)",
            G["mem_share"],
            "frac",
            "HBM ~45% of B200 COGS + most CoWoS packaging -> ~60% memory / ~40% compute. See Evidence.",
            INPUT_FILL,
        ),
        (
            "Opex / energy reduction factor",
            "=B20",
            "x",
            "DERIVED default = cost-weighted reduction (B20): energy splits memory/compute like cost. Overtype this cell to override.",
            INPUT_FILL,
        ),
        (
            "Discount rate",
            G["discount_rate"],
            "frac",
            "Perpetuity capitalization: value = annual benefit / rate (=16.7x at 6%).",
            INPUT_FILL,
        ),
        (
            "Fully-loaded cost per GPU",
            G["gpu_cost"],
            "$",
            "B200-class GPU + share of server, NVLink, networking.",
            INPUT_FILL,
        ),
        (
            "Wall power per GPU",
            G["wall_power_kw"],
            "kW",
            "GB200 NVL72: 120-132 kW per rack / 72 GPUs = ~1.7-1.8 kW IT per GPU (incl. CPUs, switches) x PUE ~1.3 = ~2.3 kW. Was 1.8 until 2026-09-29.",
            INPUT_FILL,
        ),
        (
            "Electricity rate",
            G["elec_rate"],
            "$/kWh",
            "Datacenter wholesale; raise to 0.10-0.12 for grid colo.",
            INPUT_FILL,
        ),
        (
            "Cooling/ops overhead on energy",
            G["cooling_overhead"],
            "frac",
            "Non-power running cost as fraction of electricity.",
            INPUT_FILL,
        ),
        (
            "Fleet useful life",
            G["fleet_life_yr"],
            "yr",
            "AI-GPU depreciation life.",
            INPUT_FILL,
        ),
        (
            "SpaceX market cap",
            G["spacex_mktcap"],
            "$B",
            "~$1.95T Aug 2026; IPO 2026-06-12 at ~$1.77T (market data; used by Sensitivity).",
            DATA_FILL,
        ),
        (
            "Datacenter scaling factor",
            G["dc_scale"],
            "frac",
            "TOGGLE 0-1. 0 = conservative (only accelerator silicon shrinks). 1 = whole AI datacenter (building/power/cooling/net) scales with the smaller fleet. ~0.7 ~= breakeven on net AI.",
            INPUT_FILL,
        ),
        (
            "Named share of global AI capex",
            G["named_share_of_global"],
            "frac",
            "GLOBAL estimate (Totals): named firms' share of worldwide AI capex; the rest (other clouds, China, neoclouds, xAI, sovereign) is grossed up pro-rata. ESTIMATE.",
            INPUT_FILL,
        ),
    ]
    r = 2
    KEY_INPUTS = {"Discount rate"}  # high impact (What Matters tab); H4 is not (its swing is < $10B)
    LEVER_ROWS = {"Memory reduction factor": "H4", "FLOPs reduction factor": "H5·H6"}
    KIND = {"Memory share of GPU cost (BOM)": "Market data (teardowns)", "Opex / energy reduction factor": "Derived",
            "Fully-loaded cost per GPU": "Market data", "Wall power per GPU": "Market data",
            "Electricity rate": "Market data", "SpaceX market cap": "Market data"}
    for label, val, unit, note, fill in inputs:
        code = LEVER_ROWS.get(label)
        shown = (f"{code} · " if code else "") + label + (" ★" if label in KEY_INPUTS else "")
        lab = put(inp, r, 1, shown)
        c = put(
            inp,
            r,
            2,
            val,
            fmt=("0.0%" if unit == "frac" else "#,##0" if unit == "$" else "0.0"),
            fill=(HLEV_FILL if code and fill is INPUT_FILL else fill),
            border=True,
        )
        put(inp, r, 3, unit)
        kind = (f"Helarctos lever{'s' if '·' in code else ''} {code}" if code
                else KIND.get(label, "Modelling assumption"))
        put(inp, r, 4, kind)
        if code:
            lever_cell(lab)
            lever_cell(inp.cell(row=r, column=4))
        put(inp, r, 5, note, wrap=True)
        if label in KEY_INPUTS:
            key_cell(c)
        r += 1
    header(inp, 15, "REDUCTION ENGINE (Amdahl cost-weighting) — derived; used by the technical tabs", span=5)
    put(inp, 16, 1, "Compute share of GPU cost")
    put(inp, 16, 2, "=1-B4", fmt="0.0%", fill=CALC_FILL, border=True)
    put(inp, 16, 4, "Derived")
    put(inp, 16, 5, "1 - memory share", wrap=True)
    put(inp, 17, 1, "Memory cost fraction after reduction")
    put(inp, 17, 2, "=B4/B2", fmt="0.00%", fill=CALC_FILL, border=True)
    put(inp, 17, 4, "Derived")
    put(inp, 17, 5, "memory share / memory reduction", wrap=True)
    put(inp, 18, 1, "Compute cost fraction after reduction")
    put(inp, 18, 2, "=B16/B3", fmt="0.00%", fill=CALC_FILL, border=True)
    put(inp, 18, 4, "Derived")
    put(inp, 18, 5, "compute share / FLOPs reduction", wrap=True)
    put(inp, 19, 1, "Residual cost fraction")
    put(inp, 19, 2, "=B17+B18", fmt="0.0%", fill=CALC_FILL, border=True)
    put(inp, 19, 4, "Derived")
    put(inp, 19, 5, "Amdahl: floored by the least-reduced component", wrap=True)
    put(inp, 20, 1, "COST-WEIGHTED reduction factor", bold=True)
    put(inp, 20, 2, "=1/B19", fmt="0.0", fill=CALC_FILL, border=True, bold=True)
    put(inp, 20, 3, "x")
    put(inp, 20, 4, "Derived")
    put(inp, 20, 5, "1 / residual. The $ reduction used by the technical tabs (Totals, company tabs).", wrap=True)
    header(inp, 22, "HELARCTOS LEVERS H1, H5, H6 + prompt-processing speed — 262k-token average context; "
                    "◆ purple = differs by scenario", span=5)
    lever_cell(put(inp, 23, 1, "H1 · Equal-quality parameter ratio at ~1T dense-equivalent (smaller model; H2 = same) ★"))
    put(inp, 23, 2, 4.2185, fmt="0.00", fill=HLEV_FILL, border=True)
    key_cell(inp["B23"])
    put(inp, 23, 3, "x")
    lever_cell(put(inp, 23, 4, "Helarctos lever H1"))
    put(inp, 23, 5,
        "FIT-DERIVED (sealed 2026-08-14 refit): bAttention matches the transformer fit's quality on "
        "23.7% of the params at 1T -> x4.22 compute per token at equal quality. 84.2% at 1B, 55.2% at "
        "10B, 36.2% at 100B, 15.5% at 10T (x6.44). A projection of the two fits, not a measurement. "
        "Same in both scenarios. Scale: ~1T dense-equivalent: about what today's 5-6T-total mixture-of-experts flagships amount to (Grok 5 at 6T, Kimi K3 at 2.8T; a mixture-of-experts model runs only a fraction of its parameters per token and performs roughly like a dense model of sqrt(total x active) size).", wrap=True)
    c = CAMPAIGN_LANDED_20260831
    _tp = {int(d["context_tokens"]): d["today_lever"] / param_matching_gain(DECK_DEPLOYMENT_SCALE)
           for d in serving_context_sensitivity((8192, 32768, 131072))}
    pf_rows = [
        (24, "Prompt-processing speed vs the transformer — Current kernels ◆", PF_CURRENT,
         f"Supporting input (a Helarctos kernel property, not a headline lever): our prefill tokens/s at 262k "
         f"with the banked x{KERNEL_SPEEDUP_REALIZED_20260824:.3f} kernel speed-up (MEASURED) / the "
         f"transformer's. The only inference input that differs by scenario."),
        (25, "Prompt-processing speed vs the transformer — Optimized kernels ◆", PF_OPTIMIZED,
         f"Same at the full x{CEILING_PREFILL_SPEEDUP:.2f} = x{KERNEL_SPEEDUP_REALIZED_20260824:.3f} MEASURED x "
         f"x{KERNEL_SPEEDUP_REMAINING_TARGET:.3f} TARGET (the funded kernel programme). Prompts are only ~0.2% "
         f"of the transformer's time at 262k, so this barely moves the lever."),
    ]
    for rr, lab, val, note in pf_rows:
        put(inp, rr, 1, lab)
        put(inp, rr, 2, val, fmt="0.0", fill=SCEN_FILL, border=True)
        put(inp, rr, 3, "x")
        put(inp, rr, 4, "Helarctos kernels (supporting)")
        put(inp, rr, 5, note, wrap=True)
    put(inp, 26, 1, "Prompt-processing speed — ACTIVE scenario ◆")
    put(inp, 26, 2, '=IF(Scenario="Optimized kernels",$B$25,$B$24)', fmt="0.0", fill=SCEN_FILL, border=True)
    put(inp, 26, 3, "x")
    put(inp, 26, 4, "Derived")
    put(inp, 26, 5, "Follows the scenario picked on the Levers tab (cell C4).", wrap=True)
    lever_cell(put(inp, 27, 1, "H5 × H6 → inference compute (FLOPs) lever used"))
    put(inp, 27, 2, "=(1+$B$32)/($B$32/$B$26+1/($B$28*$B$29))", fmt="0.0", fill=CALC_FILL, border=True, bold=True)
    put(inp, 27, 3, "x")
    put(inp, 27, 4, "Derived")
    put(inp, 27, 5,
        f"Tokens served per GPU vs the transformer at the same model size = (1 + B32) / (B32/B26 + 1/(B28 x B29)): "
        f"decode at H5 x H6, plus prompt processing at B26. B3 points here. Context matters: ~x{_tp[8192]:.0f} "
        f"at an 8k-token average context, ~x{_tp[32768]:.0f} at 32k, ~x{_tp[131072]:.0f} at 128k.", wrap=True)
    lever_cell(put(inp, 28, 1, "H5 · More conversations per GPU"))
    put(inp, 28, 2, H5_CONVERSATIONS, fmt="0", fill=HLEV_FILL, border=True)
    put(inp, 28, 3, "x")
    lever_cell(put(inp, 28, 4, "Helarctos lever H5"))
    put(inp, 28, 5,
        f"MEASURED 2026-08-14: {c['streams_per_gpu']} resident 262k-token conversations on one GPU in 1.6 GB of "
        f"state (a grid cap, not a ceiling) vs the transformer's 1 — a 2nd OOMs. Enabled by the small memory (H4).",
        wrap=True)
    lever_cell(put(inp, 29, 1, "H6 · Faster decode per token"))
    put(inp, 29, 2, H6_DECODE, fmt="0.00", fill=HLEV_FILL, border=True)
    put(inp, 29, 3, "x")
    lever_cell(put(inp, 29, 4, "Helarctos lever H6"))
    put(inp, 29, 5,
        f"ESTIMATE: transformer {1000 / decode_tokens_per_s_per_gpu('transformer', 262144):.2f} ms per token at "
        f"262k / our {c['decode_own_ms_per_token']:.1f} ms (post-kernel-edit estimate; aggregate-decode receipt "
        f"PENDING). On our older, unoptimised kernels we MEASURED ~5.7 vs 14.9 ms GPU-busy = x2.6.", wrap=True)
    put(inp, 30, 1, "Input : output tokens per conversation")
    put(inp, 30, 2, float(WORKLOAD["in_out_ratio"]), fmt="0.0", fill=INPUT_FILL, border=True)
    put(inp, 30, 4, "Modelling assumption")
    put(inp, 30, 5, "~10:1 = code / reasoning / agent traces.", wrap=True)
    put(inp, 31, 1, "Transformer decode ÷ prompt speed at 262k")
    put(inp, 31, 2, decode_tokens_per_s_per_gpu("transformer", 262144) / prefill_tokens_per_s("transformer", 262144),
        fmt="0.000000", fill=DATA_FILL, border=True)
    put(inp, 31, 4, "Measured (transformer)")
    put(inp, 31, 5, "Tokens/s generated ÷ prompt tokens/s processed, per GPU (measured transformer rates).", wrap=True)
    put(inp, 32, 1, "Transformer prompt time ÷ decode time")
    put(inp, 32, 2, "=$B$30*$B$31", fmt="0.00%", fill=CALC_FILL, border=True)
    put(inp, 32, 4, "Derived")
    put(inp, 32, 5, "= B30 x B31: the transformer spends ~0.23% of its time on the prompt at 262k.", wrap=True)


def build_company(ws, name, c25, c26, mcap, ai_rev, basis, sources=()):
    """c25/c26 = (total_capex, infra_share, server_share, accel_share) for FY25/FY26.
    ai_rev = (revenue_FY25, revenue_FY26[, revenue_FY27_est -- ignored here,
    the workbook stays a FY25+FY26 build]). sources = [(label, url), ...]."""
    widths(ws, {"A": 38, "B": 13, "C": 13, "D": 62})
    _title(ws, f"{name.upper()} — AI capex from filings, and what Helarctos saves", 4)
    _thead(ws, 2, ["Company data (green = disclosed, yellow = estimate)", "FY2025", "FY2026", "Basis / source"],
           height=30)
    t25, i25, s25, a25 = c25
    t26, i26, s26, a26 = c26
    # FY2025 total capex is DISCLOSED (green); FY2026 is an estimate; shares are assumptions (yellow)
    inrows = [
        (3, "Total capex ($B)", t25, t26, "#,##0.0", basis, DATA_FILL, INPUT_FILL),
        (
            4,
            "Infra / data-center share",
            i25,
            i26,
            "0%",
            INFRA_SHARE_BASIS.get(name, "strips non-datacenter capex"),
            INPUT_FILL,
            INPUT_FILL,
        ),
        (
            5,
            "Server / short-lived share",
            s25,
            s26,
            "0%",
            "CFO-disclosed (MSFT rose 50%->67%)",
            INPUT_FILL,
            INPUT_FILL,
        ),
        (
            6,
            "Accelerator share within servers",
            a25,
            a26,
            "0%",
            "from BOM teardowns (~67-80%)",
            INPUT_FILL,
            INPUT_FILL,
        ),
    ]
    for r, lab, v25, v26, fmt, note, f25, f26 in inrows:
        key = r in (4, 5, 6)  # data-center / server / accelerator shares: high impact (What Matters tab)
        put(ws, r, 1, lab + (" ★" if key else ""))
        put(ws, r, 2, v25, fmt=fmt, fill=f25, border=True)
        put(ws, r, 3, v26, fmt=fmt, fill=f26, border=True)
        put(ws, r, 4, note, wrap=True)
        if key:
            key_cell(ws.cell(row=r, column=2))
            key_cell(ws.cell(row=r, column=3))
    put(ws, 7, 1, "Market cap ($B, approx)")
    put(ws, 7, 2, mcap, fmt="#,##0", fill=DATA_FILL, border=True)
    put(ws, 7, 4, "approximate June 2026 market data -- edit", wrap=True)

    header(ws, 9, "DERIVATION — how much of the capex buys AI chips", span=4)
    der = [
        (
            10,
            "AI-infra capex ($B)",
            "=B3*B4",
            "=C3*C4",
            "#,##0.0",
            "= total x infra share",
            False,
        ),
        (
            11,
            "Server bucket ($B)",
            "=B10*B5",
            "=C10*C5",
            "#,##0.0",
            "= infra x server share",
            False,
        ),
        (
            12,
            "ACCELERATOR capex ($B)",
            "=B11*B6",
            "=C11*C6",
            "#,##0.0",
            "= server x accel share",
            True,
        ),
        (
            13,
            "Accel % of total capex",
            "=B12/B3",
            "=C12/C3",
            "0%",
            "varies across companies",
            False,
        ),
    ]
    for r, lab, bf, cf, fmt, note, bold in der:
        put(ws, r, 1, lab, bold=bold)
        put(ws, r, 2, bf, fmt=fmt, fill=CALC_FILL, border=True, bold=bold)
        put(ws, r, 3, cf, fmt=fmt, fill=CALC_FILL, border=True, bold=bold)
        put(ws, r, 4, note, wrap=True)

    header(ws, 15, "FLEET & POWER (from AI-chip capex)", span=4)
    fl = [
        (
            16,
            "Fleet size (GPU-equiv)",
            f"=B12*1000000000/{GPUCOST}",
            f"=C12*1000000000/{GPUCOST}",
            "#,##0",
            "= accel capex / $ per GPU",
        ),
        (
            17,
            "Total wall power (MW)",
            f"=B16*{PWR}/1000",
            f"=C16*{PWR}/1000",
            "#,##0",
            "= GPUs x kW",
        ),
        (18, "Daily energy (MWh)", "=B17*24", "=C17*24", "#,##0", ""),
        (
            19,
            "Daily all-in opex ($/day)",
            f"=B18*1000*{ELEC}*(1+{OH})",
            f"=C18*1000*{ELEC}*(1+{OH})",
            "#,##0",
            "= MWh x rate x (1+overhead)",
        ),
        (20, "Annual opex ($M)", "=B19*365/1000000", "=C19*365/1000000", "#,##0.0", ""),
        (
            21,
            "Lifetime opex ($B)",
            f"=B20*{LIFE}/1000",
            f"=C20*{LIFE}/1000",
            "0.00",
            "",
        ),
    ]
    for r, lab, bf, cf, fmt, note in fl:
        put(ws, r, 1, lab)
        put(ws, r, 2, bf, fmt=fmt, fill=CALC_FILL, border=True)
        put(ws, r, 3, cf, fmt=fmt, fill=CALC_FILL, border=True)
        put(ws, r, 4, note, wrap=True)

    header(ws, 23, "HELARCTOS EFFECT — technical engine (memory and compute priced separately)", span=4)
    va = [
        (
            24,
            "Efficient AI capex ($B)",
            "=B10-B25",
            "=C10-C25",
            "#,##0.0",
            "= AI capex - avoided",
            False,
        ),
        (
            25,
            "Capex avoided/yr ($B)",
            f"=(B12+(B10-B12)*{DCSCALE})*(1-1/{RED})",
            f"=(C12+(C10-C12)*{DCSCALE})*(1-1/{RED})",
            "#,##0.0",
            "Helarctos enters here: (chips + data-centre x dc-scale) x (1 - 1/cost-weighted cut, Inputs B20, built from H4 memory and H5 compute)",
            False,
        ),
        (
            26,
            "Annual opex savings ($M)",
            f"=B20*(1-1/{OPXRED})",
            f"=C20*(1-1/{OPXRED})",
            "#,##0.0",
            "Helarctos enters here: power shrinks with the fleet (Inputs B5 = B20)",
            False,
        ),
        (
            27,
            "Sustained annual benefit ($B/yr)",
            "=B25+B26/1000",
            "=C25+C26/1000",
            "#,##0.00",
            "= avoided + opex savings",
            False,
        ),
        (
            28,
            "Capitalized value ($B)",
            f"=B27/{DISC}",
            f"=C27/{DISC}",
            "#,##0",
            "= benefit / discount rate",
            True,
        ),
        (29, "% of market cap", "=B28/$B$7", "=C28/$B$7", "0.0%", "", False),
    ]
    for r, lab, bf, cf, fmt, note, bold in va:
        c = put(ws, r, 1, lab, bold=bold)
        if r in (25, 26):
            lever_cell(c)
            c.value = "H4·H5 → " + lab
        put(ws, r, 2, bf, fmt=fmt, fill=CALC_FILL, border=True, bold=bold)
        put(ws, r, 3, cf, fmt=fmt, fill=CALC_FILL, border=True, bold=bold)
        put(ws, r, 4, note, wrap=True)

    header(ws, 31, "AI ECONOMICS — cash basis: AI revenue - AI capex - AI power", span=4)
    rev25, rev26 = ai_rev[0], ai_rev[1]  # fy27 element (charts-only) ignored
    put(ws, 32, 1, "AI revenue ($B)")
    put(ws, 32, 2, rev25, fmt="#,##0.0", fill=INPUT_FILL, border=True)
    put(ws, 32, 3, rev26, fmt="#,##0.0", fill=INPUT_FILL, border=True)
    put(
        ws,
        32,
        4,
        "ESTIMATE (see Methodology). MSFT $37B & Amazon $15B run-rates disclosed; Google/Meta/SpaceX estimated.",
        wrap=True,
    )
    aer = [
        (
            33,
            "AI capex ($B)",
            "=B10",
            "=C10",
            "#,##0.0",
            "= AI-infra capex (full: accel + buildings + power + net)",
            False,
        ),
        (
            34,
            "AI opex ($B)",
            "=B20/1000",
            "=C20/1000",
            "#,##0.0",
            "= annual power/operating",
            False,
        ),
        (
            35,
            "Net AI NOW ($B)",
            "=B32-B33-B34",
            "=C32-C33-C34",
            "#,##0.0",
            "revenue - capex - opex (cash burn)",
            True,
        ),
        (
            36,
            "Spend cut with Helarctos ($B)",
            "=B27",
            "=C27",
            "#,##0.0",
            "= accel capex avoided + opex saved",
            False,
        ),
        (
            37,
            "Net AI with Helarctos ($B)",
            "=B35+B36",
            "=C35+C36",
            "#,##0.0",
            "= Net AI now + spend cut",
            True,
        ),
        (
            38,
            "% AI spend reduction",
            "=B36/(B33+B34)",
            "=C36/(C33+C34)",
            "0%",
            "spend cut / total AI spend (accel-only; upside if DC scales)",
            False,
        ),
    ]
    for r, lab, bf, cf, fmt, note, bold in aer:
        put(ws, r, 1, lab, bold=bold)
        put(ws, r, 2, bf, fmt=fmt, fill=CALC_FILL, border=True, bold=bold)
        put(ws, r, 3, cf, fmt=fmt, fill=CALC_FILL, border=True, bold=bold)
        put(ws, r, 4, note, wrap=True)

    if sources:
        header(ws, 40, "SOURCES & REFERENCES", span=4)
        for i, (label, url) in enumerate(sources):
            r = 41 + i
            put(ws, r, 1, label, wrap=True)
            link = put(ws, r, 2, url, wrap=True)
            link.hyperlink = url
            link.font = Font(color="0563C1", underline="single")
            ws.merge_cells(start_row=r, start_column=2, end_row=r, end_column=4)


def build_totals(tot, tabs):
    widths(tot, {"A": 17, "B": 12, "C": 12, "D": 11, "E": 12, "F": 13, "G": 13, "H": 9})
    header(
        tot,
        1,
        f"{len(tabs)}-COMPANY AI ECONOMICS — current AI cash burn & the effect of our architecture",
        span=8,
    )
    put(
        tot,
        2,
        1,
        "They all spend far more on AI (capex + opex) than they earn from it -- they are losing money on AI today. Helarctos cuts the AI-chip spend ~99%, shrinking the burn. The named firms are a floor; a GLOBAL estimate grosses up for the rest. Yellow assumption cells (also sliders in the app) and green disclosed-data cells live on company tabs.",
        wrap=True,
    )
    tot.merge_cells("A2:H2")

    header(
        tot,
        4,
        "NET AI ECONOMICS — FY2025 (cash basis: AI revenue - AI capex - AI opex)",
        span=8,
    )
    heads = [
        "Company",
        "AI revenue ($B)",
        "AI capex ($B)",
        "AI opex ($B)",
        "Net AI NOW ($B)",
        "Spend cut with Helarctos ($B)",
        "Net AI with Helarctos ($B)",
        "% spend cut",
    ]
    for j, h in enumerate(heads):
        put(tot, 5, 1 + j, h, bold=True, wrap=True)
    r0 = 6
    for k, t in enumerate(tabs):
        r = r0 + k
        put(tot, r, 1, t)
        put(tot, r, 2, f"={t}!{K_AIREV}", fmt="#,##0.0", fill=CALC_FILL, border=True)
        put(tot, r, 3, f"={t}!{K_AICAPEX}", fmt="#,##0.0", fill=CALC_FILL, border=True)
        put(tot, r, 4, f"={t}!{K_AIOPEX}", fmt="#,##0.0", fill=CALC_FILL, border=True)
        put(
            tot,
            r,
            5,
            f"={t}!{K_AINOW}",
            fmt="#,##0.0",
            fill=CALC_FILL,
            border=True,
            bold=True,
        )
        put(tot, r, 6, f"={t}!{K_AICUT}", fmt="#,##0.0", fill=CALC_FILL, border=True)
        put(
            tot,
            r,
            7,
            f"={t}!{K_AIARCH}",
            fmt="#,##0.0",
            fill=CALC_FILL,
            border=True,
            bold=True,
        )
        put(tot, r, 8, f"={t}!{K_AIPCT}", fmt="0%", fill=CALC_FILL, border=True)
    rt = r0 + len(tabs)
    put(tot, rt, 1, f"TOTAL ({len(tabs)})", bold=True)
    for col, L in ((2, "B"), (3, "C"), (4, "D"), (5, "E"), (6, "F"), (7, "G")):
        put(
            tot,
            rt,
            col,
            f"=SUM({L}{r0}:{L}{rt - 1})",
            fmt="#,##0",
            bold=True,
            fill=SUB_FILL,
            border=True,
        )
    put(
        tot,
        rt,
        8,
        f"=F{rt}/(C{rt}+D{rt})",
        fmt="0%",
        bold=True,
        fill=SUB_FILL,
        border=True,
    )
    fr = rt + 1
    nn26 = "SUM(" + ",".join(f"{t}!$C$35" for t in tabs) + ")"
    cut26 = "SUM(" + ",".join(f"{t}!$C$36" for t in tabs) + ")"
    arch26 = "SUM(" + ",".join(f"{t}!$C$37" for t in tabs) + ")"
    put(tot, fr, 1, "FY2026 (estimate)", bold=True)
    put(tot, fr, 5, f"={nn26}", fmt="#,##0", border=True, fill=SUB_FILL)
    put(tot, fr, 6, f"={cut26}", fmt="#,##0", border=True, fill=SUB_FILL)
    put(tot, fr, 7, f"={arch26}", fmt="#,##0", border=True, fill=SUB_FILL)
    capex26 = "SUM(" + ",".join(f"{t}!$C$33" for t in tabs) + ")"
    opex26 = "SUM(" + ",".join(f"{t}!$C$34" for t in tabs) + ")"
    put(tot, fr, 8, f"={cut26}/({capex26}+{opex26})", fmt="0%", border=True, fill=SUB_FILL)
    # FY27 prediction row (2026-09-01 user request): STATIC values at model
    # defaults (compute_year on the fy27 street-estimate capex — Morgan Stanley
    # +57% path; Oracle's ~$70B is the only real FY27 guide). Company tabs carry
    # no fy27 formula chain, so this row does not recompute with the levers —
    # rebuild the workbook to refresh.
    fr27 = fr + 1
    _, _tot27 = compute_year(GLOBALS, COMPANIES, "fy27")
    put(tot, fr27, 1, "FY2027 (STREET-EST prediction, static)", bold=True)
    put(tot, fr27, 5, _tot27["net_now"], fmt="#,##0", border=True, fill=SUB_FILL)
    put(tot, fr27, 6, _tot27["spend_cut"], fmt="#,##0", border=True, fill=SUB_FILL)
    put(tot, fr27, 7, _tot27["net_arch"], fmt="#,##0", border=True, fill=SUB_FILL)
    put(tot, fr27, 8, _tot27["pct_cut"], fmt="0%", border=True, fill=SUB_FILL)

    sb = fr27 + 2
    header(tot, sb, "SAVINGS BREAKDOWN — the spend cut, split", span=8)
    put(tot, sb + 1, 1, "Cost-weighted reduction (Amdahl)")
    put(tot, sb + 1, 2, f"={RED}", fmt="0.0", fill=CALC_FILL, border=True, bold=True)
    put(tot, sb + 1, 3, "x")
    put(tot, sb + 3, 2, "Saved OPEX", bold=True)
    put(tot, sb + 3, 3, "Avoided CAPEX (Overspend)", bold=True, wrap=True)
    put(tot, sb + 3, 4, "Total", bold=True)
    opex25 = "SUM(" + ",".join(f"{t}!B26" for t in tabs) + ")/1000"
    capex25 = "SUM(" + ",".join(f"{t}!B25" for t in tabs) + ")"
    opex26 = "SUM(" + ",".join(f"{t}!C26" for t in tabs) + ")/1000"
    capex26 = "SUM(" + ",".join(f"{t}!C25" for t in tabs) + ")"
    vrows = [
        ("FY2025 annual ($B/yr)", f"={opex25}", f"={capex25}", "#,##0.0"),
        (
            "FY2025 capitalized ($B)",
            f"=B{sb + 4}/{DISC}",
            f"=C{sb + 4}/{DISC}",
            "#,##0",
        ),
        ("FY2026 annual ($B/yr)", f"={opex26}", f"={capex26}", "#,##0.0"),
        (
            "FY2026 capitalized ($B)",
            f"=B{sb + 6}/{DISC}",
            f"=C{sb + 6}/{DISC}",
            "#,##0",
        ),
    ]
    for k, (lab, bf, cf, fmt) in enumerate(vrows):
        r = sb + 4 + k
        bold = "capitalized" in lab
        put(tot, r, 1, lab, bold=bold)
        put(tot, r, 2, bf, fmt=fmt, fill=CALC_FILL, border=True, bold=bold)
        put(tot, r, 3, cf, fmt=fmt, fill=CALC_FILL, border=True, bold=bold)
        put(tot, r, 4, f"=B{r}+C{r}", fmt=fmt, fill=CALC_FILL, border=True, bold=bold)
    gr = sb + 8
    put(tot, gr, 1, "GLOBAL capitalized -- est. ($B)", bold=True)
    put(
        tot,
        gr,
        2,
        f"=D{sb + 5}/Inputs!$B$14",
        fmt="#,##0",
        fill=CALC_FILL,
        border=True,
        bold=True,
    )
    put(
        tot,
        gr,
        3,
        f"=D{sb + 7}/Inputs!$B$14",
        fmt="#,##0",
        fill=CALC_FILL,
        border=True,
        bold=True,
    )
    put(
        tot,
        gr,
        4,
        "named-floor capitalized / named-share-of-global (Inputs B14). ESTIMATE: grosses up for other clouds, China, neoclouds, xAI, sovereign.",
        wrap=True,
    )
    pr = sb + 9
    put(tot, pr, 1, "% reduction")
    put(tot, pr, 2, f"=1-1/{OPXRED}", fmt="0%", fill=CALC_FILL, border=True)
    put(tot, pr, 3, f"=1-1/{RED}", fmt="0%", fill=CALC_FILL, border=True)
    note_r = pr + 1
    put(
        tot,
        note_r,
        1,
        "OPEX = power saved each year (recoupable). CAPEX 'Overspend' = AI capex made unnecessary. TOGGLE: Inputs 'Datacenter scaling factor' (0 = accelerator-only/conservative; 1 = whole datacenter scales with the smaller fleet). At 0 the burn ~halves; at ~0.7 net AI hits breakeven; at 1 it flips positive. Capitalized = annual / discount rate.",
        wrap=True,
    )
    tot.merge_cells(start_row=note_r, start_column=1, end_row=note_r, end_column=8)

    gb = note_r + 2
    header(tot, gb, "TABS", span=8)
    guide = [
        ("Summary / Value Bridge / Levers / What Matters",
         "the plain-language front tabs: headline, training vs inference split, the Helarctos levers H1–H6, what moves the answer"),
        (
            " / ".join(tabs),
            "one tab each: full build capex -> infra -> servers -> accelerator -> fleet/opex -> value",
        ),
        ("Inputs", "global assumptions + the reduction engine (cost-weighted factor)"),
        (
            "Sensitivity",
            "SpaceX value under both pricing methods + how the compute lever varies with context",
        ),
        ("CostLadder", "own-silicon vs buy-NVIDIA vs rent $/GPU-hr"),
        (
            "ServingTraining",
            "the serving/training measurement ledger behind the levers — status-tagged (MEASURED / ESTIMATE / TARGET)",
        ),
        (
            "Evidence",
            "BOM cost split, accelerator-share data, own-silicon TCO, filing top-lines",
        ),
        ("Methodology", "step-by-step logic, caveats, sources"),
    ]
    for k, (n, d) in enumerate(guide):
        r = gb + 1 + k
        put(tot, r, 1, n, bold=True)
        put(tot, r, 2, d, wrap=True)
        tot.merge_cells(start_row=r, start_column=2, end_row=r, end_column=8)
    lr = gb + 1 + len(guide) + 1
    put(
        tot,
        lr,
        1,
        "Legend: teal = Helarctos lever (H1–H6)  ·  yellow = market / modelling assumption  ·  green = disclosed filing or market data  ·  blue = formula  ·  ◆ purple = differs by scenario  ·  ★ = high impact.",
        bold=True,
    )
    tot.merge_cells(start_row=lr, start_column=1, end_row=lr, end_column=8)


def build_sensitivity(sens):
    widths(sens, {"A": 30, "B": 16, "C": 16, "D": 16, "E": 18, "F": 18})
    header(
        sens,
        1,
        "SENSITIVITY (SpaceX) — memory/compute cost split vs whole-GPU pricing",
        span=5,
    )
    # base = ACCELERATOR capex (matches the SpaceX tab, which reduces only accelerator silicon
    # at the conservative dc_scale=0; NOT total capex, which would overstate the avoided spend)
    SXCAP, SXOPX = "SpaceX!$B$12", "SpaceX!$B$26"
    cols = [
        ("Memory/compute cost split (Totals engine)", RED),
        ("Whole GPUs (Value Bridge logic)", "MIN(Inputs!$B$2,Inputs!$B$3)"),
    ]
    put(sens, 2, 1, "Metric", bold=True)
    for j, (name, _) in enumerate(cols):
        put(sens, 2, 2 + j, name, bold=True, wrap=True)

    def srow(rr, lab, fmt, make):
        put(sens, rr, 1, lab)
        for j, (_, e) in enumerate(cols):
            put(sens, rr, 2 + j, make(e), fmt=fmt, fill=CALC_FILL, border=True)

    srow(3, "Efficient acquisition ($B)", "0.00", lambda e: f"={SXCAP}/({e})")
    srow(4, "Capex avoided/yr ($B)", "0.00", lambda e: f"={SXCAP}-{SXCAP}/({e})")
    srow(5, "Annual opex savings ($M)", "#,##0.0", lambda e: f"={SXOPX}")
    srow(
        6,
        "Sustained annual benefit ($B)",
        "0.00",
        lambda e: f"=({SXCAP}-{SXCAP}/({e}))+{SXOPX}/1000",
    )
    srow(
        7,
        "Capitalized value ($B)",
        "#,##0",
        lambda e: f"=(({SXCAP}-{SXCAP}/({e}))+{SXOPX}/1000)/{DISC}",
    )
    srow(
        8,
        "% of market cap",
        "0.0%",
        lambda e: f"=((({SXCAP}-{SXCAP}/({e}))+{SXOPX}/1000)/{DISC})/{MCAP}",
    )
    put(
        sens,
        10,
        1,
        "Base = SpaceX accelerator capex (SpaceX!B12), not total capex. The first column ties to the "
        "SpaceX tab (B25/B27) at the conservative dc_scale=0; the second prices the fleet on whole GPUs "
        "(shrinks by the smaller of the memory and compute levers).",
        wrap=True,
    )
    sens.merge_cells("A10:E10")

    # ---- context sensitivity (2026-09-01 user request) --------------------------
    # Static values computed by ai_capex_model.serving_context_sensitivity() at
    # build time (the levers need the model's prefill/decode interpolators, which
    # have no in-sheet formula equivalent).
    header(sens, 12, "CONVERSATION-LENGTH SENSITIVITY — prompt share of serving cost vs average conversation length", span=6)
    ctx_cols = ["Conversation length", "TF prefill share", "Our prefill share",
                "◆ Inference lever, current kernels (x)", "◆ Inference lever, optimized kernels (x)", ""]
    for j, name in enumerate(ctx_cols):
        put(sens, 13, 1 + j, name, bold=True, wrap=True)
    for i, cs in enumerate(serving_context_sensitivity()):
        rr = 14 + i
        lbl = ctx_label(cs["context_tokens"]) + (" (used)" if cs["pinned"] else "")
        put(sens, rr, 1, lbl, bold=cs["pinned"])
        put(sens, rr, 2, cs["tf_prefill_share"], fmt="0.00%", fill=CALC_FILL, border=True)
        put(sens, rr, 3, cs["own_prefill_share_today"], fmt="0.0%", fill=CALC_FILL, border=True)
        put(sens, rr, 4, cs["today_lever"] / param_matching_gain(DECK_DEPLOYMENT_SCALE), fmt="#,##0.0",
            fill=CALC_FILL, border=True)
        put(sens, rr, 5, cs["ceiling_lever"] / param_matching_gain(DECK_DEPLOYMENT_SCALE), fmt="#,##0.0",
            fill=CALC_FILL, border=True)
    put(
        sens,
        19,
        1,
        "Cost = box wall-clock, not FLOPs. The transformer's decode leg is KV-bandwidth-bound "
        "(measured 1/context law) while its prefill runs near peak, so prefill is <1% of its "
        "serving cost at every context, and the lever is set by decode. Inputs!B24 quotes the pinned "
        "262k row (2026-09-29 ruling). Static values computed by the model — rebuild the workbook to "
        "refresh.",
        wrap=True,
    )
    sens.merge_cells("A19:F19")


def build_ladder(ladder):
    widths(ladder, {"A": 34, "B": 11, "C": 11, "D": 62})
    header(
        ladder,
        1,
        "COST LADDER — $/H100-equivalent GPU-hour (at scale). Green = market price; yellow = assumption.",
        span=4,
    )
    put(ladder, 2, 1, "Procurement mode", bold=True)
    put(ladder, 2, 2, "$/hr low", bold=True)
    put(ladder, 2, 3, "$/hr high", bold=True)
    put(ladder, 2, 4, "What's baked into the price", bold=True)
    lad = [
        (
            "Own custom silicon (TPU/Trainium)",
            0.9,
            1.4,
            "COGS + modest Broadcom/Marvell margin + power + DC. NO NVIDIA margin.",
        ),
        (
            "Buy + operate NVIDIA (scale)",
            1.5,
            2.0,
            "NVIDIA ~84% gross margin baked into capex + power + DC.",
        ),
        (
            "Rent NVIDIA - neocloud / committed",
            2.0,
            3.5,
            "+ cloud provider capex recovery & margin.",
        ),
        (
            "Rent NVIDIA - hyperscaler on-demand",
            3.0,
            7.0,
            "+ utilization risk + flexibility premium (new B200 to ~$14).",
        ),
    ]
    for k, (m, lo, hi, note) in enumerate(lad):
        rr = 3 + k
        put(ladder, rr, 1, m)
        put(
            ladder, rr, 2, lo, fmt="0.00", fill=DATA_FILL, border=True
        )  # market-observed price
        put(ladder, rr, 3, hi, fmt="0.00", fill=DATA_FILL, border=True)
        put(ladder, rr, 4, note, wrap=True)
    header(ladder, 8, "OWNED-NVIDIA TCO CROSS-CHECK (from Inputs)", span=4)
    put(ladder, 9, 1, "Utilization")
    put(ladder, 9, 2, 0.85, fmt="0%", fill=INPUT_FILL, border=True)
    put(ladder, 10, 1, "Capex $/hr")
    put(
        ladder,
        10,
        2,
        f"={GPUCOST}/({LIFE}*8760*B9)",
        fmt="0.00",
        fill=CALC_FILL,
        border=True,
    )
    put(ladder, 10, 4, "fully-loaded $/GPU / (life x 8760h x utilization)", wrap=True)
    put(ladder, 11, 1, "Power $/hr")
    put(ladder, 11, 2, f"={PWR}*{ELEC}", fmt="0.00", fill=CALC_FILL, border=True)
    put(ladder, 11, 4, "wall kW x $/kWh", wrap=True)
    put(ladder, 12, 1, "DC/staff adder $/hr")
    put(ladder, 12, 2, 0.30, fmt="0.00", fill=INPUT_FILL, border=True)
    put(ladder, 13, 1, "Owned TCO $/hr", bold=True)
    put(
        ladder,
        13,
        2,
        "=B10+B11+B12",
        fmt="0.00",
        fill=CALC_FILL,
        border=True,
        bold=True,
    )
    put(ladder, 13, 4, "cross-checks the 'Buy + operate NVIDIA' row", wrap=True)
    nrow = 15
    for t in [
        "MARGIN STACK: own-silicon -> buy-NVIDIA ~1.4-2x (NVIDIA margin). buy -> rent ~2-3.5x (cloud margin). own -> rent ~3-5x.",
        "B200 builds ~$6,400, sells ~$40,000 -> ~84% gross margin. Hyperscalers charge 3-6x neocloud rates for identical HW.",
        "OWN-SILICON: TPU/Trainium/MTIA cost ~1/3 less per useful FLOP than NVIDIA. The model values compute at each company's ACTUAL cost (no gross-up).",
        "SemiAnalysis: TPU 20-50% lower TCO per useful FLOP vs GB200/GB300; Trainium3 ~30% better vs GB300.",
        "Sources: Spheron/IntuitionLabs (rental), Silicon Analysts (B200 cost/margin), SemiAnalysis (TPU/Trainium TCO).",
    ]:
        put(ladder, nrow, 1, t, wrap=True)
        ladder.merge_cells(start_row=nrow, start_column=1, end_row=nrow, end_column=4)
        nrow += 1


def build_evidence(ev):
    widths(ev, {"A": 40, "B": 14, "C": 12, "D": 52})
    header(
        ev,
        1,
        "EVIDENCE — cost split, accelerator share, own-silicon TCO, filing data",
        span=4,
    )

    def erow(rr, a, b="", c="", d="", bold=False):
        put(ev, rr, 1, a, bold=bold)
        put(ev, rr, 2, b)
        put(ev, rr, 3, c)
        put(ev, rr, 4, d, wrap=True)

    erow(
        3,
        "GPU cost split (BOM)",
        "$ cost",
        "% COGS",
        "Method 1: teardown = true resource cost",
        bold=True,
    )
    erow(4, "H100: HBM3 memory (80GB)", 1350, "41%", "MEMORY")
    erow(5, "H100: CoWoS packaging", 750, "23%", "mostly memory (interposer hosts HBM)")
    erow(6, "H100: test & assembly", 920, "28%", "shared")
    erow(7, "H100: logic die (compute)", 300, "9%", "COMPUTE -- cheapest part")
    erow(8, "H100 total COGS", 3320, "100%", "sells ~$28k -> ~88% margin")
    erow(
        9,
        "B200 total COGS",
        6400,
        "HBM 45%",
        "memory > logic die; sells ~$40k -> ~84% margin",
    )
    erow(
        11,
        "Accelerator share of server BOM",
        "",
        "",
        "for accel-within-server share",
        bold=True,
    )
    erow(12, "8x H100 server (J.P. Morgan)", "83%", "", "accelerator = $200k of $240k")
    erow(
        13,
        "8x A100 server (J.P. Morgan)",
        "71%",
        "",
        "GB200 rack ~76-80% (SemiAnalysis)",
    )
    erow(
        15,
        "Own-silicon vs NVIDIA (TCO)",
        "",
        "",
        "cheaper, but valued at cost in model",
        bold=True,
    )
    erow(
        16,
        "Google TPU v7 vs GB200/GB300",
        "20-50% lower",
        "",
        "per useful FLOP (SemiAnalysis)",
    )
    erow(
        17,
        "Amazon Trainium3 vs GB300",
        "~30% better",
        "",
        "chips ~1/3 cheaper to build",
    )
    erow(
        19,
        "Filing data (FY2025 actuals)",
        "total capex",
        "server life",
        "DISCLOSED top-lines",
        bold=True,
    )
    erow(
        20,
        "Microsoft (Jun'25, incl leases)",
        "~$88B",
        "2-6 yr",
        "'roughly half' short-lived (CFO)",
    )
    erow(21, "Alphabet", "$91.4B", "6 yr", "60% servers / 40% DC (CFO)")
    erow(
        22,
        "Amazon (cash capex)",
        "$128.3B",
        "5 yr (cut)",
        "AWS 67.8% of net P&E additions",
    )
    erow(
        23,
        "Meta (incl finance leases)",
        "$72.2B",
        "5.5 yr",
        "servers 'largest portion' (CFO)",
    )
    erow(
        24,
        "FY26 capex guidance",
        "",
        "",
        "MSFT ~$190B, Alphabet $180-190B, Amazon ~$200B, Meta $125-145B, Oracle ~$50B; SpaceX ~$18B (est)",
    )
    erow(
        26,
        "Cost-weighted reduction vs memory share",
        "reduction (x)",
        "",
        "live: =1/(w/memfac+(1-w)/flopfac)",
        bold=True,
    )
    for k, w in enumerate([0.45, 0.50, 0.60, 0.70, 0.82]):
        rr = 27 + k
        put(ev, rr, 1, f"memory share = {int(w * 100)}%")
        put(ev, rr, 2, w, fmt="0%", fill=INPUT_FILL, border=True)
        put(
            ev,
            rr,
            3,
            f"=1/(B{rr}/{MEMFAC}+(1-B{rr})/{FLOPFAC})",
            fmt="0.0",
            fill=CALC_FILL,
            border=True,
        )
        put(ev, rr, 4, "memory term dominates -> pinned near the 100x memory lever")


def build_methodology(meth):
    widths(meth, {"A": 120})
    lines = [
        ("METHODOLOGY & SOURCES", True),
        ("", False),
        (
            "Engine: GPU cost ~60% memory / ~40% compute. Memory x2,032 (lever H4, MEASURED at 262k on our test model) + inference compute x368 (levers H5 x H6: 64 conversations per GPU x 6.25x faster decode, with prompt processing added back at the active scenario's prefill speed; inference cost does not depend on model size) -> ~x724 cost-weighted (Inputs B20). The front tabs price inference on whole GPUs instead (the fleet shrinks by the smaller lever, x368) and training on GPU-hours (H1 x H2 x H3 = ~x18).",
            False,
        ),
        (
            "  Floored by the least-reduced part — at these levers that is COMPUTE (tokens per GPU, x368). With memory at the frontier-geometry ~x208 it would be memory again. The Today/Ceiling pair (retired 2026-09-29) and the pre-campaign family (x9.24 -> ~20x) are no longer used.",
            False,
        ),
        ("", False),
        (
            "Per company (own tab): total capex (DISCLOSED) x infra share x server share x accelerator share = accelerator capex; then fleet -> opex -> efficient -> value. FY25 + FY26.",
            False,
        ),
        (
            "  infra (data-center) share from 10-K/10-Q P&E and segment notes (MSFT 97%, Alphabet 93-95%, Meta 95-98%, Amazon AWS 68-76%); server share CFO-disclosed (MSFT ~50%, Google 60%); accel-within-server ~67-80% from BOM.",
            False,
        ),
        (
            "  Accel % of total varies: Amazon ~30% (legacy fleet + non-AI), MSFT ~37%, Google ~40%, Meta ~48%, SpaceX ~79% (greenfield).",
            False,
        ),
        (
            "Totals (front page): rolls up the company tabs live; no double-count (they don't pay each other for the bulk). GLOBAL row grosses the named total up to a worldwide estimate (Inputs 'Named share of global AI capex').",
            False,
        ),
        (
            "Value: spend cut/yr = accel x (1 - 1/reduction) ~95% + opex savings; capitalized = annual / discount rate. Compute valued at ACTUAL cost.",
            False,
        ),
        (
            "Energy/opex reduction: DERIVED = cost-weighted reduction (energy splits memory/compute like cost). Inputs B5 defaults to =B20; overtype that cell to override.",
            False,
        ),
        (
            "HOW EACH INPUT IS DERIVED: Inputs sheet column D notes every global assumption; Evidence sheet has the BOM cost split (mem share), accelerator-share teardowns, own-silicon TCO and filing top-lines; per-company capex/shares/revenue carry a basis + clickable Sources on each company tab.",
            False,
        ),
        (
            "Net AI economics (Totals + company tabs): Net AI = AI revenue - AI capex (full AI-infra) - AI opex (cash basis). With our arch: Net AI + spend cut.",
            False,
        ),
        (
            "  Shows they all LOSE money on AI today. TOGGLE 'Datacenter scaling factor' (Inputs B13): 0 = accelerator-only (conservative); 1 = whole datacenter scales -> net AI flips positive (~0.7 = breakeven).",
            False,
        ),
        ("", False),
        ("KEY RESULTS (defaults)", True),
        (
            "- Cost-weighted reduction ~x724 — compute-floored (tokens per GPU x368; memory x2,032).",
            False,
        ),
        (
            "- Net AI FY25: 6 named firms spend ~$363B (capex+opex) vs ~$79B AI revenue = ~ -$284B/yr cash burn.",
            False,
        ),
        (
            "- With our architecture: spend cut ~$163B -> burn shrinks to ~ -$121B/yr (~45% of AI spend cut).",
            False,
        ),
        (
            "- Spend-cut value (named floor, 6% discount rate): FY25 ~$163B/yr (~$2.7T capitalized); FY26 ~$385B/yr (~$6.4T). The cut saturates: ~99% of the chip bill.",
            False,
        ),
        (
            "- GLOBAL estimate (named ~80% of world AI capex): FY25 ~$3.4T, FY26 ~$8.0T capitalized. Clearly an estimate.",
            False,
        ),
        ("", False),
        ("CAVEATS", True),
        (
            "- AI REVENUE is the softest input: MSFT $37B & Amazon $15B run-rates are DISCLOSED; Google/Meta/Oracle/SpaceX are ESTIMATES. Meta's real AI payoff is indirect ad-uplift (~$20B), not direct revenue -- so its 'loss' here overstates.",
            False,
        ),
        (
            "- Net AI is CASH basis (capex not depreciated). On an accounting (depreciation) basis the loss is smaller; on a depreciation basis it is closer to a true P&L.",
            False,
        ),
        (
            "- No company reports 'accelerator capex'. Totals DISCLOSED; server/accel split ESTIMATED (server CFO-disclosed; accel-within from BOM). +/-15-20%.",
            False,
        ),
        (
            "- FY26 is a full live chain (FY26 total-capex guidance x FY26 shares). SpaceX FY26 ~$18B is an estimate.",
            False,
        ),
        (
            "- Hyperscaler market caps (company tab row 7) are approximate placeholders -- edit to current.",
            False,
        ),
        (
            "- Own-silicon (TPU/Trainium/MTIA) is cheaper per FLOP, but compute is valued at ACTUAL cost; CostLadder shows the buy/rent premium.",
            False,
        ),
        (
            "- Multiplying the levers (memory x2,032 * tokens per GPU x368) is NOT physical: a GPU is bought whole, so the fleet shrinks by the smaller of the two.",
            False,
        ),
        (
            "- JEVONS: savings reinvested into more AI, not budget cuts. Capitalization is a simple perpetuity.",
            False,
        ),
        ("", False),
        ("SOURCES", True),
        (
            "SpaceX S-1 / IPO: https://www.hl.co.uk/news/inside-spacexs-ipo-filing-revenue-starlink-ai-and-key-financials",
            False,
        ),
        (
            "Microsoft FY25 capex (Q3 FY26 call): https://www.fool.com/earnings/call-transcripts/2026/04/29/microsoft-msft-q3-2026-earnings-transcript/",
            False,
        ),
        (
            "Alphabet FY25 $91.4B, 60/40 split (Q4'25 call): https://www.fool.com/earnings/call-transcripts/2026/02/04/alphabet-googl-q4-2025-earnings-call-transcript/",
            False,
        ),
        (
            "Amazon FY25 $128.3B + server life 6->5yr (10-K): https://www.sec.gov/Archives/edgar/data/1018724/000101872426000004/amzn-20251231.htm",
            False,
        ),
        (
            "Meta FY25 $72.2B (Q4/FY25 release): https://investor.atmeta.com/investor-news/press-release-details/2026/Meta-Reports-Fourth-Quarter-and-Full-Year-2025-Results/default.aspx",
            False,
        ),
        (
            "H100/B200 BOM + margin (Silicon Analysts): https://siliconanalysts.com/analysis/nvidia-b200-blackwell-cost-breakdown",
            False,
        ),
        (
            "TPU/Trainium TCO (SemiAnalysis): https://newsletter.semianalysis.com/p/tpuv7-google-takes-a-swing-at-the",
            False,
        ),
        (
            "GPU rental prices 2026 (Spheron): https://www.spheron.network/blog/gpu-cloud-pricing-comparison-2026/",
            False,
        ),
        (
            "Microsoft $37B AI run-rate (Q3 FY26): https://news.alphastreet.com/microsoft-msft-q3-fy2026-azure-hits-40-growth-as-ai-business-reaches-37-billion-run-rate/",
            False,
        ),
        (
            "Amazon >$15B AWS AI run-rate (Q1 FY26): https://www.bnnbloomberg.ca/business/artificial-intelligence/2026/04/09/amazon-cloud-units-ai-revenue-run-rate-exceeds-us15-billion-in-first-quarter-ceo-says/",
            False,
        ),
        (
            "Hyperscalers losing money on AI (capex vs revenue): https://fortune.com/2026/04/15/data-centers-hyperscalers-spending-billions-on-hardware-thats-worthless-in-3-years/",
            False,
        ),
    ]
    for i, (t, b) in enumerate(lines, start=1):
        put(meth, i, 1, t, bold=b, wrap=True)


def build_serving_training(ws):
    """The 2026-08-07/08 multi-GPU receipts, priced on the CostLadder rates.
    Values are computed live from ai_capex_model (MEASURED / serving_economics /
    training_throughput_ratio); every measured cell names its derived.json key
    in ai_capex_model.MEASURED. Green = measured, yellow = assumption/projection,
    blue = derived."""
    from ai_capex_model import (MEASURED, SERVING, SERVING_TRAINING_ASSUMPTIONS,
                                PREFILL_BF16_D2048_SWEEP,
                                PREFILL_TOKENS_PER_S_D2048, KV_MB_PER_TOKEN_PER_STREAM,
                                TRAIN_ADVANTAGE_SHORT_SEQ,
                                serving_cost_curve, serving_economics,
                                training_throughput_ratio)

    widths(ws, {"A": 44, "B": 42, "C": 30, "D": 14, "E": 12, "F": 60})
    header(ws, 1,
           "SERVING & TRAINING — measured 2026-08-07/08 (2/4/8 x H100), priced at the CostLadder rent mid "
           f"(${SERVING['gpu_hr']:.2f}/GPU-hr). Source: bdm/docs/deck/build/derived.json (keys in ai_capex_model.MEASURED).")
    # --- measured workload block --------------------------------------------
    # Fixed rows on purpose. Since the 2026-09-01 re-base this blend is
    # measured BACKGROUND (pre-campaign kernels): Inputs!B24-B29 carry the
    # model-computed full-workload levers and Inputs!B3 points at B24, so
    # editing B7/B8/B9 moves this sheet but no longer drives the workbook.
    header(ws, 4, "WORKLOAD — measured pre-campaign blend (background; the workbook lever lives at Inputs B24-B29)")
    put(ws, 5, 1,
        "Training, prefill and decode point in opposite directions at today's kernel maturity, so the "
        "compute lever is blended over the workload rather than fixed. Cost per GENERATED token = "
        "in:out input tokens through prefill + one token through decode. Transformer decode follows "
        "the MEASURED 1/context law at its own KV ceiling (granted no KV compression by default; "
        "raise the KV-compression cell below to grant it a mature stack).",
        wrap=True)
    put(ws, 7, 1, "Input : output token ratio")
    put(ws, 7, 2, 10.0, "0.0", fill=INPUT_FILL, border=True)
    put(ws, 7, 6, "~10:1 = code / reasoning / agent traces. 50-100:1 = RAG and document QA.", wrap=True)
    put(ws, 8, 1, "E[context] (tokens)")
    put(ws, 8, 2, 65536, "#,##0", fill=INPUT_FILL, border=True)
    put(ws, 8, 6, "Our decode cost is context-flat and the transformer's is linear, so the decode advantage is linear in this.", wrap=True)
    put(ws, 9, 1, "Training share of accelerator cost")
    put(ws, 9, 2, 0.0, "0.00", fill=INPUT_FILL, border=True)
    put(ws, 9, 6, "0 = a serving-cost claim. Training is a LOSS at short sequence today; above ~0.06 the blend drops below 1.", wrap=True)

    put(ws, 11, 1, "Measured constants", bold=True)
    put(ws, 12, 1, "Owned decode, tokens/s/box (context-flat)")
    put(ws, 12, 2, MEASURED["serve_tokens_per_s_box"], "#,##0", fill=DATA_FILL, border=True)
    put(ws, 13, 1, "KV MB per token per stream")
    put(ws, 13, 2, KV_MB_PER_TOKEN_PER_STREAM, "0.000000", fill=DATA_FILL, border=True)
    put(ws, 14, 1, "HBM TB/s per GPU")
    put(ws, 14, 2, SERVING["hbm_tbps"], "0.00", fill=INPUT_FILL, border=True)
    put(ws, 15, 1, "Transformer KV compression (mature stack)")
    put(ws, 15, 2, SERVING["tf_kv_compression"], "0.0", fill=INPUT_FILL, border=True)
    put(ws, 16, 1, "GPUs per box")
    put(ws, 16, 2, SERVING["gpus_per_box"], "0", fill=INPUT_FILL, border=True)
    put(ws, 17, 1, "Training advantage, short sequence (owned / transformer)")
    put(ws, 17, 2, TRAIN_ADVANTAGE_SHORT_SEQ, "0.000", fill=DATA_FILL, border=True)
    put(ws, 17, 6,
        f"RE-BASED 2026-08-24/25 (1 x GH200, ONE layer, fwd+bwd, bf16, checkpointing off both arms, modern "
        f"transformer block 24Q/4KV head_dim 256 RoPE 64 gated): transformer is "
        f"{KERNEL_CAMPAIGN_20260824['step_ratio_2k']:.2f}x faster at T=2,048 "
        f"({KERNEL_CAMPAIGN_20260824['battn_ms_2k']:.1f} vs {KERNEL_CAMPAIGN_20260824['tf_ms_2k']:.1f} ms/step) and "
        f"{KERNEL_CAMPAIGN_20260824['step_ratio_8k']:.2f}x at T=8,192; the equal-token B4/T32,768 ~parity cell is "
        f"RETIRED 2026-08-25 as a B4 occupancy artifact, and the flat per-token model puts the crossover at a "
        f"PROJECTED T~{training_context_crossover():,.0f} (from T~88,000 pre-campaign). It read "
        f"{KERNEL_CAMPAIGN_20260824['step_ratio_2k_precampaign']:.2f}x "
        f"({KERNEL_CAMPAIGN_20260824['battn_ms_2k_precampaign']:.1f} ms) the same morning. Training PEAK MEMORY, same "
        f"frame: {KERNEL_CAMPAIGN_20260824['mem_ratio_2k']:.3f}x at T=2,048 "
        f"({KERNEL_CAMPAIGN_20260824['battn_peak_mib_2k']:,.1f} vs {KERNEL_CAMPAIGN_20260824['tf_peak_mib_2k']:,.1f} MiB), "
        f"from {KERNEL_CAMPAIGN_20260824['mem_ratio_2k_precampaign']:.3f}x -- that is TRAINING memory, not the H4 "
        f"serving-memory lever. TARGET (no receipt): T=8,192 step <= {KERNEL_CAMPAIGN_20260824['target_8k_win_gate_ms']:.2f} ms "
        f"and memory {KERNEL_CAMPAIGN_20260824['target_mem_ratio']:.2f}x.", wrap=True)

    put(ws, 19, 1, "Derived", bold=True)
    put(ws, 20, 1, "KV GB per stream")
    put(ws, 20, 2, "=$B$8*$B$13/1024", "0.00", fill=CALC_FILL, border=True)
    put(ws, 21, 1, "KV GB per stream, compressed")
    put(ws, 21, 2, "=$B$20/$B$15", "0.000", fill=CALC_FILL, border=True)
    put(ws, 22, 1, "Transformer decode, tokens/s/box")
    put(ws, 22, 2, "=511.8416*32768/$B$8*$B$15*$B$16", "#,##0", fill=CALC_FILL, border=True)
    put(ws, 22, 6,
        "MEASURED 1/context law (2026-08-14 re-base): 511.84 tok/s/GPU at 32,768 ctx x 32768/E[context] "
        "x KV-compression x GPUs. Crosses our flat 562/GPU near ~30k context.", wrap=True)
    # Log-log interpolation over the measured prefill table at rows 33..39.
    put(ws, 23, 1, "Prefill table row (lookup)")
    put(ws, 23, 2, "=MATCH($B$8,$A$33:$A$39,1)", "0", fill=CALC_FILL, border=True)
    _interp = (
        "=IF(INDEX({col}$33:{col}$39,MIN($B$23+1,7))=INDEX({col}$33:{col}$39,$B$23),"
        "INDEX({col}$33:{col}$39,$B$23),"
        "EXP(LN(INDEX({col}$33:{col}$39,$B$23))"
        "+(LN($B$8)-LN(INDEX($A$33:$A$39,$B$23)))"
        "/(LN(INDEX($A$33:$A$39,MIN($B$23+1,7)))-LN(INDEX($A$33:$A$39,$B$23)))"
        "*(LN(INDEX({col}$33:{col}$39,MIN($B$23+1,7)))-LN(INDEX({col}$33:{col}$39,$B$23)))))*$B$16"
    )
    put(ws, 24, 1, "Prefill tokens/s/box — transformer")
    put(ws, 24, 2, _interp.format(col="$B"), "#,##0", fill=CALC_FILL, border=True)
    put(ws, 25, 1, "Prefill tokens/s/box — owned")
    put(ws, 25, 2, _interp.format(col="$C"), "#,##0", fill=CALC_FILL, border=True)
    put(ws, 26, 1, "Transformer seconds per generated token")
    put(ws, 26, 2, "=$B$7/$B$24+1/$B$22", "0.000000", fill=CALC_FILL, border=True)
    put(ws, 27, 1, "Owned seconds per generated token")
    put(ws, 27, 2, "=$B$7/$B$25+1/$B$12", "0.000000", fill=CALC_FILL, border=True)
    put(ws, 28, 1, "BLENDED COMPUTE LEVER (flop_factor)", bold=True)
    put(ws, 28, 2,
        "=IF($B$9>0,1/($B$9/$B$17+(1-$B$9)/($B$26/$B$27)),$B$26/$B$27)",
        "0.00", fill=CALC_FILL, border=True, bold=True)
    put(ws, 28, 6, "Measured pre-campaign workload blend (2026-08-14 basis), kept as BACKGROUND. RETIRED as the workbook lever 2026-09-01: Inputs!B24-B29 now carry the model-computed full-workload levers (decode 2.5 ms x 64 streams estimate; this cell no longer feeds Inputs!B3).", wrap=True)
    put(ws, 29, 1, "  of which: prefill advantage")
    put(ws, 29, 2, "=($B$7/$B$24)/($B$7/$B$25)", "0.00", fill=CALC_FILL, border=True)
    put(ws, 30, 1, "  of which: decode advantage")
    put(ws, 30, 2, "=(1/$B$22)/(1/$B$12)", "0.0", fill=CALC_FILL, border=True)

    put(ws, 32, 1, "Measured prefill throughput, tokens/s/GPU (bf16 saturated lane, d2048)", bold=True)
    for c, t in enumerate(["Context", "Transformer", "Owned"], start=1):
        put(ws, 32, c + 3, t, bold=True, fill=SUB_FILL, border=True)
    for i, ctx in enumerate(sorted(PREFILL_TOKENS_PER_S_D2048)):
        tf_tps, own_tps = PREFILL_TOKENS_PER_S_D2048[ctx]
        put(ws, 33 + i, 1, ctx, "#,##0", border=True)
        put(ws, 33 + i, 2, tf_tps, "#,##0", fill=DATA_FILL, border=True)
        put(ws, 33 + i, 3, own_tps, "#,##0", fill=DATA_FILL, border=True)

    put(ws, 41, 1,
        "SCOPE: the serving and training blocks below are component-scope systems benches (bAttention "
        "d1536 fwd+bwd h-recurrence sub-block vs transformer d2048 forward-only layer); every speedup "
        "there is each family vs its OWN 1-GPU baseline. The prefill lanes quoted further down are "
        "cross-family: 16-bit parameter-matched at block scope (the fp32 full-model sweep is retired to an assumption row).",
        wrap=True)

    header(ws, 43, "Serving at long context — $/1M generated tokens, one 8xH100 box")
    for c, t in enumerate(["Context (tokens)", "bAttention $/1M tok", "Transformer $/1M tok (mature stack)",
                           "TF streams/GPU", "Cost ratio", "Note"], start=1):
        put(ws, 44, c, t, bold=True, fill=SUB_FILL, border=True)
    r = 45
    for row in serving_cost_curve():
        note = ("bAttention rate extrapolated past 262k (flatness measured 4k-262k)"
                if row["battn_extrapolated"] else
                ("a measured decode cell (64 streams/GPU against the transformer 1)" if row["ctx"] == 262144 else ""))
        put(ws, r, 1, row["ctx"], "#,##0", border=True)
        put(ws, r, 2, row["battn_usd_per_mtok"], "$0.0000",
            fill=INPUT_FILL if row["battn_extrapolated"] else DATA_FILL, border=True)
        put(ws, r, 3, row["tf_usd_per_mtok"], "$0.00", fill=CALC_FILL, border=True)
        put(ws, r, 4, row["tf_streams_per_gpu"], "0", fill=CALC_FILL, border=True)
        put(ws, r, 5, row["cost_ratio"], "0.0×", fill=CALC_FILL, border=True)
        put(ws, r, 6, note, wrap=True, border=True)
        r += 1
    put(ws, r, 1,
        "RE-BASED 2026-08-14 on the full-model decode receipt: per-GPU aggregate throughput at each family's "
        "own concurrency ceiling, transformer granted no KV compression. Ratio ~ context/29,800, so the "
        "TRANSFORMER is ahead below ~30k context; x2.2 at 64k, x8.8 at 262k. Memory-ceiling result, not a "
        "per-token one - at one stream and 64k the transformer decodes a token faster than we do. Granting it "
        f"KV /8: x{serving_economics(262144, s={'tf_kv_compression': 8.0})['cost_ratio']:.1f} at 262k.", wrap=True)
    r += 2

    header(ws, r, "Training at scale — same cluster, more steps/s")
    r += 1
    train_rows = [
        ("8-GPU speedup, matched load (16 in flight)", "x5.15 fwd+bwd vs x3.70 fwd-only Ulysses best",
         f"x{training_throughput_ratio():.2f} cluster throughput", DATA_FILL, "MEASURED — matched_load_curve"),
        ("8-GPU speedup, deep load (256 in flight)", "x6.27 (pipeline keeps filling; TF saturated by 16)",
         f"x{training_throughput_ratio(deep=True):.2f}", DATA_FILL, "MEASURED — scaling_1to8_best"),
        ("GPU-hours for the same training work", "-28% (matched) … -41% (deep)", "", CALC_FILL, "derived"),
        ("64k-token sequence on one 80 GB GPU", "30.9 GB fits vs OOM (78.4 GB attempted)", "", DATA_FILL,
         "MEASURED — pipeline_feasibility"),
        ("Pipeline per-GPU peak (flat in load)", "9.05 GB vs 43.3 GB GPipe stage", "x4.8 (component-scope, width-unmatched)",
         DATA_FILL, "MEASURED — pipeline_memory_asymmetry"),
        ("Params for equal quality at 70B", f"x{MEASURED['parity_70B_param_multiple']:.1f} [2.2-7.7] transformer",
         "", INPUT_FILL, "PROJECTION — quality_fit.ref_70B"),
        ("Tokens for equal quality at 70B", f"x{MEASURED['parity_70B_token_multiple']:.2f} [1.33-2.10] (beta=0.28 assumption)",
         "", INPUT_FILL, "PROJECTION — quality_fit.ref_70B"),
        ("Token-by-token vs production training", f"x{MEASURED['stepped_vs_scanned']:.0f} cost to run training token-by-token instead of the production path",
         "", DATA_FILL, "MEASURED — stepped_vs_scanned"),
    ]
    for c, t in enumerate(["Metric", "Value", "Ratio / consequence", "", "", "Status — derived.json key"], start=1):
        put(ws, r, c, t, bold=True, fill=SUB_FILL, border=True)
    r += 1
    for name, val, ratio, fill, status in train_rows:
        put(ws, r, 1, name, border=True)
        put(ws, r, 2, val, fill=fill, border=True, wrap=True)
        put(ws, r, 3, ratio, fill=CALC_FILL if ratio else None, border=True)
        put(ws, r, 6, status, border=True, wrap=True)
        r += 1
    r += 1

    header(ws, r, "The measured workload blend (pre-campaign background; the workbook lever lives at Inputs B24-B29)")
    r += 1
    put(ws, r, 1,
        "Training, prefill and decode point in OPPOSITE directions at today's kernel maturity, so a "
        "single number cannot represent them. Training is still a loss at short sequence, but a much "
        f"smaller one since the 2026-08-24 kernel re-base (transformer "
        f"{KERNEL_CAMPAIGN_20260824['step_ratio_2k']:.2f}x faster at T=2,048 and "
        f"{KERNEL_CAMPAIGN_20260824['step_ratio_8k']:.2f}x at T=8,192, crossover near T~13,000 (2026-08-29 anchor); it was "
        "5.3-9.6x with a crossover near T~88,000 on the pre-campaign d1152 sweep). Prefill crosses over near 65k context. "
        "Decode crosses in our favour near ~30k context. The lever is therefore blended over the workload: per "
        "generated token, in:out input tokens through prefill plus one token through decode, using "
        "measured per-GPU throughput at each family's own concurrency ceiling, the transformer granted "
        "no KV compression. At the default operating point (10:1, 64k context, training share 0): "
        "prefill x1.17, decode x2.20, blended x2.19, with decode ~99.7% of transformer serving cost. "
        "The decode half is a MEMORY CEILING, not a per-token result - at one stream and 64k the "
        "transformer decodes a token more cheaply than we do - and it crosses 1 at ~30k context, below "
        "which the transformer wins. Granting it KV/8 divides our decode lever by 8. A training share "
        "above ~0.30 also takes the blend below 1 (it was ~0.06 before the kernel re-base). "
        "All are reachable in the live model. NOTE (2026-09-01): this blend was measured on the "
        "PRE-CAMPAIGN kernels and no longer sets the workbook's flop_factor — the scenario levers at "
        "Inputs B24-B29 fold in the banked kernel results and the aggregate-decode estimate.",
        wrap=True)
    r += 2

    # --- estimated TRAINING cost at equal quality (mirrors the app's block)
    header(ws, r, "Estimated TRAINING cost at equal quality (ESTIMATE: measured kernels x fitted quality)")
    r += 1
    put(ws, r, 1,
        "Training compute ~ 6*N*D FLOPs. At equal quality we need N/s parameters, where s(N) is the "
        "FIT-DERIVED equal-quality parameter ratio (the same sealed 2026-08-14 refit as the compute "
        "lever). Two token regimes differ by exactly one power of s: COMPUTE-OPTIMAL (Chinchilla, D "
        "proportional to N -> cost ratio s^2/r) and FIXED TOKEN BUDGET (D equal on both arms -> s/r). "
        f"r(T) is the step-time ratio at matched parameters: MEASURED at T=2,048 ({KERNEL_CAMPAIGN_20260824['step_ratio_2k']:.2f}x, 2026-08-25) "
        f"and T=8,192 ({KERNEL_CAMPAIGN_20260824['step_ratio_8k']:.2f}x, 2026-08-24), MODELED beyond. Above 1.00 "
        "we are cheaper to train to the same quality. The 256k column is MODELED from the per-token cost "
        "form -- the transformer pays base + attn*T per token (FlashAttention is linear per token), our "
        "O(1) recurrent state pays ~flat -- with base and attn fitted to the 2k and 8k cells ONLY. The fit "
        f"is flat -- the +5.8%-per-4x residual and the B4/T32,768 ~parity cell that shared it are RETIRED 2026-08-25 as a "
        f"B4 occupancy artifact -- and puts the crossover at a PROJECTED T~{training_context_crossover():,.0f}. The old constant-decay "
        f"extrapolation ({KERNEL_CAMPAIGN_20260824['ratio_decay_per_4x']:.3f} per 4x) is kept as the CONSERVATIVE bound and reads "
        f"{training_step_ratio(262144, mode='constant_decay'):.2f} at 256k against this form's {training_step_ratio(262144):.2f}. The fits cross at 392M, so below that we need "
        "MORE parameters, and every row past ~1B extrapolates beyond the measured 47M-663M rungs. Peak "
        f"training memory ({KERNEL_CAMPAIGN_20260824['mem_ratio_2k']:.2f}x measured, {KERNEL_CAMPAIGN_20260824['target_mem_ratio']:.2f}x targeted) is NOT in "
        "this model: it caps per-GPU batch density, not FLOPs. Chart: "
        "paper/figures/vc_memo_training_cost_scaling.png",
        wrap=True)
    r += 2
    for c, t in enumerate(["Transformer params", "s(N) fit", "Chinchilla 2k", "Chinchilla 8k",
                           "Chinchilla 32k", "Chinchilla 256k*", "Fixed-D 2k / 8k / 32k / 256k*"], start=1):
        put(ws, r, c, t, bold=True, fill=SUB_FILL, border=True)
    r += 1
    for n_tf in (1e9, 1e10, 1e11, 1e12, 1e13):
        label = f"{n_tf / 1e9:,.0f}B" if n_tf < 1e12 else f"{n_tf / 1e12:,.0f}T"
        put(ws, r, 1, label, border=True)
        put(ws, r, 2, param_matching_gain(n_tf), "0.00", fill=INPUT_FILL, border=True)
        for c, ctx in enumerate((2048, 8192, 32768, 262144), start=3):
            put(ws, r, c, training_cost_saving(n_tf, ctx, chinchilla=True), "0.00",
                fill=CALC_FILL, border=True)
        put(ws, r, 7, " / ".join(
            f"{training_cost_saving(n_tf, ctx, chinchilla=False):.2f}"
            for ctx in (2048, 8192, 32768, 262144)), fill=CALC_FILL, border=True)
        r += 1
    r += 1

    put(ws, r, 1,
        "TRAINING IS A CURRICULUM, NOT ONE CONTEXT. Quoting r at a single T assumes ALL training happens "
        "there; this model used to assume T=2,048 for all of it, which is the pessimistic corner. The "
        "correct training term is the ratio of COSTS integrated over the mix of contexts a run actually "
        "uses, NOT the average of the per-context ratios. That distinction does the work: our per-token "
        "cost is flat in context and the transformer's grows, so long-context tokens dominate its bill "
        "while staying a token minority. TOKEN SHARES BELOW ARE ILLUSTRATIVE KNOBS -- this repo carries no "
        "citation for any lab's training recipe and none is invented here; only the first row (the "
        "degenerate one-context case) is anchored to a measurement. Effective r crosses below 1 at every "
        f"modern mix, but RAISING the dollar headline is a higher bar: the blended lever is a cost-weighted "
        f"harmonic mean, so training only pushes the headline up once it beats the SERVING advantage of "
        f"{training_helps_headline_threshold():.2f}x, not merely 1.00x. Between the two it is profitable and still dilutive.",
        wrap=True)
    r += 2
    for c, t in enumerate(["Training curriculum", "Advantage (TF / ours)", "Effective r",
                           "Mix: token share -> transformer cost share", "FY25 cut @ 20% training",
                           "vs serving-only", "Status"], start=1):
        put(ws, r, c, t, bold=True, fill=SUB_FILL, border=True)
    r += 1
    for key, spec in TRAINING_CURRICULA.items():
        a = training_advantage_mix(key)
        h = headline_with_training(key, 0.20)
        put(ws, r, 1, spec["label"], border=True)
        put(ws, r, 2, a["advantage"], "0.00×", fill=CALC_FILL, border=True)
        put(ws, r, 3, a["effective_step_ratio"], "0.000×", fill=CALC_FILL, border=True)
        put(ws, r, 4, "  ".join(
            f"{g['ctx'] // 1024}k: {g['token_share']:.0%} tok -> {g['tf_cost_share']:.0%} cost"
            for g in a["rungs"]), wrap=True, border=True)
        put(ws, r, 5, h["fy25_spend_cut"], "$#,##0.0", fill=CALC_FILL, border=True)
        put(ws, r, 6, h["delta_vs_serving_only"], "+$#,##0.00;-$#,##0.00", fill=CALC_FILL, border=True)
        put(ws, r, 7, spec["status"], fill=DATA_FILL if "MEASURED" in spec["status"] else INPUT_FILL,
            border=True)
        r += 1
    r += 1

    put(ws, r, 1,
        "EXCLUDED, AND IT RUNS OUR WAY. At 262k+ the transformer additionally pays sequence-parallelism "
        f"and activation-memory costs an O(1) recurrent state avoids: its KV grows at {KV_MB_PER_TOKEN_PER_STREAM * 1024:.0f} KB per token "
        "of context per stream, which is what forces sharding, ring/Ulysses attention and their "
        f"communication overhead. Our own measurement shows the asymmetry in the small -- from 2k to 8k the "
        f"peak-memory ratio went {KERNEL_CAMPAIGN_20260824['mem_ratio_2k']:.3f}x -> {KERNEL_CAMPAIGN_20260824['mem_ratio_8k']:.3f}x, i.e. flat-to-falling in context, while the "
        "absolute footprint the transformer must shard keeps growing. None of it is in the cost model "
        "above, which counts step time only, so the omission is CONSERVATIVE. Quantifying it needs a "
        "multi-GPU long-context TRAINING receipt we do not have.",
        wrap=True)
    r += 2

    put(ws, r, 1,
        "The prefill component, 16-bit and parameter-matched — the lane the competition serves in. "
        "Parameter counts exactly matched, 1 layer, batch 16, 1 x H100 80GB HBM3, 2026-07-21 — "
        "x4.02 at d512/1M, x3.83 at d1024/1M, x2.01 at d2048/262k. The fp32 full-model sweep and the "
        "own-baseline 16-bit-IO training gate are retired to one assumption row each below. Receipts: "
        "experiments/paper_figures/output/matched_d{512,1024,2048}_h100.json",
        wrap=True)
    r += 2
    for c, t in enumerate(["Context (tokens)", "Transformer ms", "bAttention ms", "TF / bAttention", "", "Note"], start=1):
        put(ws, r, c, t, bold=True, fill=SUB_FILL, border=True)
    r += 1
    for T_tok, tf_ms, ba_ms in PREFILL_BF16_D2048_SWEEP:
        ratio = tf_ms / ba_ms
        note = "transformer ahead" if ratio < 1.0 else ""
        put(ws, r, 1, T_tok, "#,##0", border=True)
        put(ws, r, 2, tf_ms, "#,##0.0", fill=CALC_FILL, border=True)
        put(ws, r, 3, ba_ms, "#,##0.0", fill=DATA_FILL, border=True)
        put(ws, r, 4, ratio, "0.00×", fill=CALC_FILL, border=True)
        put(ws, r, 6, note, wrap=True, border=True)
        r += 1
    r += 1

    header(ws, r, "Assumptions — one row each (MEASURED / PROJECTION / ASSUMPTION)")
    r += 1
    for c, t in enumerate(["Assumption", "Value", "", "", "Status", "Source"], start=1):
        put(ws, r, c, t, bold=True, fill=SUB_FILL, border=True)
    r += 1
    for a, b, st_, src in SERVING_TRAINING_ASSUMPTIONS:
        put(ws, r, 1, a, border=True)
        put(ws, r, 2, b, wrap=True, border=True)
        put(ws, r, 5, st_, fill=DATA_FILL if st_ == "MEASURED" else INPUT_FILL, border=True)
        put(ws, r, 6, src, wrap=True, border=True)
        r += 1


# =====================================================================================
# AUDIENCE LAYER: Summary / Value Bridge / Levers / What Matters.
# Plain-language front tabs. Same base as the company tabs, with the chip fleet
# split into a TRAINING fleet and an INFERENCE fleet so every dollar is attributable
# to the Helarctos lever (H1-H6) that produces it (ai_capex_model.value_bridge is the
# Python twin). Formulas use workbook-level NAMES (TrainLever, ServeGPULever, ...) so
# a reader can follow them without decoding cell addresses.
# =====================================================================================
from openpyxl.workbook.defined_name import DefinedName
from openpyxl.chart import BarChart, Reference, Series
from openpyxl.worksheet.datavalidation import DataValidation
from openpyxl.chart.marker import DataPoint
from openpyxl.chart.label import DataLabelList

from ai_capex_model import (
    CAMPAIGN_LANDED_20260831 as _LANDED,
    serving_context_sensitivity as _ctx_sens,
    HELARCTOS_LEVERS, lever_label,
)

# Chart colours: validated categorical palette (adjacent-pair CVD separation; two
# sit below 3:1 on white, so every bar carries its value label).
SOURCE_COLORS = ["2A78D6", "EB6834"]  # training, inference (each incl. its power)
LADDER_COLORS = ["A3A29C", "2A78D6", "EB6834"]  # baseline gray, then one hue per step
HELARCTOS_BAR, MARKET_BAR = "1BAF7A", "A3A29C"  # What Matters: Helarctos lever vs market data

FMT_B = '"$"#,##0"B";-"$"#,##0"B"'
FMT_B1 = '"$"#,##0.0"B";-"$"#,##0.0"B"'
FMT_T = '"$"#,##0.0"T";-"$"#,##0.0"T"'
FMT_X1 = '"×"0.0'
FMT_X2 = '"×"0.00'
FMT_X0 = '"×"#,##0'
FMT_DIV = '"÷"#,##0'
FMT_P = "0%"
FMT_DELTA = '+"$"#,##0.0"B";-"$"#,##0.0"B";"$0"'
BRAND_FILL = PatternFill("solid", fgColor="0B2545")
KPI_FILL = PatternFill("solid", fgColor="FCE4D6")
GROUP_FILL = PatternFill("solid", fgColor="EDEDED")
BIG = Font(bold=True, size=16, color="FFFFFF")
KPI_FONT = Font(bold=True, size=13)
NOTE_FONT = Font(size=10, color="595959")
MUTED = Font(italic=True, color="7F7F7F")
VB = "'Value Bridge'"
TAB_BRAND, TAB_COMPANY, TAB_TECH = "0B2545", "70AD47", "A6A6A6"
CENTER = Alignment(horizontal="center", vertical="center", wrap_text=True)
LEV = {x["code"]: x for x in HELARCTOS_LEVERS}


def _name(wb, name, ref):
    wb.defined_names[name] = DefinedName(name, attr_text=ref)


def _title(ws, text, span):
    ws.row_dimensions[1].height = 30
    c = ws.cell(row=1, column=1, value=scrub_external(text))
    c.font = BIG
    c.alignment = Alignment(vertical="center")
    for col in range(1, span + 1):
        ws.cell(row=1, column=col).fill = BRAND_FILL


def _section(ws, r, text, span, labels=None):
    """Dark section band (the same style as the technical tabs' headers).
    labels: {col: text} small column captions inside the band."""
    header(ws, r, text, span=span)
    ws.row_dimensions[r].height = 20
    ws.cell(row=r, column=1).alignment = Alignment(vertical="center")
    for col, t in (labels or {}).items():
        c = ws.cell(row=r, column=col, value=t)
        c.font = WHITE_BOLD
        c.alignment = Alignment(horizontal="center", vertical="center")


def _thead(ws, r, heads, c0=1, height=None):
    for j, h in enumerate(heads):
        c = put(ws, r, c0 + j, h, bold=True, border=True, fill=SUB_FILL)
        c.alignment = CENTER if j else Alignment(vertical="center", wrap_text=True)
    if height:
        ws.row_dimensions[r].height = height


def _para(ws, r, text, c1=1, c2=8, height=None, bold=False, note=False):
    c = put(ws, r, c1, text, wrap=True, bold=bold)
    if note:
        c.font = NOTE_FONT
    if c2 > c1:
        ws.merge_cells(start_row=r, start_column=c1, end_row=r, end_column=c2)
    if height:
        ws.row_dimensions[r].height = height
    return c


KEY_SIDE = Side(style="medium", color="C55A11")
KEY_BORDER = Border(left=KEY_SIDE, right=KEY_SIDE, top=KEY_SIDE, bottom=KEY_SIDE)


def key_cell(cell):
    """High-impact input: orange border (the label carries a ★)."""
    cell.border = KEY_BORDER


def lever_cell(cell):
    """Helarctos lever label: teal fill, dark-teal bold text, H1-H6 badge in the text."""
    cell.fill = HLEV_FILL
    cell.font = HLEV_FONT
    cell.border = BORDER
    cell.alignment = Alignment(vertical="center", wrap_text=True)
    return cell


def _lever_row_label(ws, r, c, code, prefix=""):
    return lever_cell(put(ws, r, c, prefix + lever_label(code)))


def _color_bars(ch, colors, fmt='"$"#,##0"B"'):
    ser = ch.series[0]
    for i, hexc in enumerate(colors):
        pt = DataPoint(idx=i)
        pt.graphicalProperties.solidFill = hexc
        pt.graphicalProperties.line.solidFill = hexc
        ser.dPt.append(pt)
    _labels(ser, fmt)
    ch.gapWidth = 60
    ch.x_axis.delete = False  # openpyxl 3.1 hides axes unless told otherwise
    ch.y_axis.delete = False
    ch.y_axis.majorGridlines = None
    ch.y_axis.numFmt = '"$"#,##0'
    if ch.type == "bar":
        ch.x_axis.scaling.orientation = "maxMin"  # list bars top-down in table order
        ch.y_axis.crosses = "max"                 # ...and keep the $ axis at the bottom


def _labels(ser, fmt):
    ser.dLbls = DataLabelList()
    ser.dLbls.showVal = True
    for attr in ("showSerName", "showCatName", "showLegendKey", "showPercent"):
        setattr(ser.dLbls, attr, False)
    ser.dLbls.numFmt = fmt


# ---- Levers ---------------------------------------------------------------------
# Fixed anchors (ai_capex_model.sensitivity_table quotes them; main() asserts).
LV_SCEN_ROW = 4
LV_LEVER_ROW0 = 9          # H1..H6 on rows 9..14, prompt-processing speed (supporting) on 15
LV_COMBO_ROW0 = 18
LV_MARKET_ROW0 = 27
LV_COMPANY_ROW0 = 37       # per-company training shares


def build_levers(ws, wb):
    widths(ws, {"A": 6, "B": 44, "C": 17, "D": 15, "E": 15, "F": 28, "G": 58, "H": 24})
    _title(ws, "LEVERS — what Helarctos changes (H1–H6) vs. market & company data", 8)
    _para(ws, 2,
          "Only the six teal levers (H1–H6) are about the Helarctos architecture. Everything else the front "
          "tabs use is market or company data, listed further down. ◆ purple = differs between the two "
          "scenarios · ★ orange border = high-impact input (What Matters tab).", 1, 8)

    r = LV_SCEN_ROW
    put(ws, r, 2, "◆ Scenario (pick from the list)", bold=True)
    put(ws, r, 3, "Current kernels", fill=SCEN_FILL, border=True, bold=True)
    dv = DataValidation(type="list", formula1='"Current kernels,Optimized kernels"', allow_blank=False)
    ws.add_data_validation(dv)
    dv.add(f"C{r}")
    _name(wb, "Scenario", f"Levers!$C${r}")
    _para(ws, r, "Current kernels = the software we have built and measured. Optimized kernels = our funded "
                 "kernel programme lands (a TARGET). Only the ◆ cells change: H3 and prompt-processing speed.", 4, 8,
          note=True)
    put(ws, r + 1, 2, "Average conversation length (context)", bold=True)
    put(ws, r + 1, 3, f"{ctx_label(CAMPAIGN_LANDED_20260831['context_tokens'])} tokens", fill=CALC_FILL,
        border=True, bold=True)
    _para(ws, r + 1, "H4, H5 and H6 are quoted at this context. Longer conversations favour Helarctos more.",
          4, 8, note=True)

    _section(ws, LV_LEVER_ROW0 - 2, "HELARCTOS LEVERS — the six things the architecture changes", 8)
    _thead(ws, LV_LEVER_ROW0 - 1, ["#", "Helarctos lever", "Active value", "◆ Current kernels",
                                   "◆ Optimized kernels", "How sure are we?", "What it means",
                                   "Saves money in"], height=32)
    sens = {int(d["context_tokens"]): d["today_lever"] for d in _ctx_sens()}
    s1t = param_matching_gain(DECK_DEPLOYMENT_SCALE)
    tp = lambda ctx: sens[ctx] / s1t  # noqa: E731
    r9 = LV_LEVER_ROW0
    rows = {
        "H1": ("=Inputs!$B$23", None, FMT_X2,
               "PROJECTED — quality trends measured on models we trained (47M–663M parameters), extended to "
               "~1T dense-equivalent",
               f"A Helarctos model matches a ~1T dense-equivalent transformer — about what today's 5–6T-total "
               f"mixture-of-experts flagships amount to — with ~{1 / s1t:.0%} of the parameters: {s1t:.1f}× fewer "
               f"numbers to store, update and run.",
               "Training (inference cost doesn't depend on model size)"),
        "H2": (f"=C{r9}", None, FMT_X2,
               "PROJECTED — standard compute-optimal scaling",
               f"Frontier labs train on data in proportion to model size, so a {s1t:.1f}× smaller model reaches "
               f"its best quality on ~{s1t:.1f}× fewer tokens.",
               "Training"),
        "H3": (TRAIN_SPEED_SAME_SIZE, TRAIN_SPEED_SAME_SIZE, FMT_X2,
               "ASSUMPTION — about the same as a transformer; no credit taken",
               "At the same size Helarctos trains at about the same speed per token, so no speed-up is counted "
               "(enter one under Optimized kernels if the kernel programme delivers it).",
               "Training"),
        "H4": ("=Inputs!$B$2", None, FMT_DIV,
               "MEASURED ×2,032 at 262k (on our test model). Frontier-size models with grouped-query attention: "
               f"~÷{fleet_memory_lever(262144):.0f} (estimate), where memory would bind again (about −$0.5B)",
               "A transformer's memory (the KV cache) grows with every token of every live conversation — "
               "51.6 GB for one 262k-token conversation. Helarctos keeps a fixed-size state (25.4 MB).",
               "Inference GPUs (the memory limit; not binding at 262k)"),
        "H5": ("=Inputs!$B$28", None, FMT_X0,
               "MEASURED — 64 conversations in 1.6 GB on one GPU (a grid cap); the transformer's 2nd OOMs",
               "The small memory (H4) lets one GPU hold 64 resident 262k-token conversations at once, where a "
               "transformer fits 1.",
               "Inference GPUs (tokens per GPU)"),
        "H6": ("=Inputs!$B$29", None, FMT_X1,
               "ESTIMATE — 2.5 ms per token after the kernel edits (receipt pending); MEASURED ×2.6 on our older "
               "kernels",
               f"Each conversation's next token takes 2.5 ms instead of the transformer's "
               f"{1000 / decode_tokens_per_s_per_gpu('transformer', 262144):.1f} ms at 262k: no growing cache to "
               f"re-read for every token.",
               "Inference GPUs (tokens per GPU)"),
    }
    for i, lv in enumerate(HELARCTOS_LEVERS):
        code, r = lv["code"], r9 + i
        today, opt, fmt, status, meaning, where = rows[code]
        badge = lever_cell(put(ws, r, 1, code))
        badge.alignment = CENTER
        _lever_row_label(ws, r, 2, code).value = lever_label(code).split(" · ", 1)[1]
        if opt is None:  # shared by both scenarios
            put(ws, r, 3, today, fmt, CALC_FILL, border=True, bold=True)
            for col in (4, 5):
                c = put(ws, r, col, "same", border=True)
                c.font = MUTED
                c.alignment = CENTER
        else:            # differs by scenario: active value follows the picker
            put(ws, r, 3, f'=IF(Scenario="Optimized kernels",E{r},D{r})', fmt, SCEN_FILL, border=True, bold=True)
            put(ws, r, 4, today, fmt, SCEN_FILL, border=True)
            put(ws, r, 5, opt, fmt, SCEN_FILL, border=True)
        if lv["high_impact"]:
            key_cell(ws[f"C{r}"])
        put(ws, r, 6, status, border=True, wrap=True)
        put(ws, r, 7, meaning, border=True, wrap=True)
        put(ws, r, 8, where, border=True, wrap=True)
    for i, nm in enumerate(("SmallerModel", "FewerTokens", "TrainSpeed", "MemoryLever", "MoreConversations",
                            "FasterDecode")):
        _name(wb, nm, f"Levers!$C${r9 + i}")
    # supporting input: prompt-processing (prefill) speed — the only inference cell that differs by scenario
    r = r9 + len(HELARCTOS_LEVERS)
    put(ws, r, 1, "", border=True)
    put(ws, r, 2, "Supporting: prompt-processing speed vs the transformer ◆", border=True, wrap=True)
    put(ws, r, 3, f'=IF(Scenario="Optimized kernels",E{r},D{r})', FMT_X1, SCEN_FILL, border=True, bold=True)
    put(ws, r, 4, "=Inputs!$B$24", FMT_X1, SCEN_FILL, border=True)
    put(ws, r, 5, "=Inputs!$B$25", FMT_X1, SCEN_FILL, border=True)
    put(ws, r, 6, "MEASURED ×3.94 kernel speed-up banked; the full ×7.03 is a TARGET", border=True, wrap=True)
    put(ws, r, 7, "A Helarctos kernel property, not a headline lever: prompts are ~0.2% of a transformer's time at "
                  "262k, so it barely moves the dollars.", border=True, wrap=True)
    put(ws, r, 8, "Inference GPUs (tokens per GPU)", border=True, wrap=True)
    _name(wb, "PromptSpeed", f"Levers!$C${r}")

    r0 = LV_COMBO_ROW0
    _section(ws, r0 - 1, "HOW THE LEVERS COMBINE — one number per fleet", 8)
    combos = [
        ("Training: GPU-hours per training run fall by", "=SmallerModel*FewerTokens*TrainSpeed", FMT_X1,
         "H1 × H2 × H3. Training clusters are sized to GPU-hours, so the training fleet can shrink by this "
         "much for the same training programme.", "TrainLever"),
        ("Inference: memory per conversation falls by", "=MemoryLever", FMT_DIV, "H4", None),
        ("Inference: decode tokens per GPU rise by", "=MoreConversations*FasterDecode", FMT_X0,
         "H5 × H6: more conversations per GPU, each decoded faster", "DecodeLever"),
        ("Inference: tokens per GPU, including prompt processing", "=Inputs!$B$27", FMT_X0,
         "H5 × H6 with the prompts added back at the supporting prompt-processing speed (inference cost doesn't "
         "depend on model size)", "ServeComputeLever"),
        ("Inference: GPUs needed fall by", "=MIN(MemoryLever,ServeComputeLever)", FMT_X0,
         '="The smaller of memory (H4) and tokens per GPU. A GPU is bought whole — memory and compute together — '
         'so the fleet covers whichever runs out first: here tokens per GPU binds, so faster decode counts. '
         'Multiplying them ("&TEXT(MemoryLever,"#,##0")&" × "&TEXT(ServeComputeLever,"#,##0")&") would be '
         'wrong."', "ServeGPULever"),
        ("For reference: technical tabs' blended cut", "=Inputs!$B$20", FMT_X0,
         "The technical appendix (Totals, company tabs) prices memory (~60% of a GPU's cost) and compute "
         "(~40%) separately and blends them. Same chip bill, within ~2% on FY2026.", None),
    ]
    for i, (lab, f, fmt, note, nm) in enumerate(combos):
        r = r0 + i
        put(ws, r, 2, lab, bold=i < 5, border=True, wrap=True)
        put(ws, r, 3, f, fmt, CALC_FILL, border=True, bold=i < 5)
        _para(ws, r, note, 4, 8, note=True)
        if nm:
            _name(wb, nm, f"Levers!$C${r}")

    r0 = LV_MARKET_ROW0
    _section(ws, r0 - 2, "MARKET & COMPANY DATA — the other inputs (not about Helarctos)", 8)
    _thead(ws, r0 - 1, ["", "Input", "Value", "Where to edit", "", "Note", "", ""])
    ws.merge_cells(start_row=r0 - 1, start_column=4, end_row=r0 - 1, end_column=5)
    ws.merge_cells(start_row=r0 - 1, start_column=6, end_row=r0 - 1, end_column=8)
    market = [
        ("AI data-centre capex, FY2026 (six companies)", f"={VB}!$B$6", FMT_B,
         "company tabs, rows 3–4 ★", "Disclosed / guided capex × data-centre share (from 10-K/10-Q notes)."),
        ("…share that buys AI chips", f"={VB}!$B$7/{VB}!$B$6", FMT_P,
         "company tabs, rows 5–6 ★", "Server share (CFO-disclosed) × accelerator share of servers (teardowns)."),
        ("Training share of the chip fleet (chip-weighted)", f"={VB}!$C$13", FMT_P,
         "per company, below", "Analyst estimates; no company discloses this."),
        ("Fully-loaded cost per GPU", "=Inputs!$B$7", '"$"#,##0', "Inputs B7",
         "Sizes the fleet for the power calculation."),
        ("Wall power per GPU (kW)", "=Inputs!$B$8", "0.0", "Inputs B8", "Power is ~4% of the saving."),
        ("Electricity ($/kWh)", "=Inputs!$B$9", '"$"0.00', "Inputs B9", "Power is ~4% of the saving."),
        ("Discount rate ★ (capitalized value only)", "=Inputs!$B$6", "0%", "Inputs B6",
         "Changes only the capitalized value, never the yearly saving."),
    ]
    for i, (lab, f, fmt, where, note) in enumerate(market):
        r = r0 + i
        put(ws, r, 2, lab, border=True, wrap=True)
        c = put(ws, r, 3, f, fmt, CALC_FILL, border=True)
        if "★" in lab:
            key_cell(c)
        put(ws, r, 4, where, border=True)
        ws.merge_cells(start_row=r, start_column=4, end_row=r, end_column=5)
        _para(ws, r, note, 6, 8, note=True)

    r = LV_COMPANY_ROW0
    _section(ws, r - 2, "TRAINING SHARE OF EACH COMPANY'S AI-CHIP FLEET — edit the yellow cells", 8)
    _para(ws, r - 1, "Estimates: no company discloses this. Analysts put training at 30–45% of AI compute in 2026 "
                     "(Gartner, Deloitte); the rest is inference, i.e. serving users.", 2, 8, note=True)
    why = {
        "Microsoft": "Inference-tilted: Azure serves the OpenAI API and Copilot; frontier training is moving to Stargate sites.",
        "Alphabet": "Trains Gemini on TPUs, but runs the largest inference estate (Search, ads, Cloud).",
        "Amazon": "Hosts large training clusters (Trainium) inside a broad AWS inference-rental business.",
        "Meta": "FY26 capex raise is driven by frontier training clusters, on top of ads-ranking inference.",
        "Oracle": "Stargate's dedicated OpenAI training data centres run on OCI.",
        "SpaceX": "Greenfield build (Colossus); early workloads are training-heavy.",
    }
    shares = _LANDED["train_share_by_company"]
    for i, c in enumerate(COMPANIES):
        rr = LV_COMPANY_ROW0 + i
        put(ws, rr, 2, c["name"], border=True)
        put(ws, rr, 3, shares.get(c["name"], _LANDED["train_share"]), FMT_P, INPUT_FILL, border=True)
        _para(ws, rr, why.get(c["name"], ""), 4, 8, note=True)

    fr = LV_COMPANY_ROW0 + len(COMPANIES) + 1
    _section(ws, fr, "FINE PRINT", 8)
    fine = [
        "Only AI chips and the power they draw are counted. The buildings, power infrastructure and networking "
        "around them would shrink too — upside, not in these numbers.",
        "Training is priced on GPU-hours only; no memory credit is taken on training clusters.",
        "H4 (×2,032) is measured on our test model; for frontier-size models with grouped-query attention the "
        "estimate is ~×208, where memory would bind again (about −$0.5B on FY2026). H6 is an estimate until the "
        "aggregate-decode receipt lands (×2.6 measured on our older kernels).",
        "H1 is a projection: quality trends measured up to 663M parameters, extended to ~1T dense-equivalent — "
        "about what today's 5–6T-total mixture-of-experts flagships (Grok 5 at 6T, Kimi K3 at 2.8T) amount to.",
        "The saving is spend no longer needed for the same AI output; firms will likely reinvest it. Cash basis; "
        "'capitalized' = yearly saving ÷ discount rate (6%, roughly the long-bond yield).",
    ]
    for i, t in enumerate(fine):
        put(ws, fr + 1 + i, 1, "•").alignment = Alignment(horizontal="center", vertical="top")
        _para(ws, fr + 1 + i, t, 2, 8)
    ws.freeze_panes = "A3"


# ---- Value Bridge -----------------------------------------------------------------
def _company_ref(name, col, row):
    return f"'{name}'!${col}${row}"


DETAIL_HEADS = ["Company", "AI-chip capex", "Training share", "Training fleet", "Inference fleet",
                "Training capex avoided", "Inference capex avoided", "Chip capex avoided",
                "Power & ops today", "Training power saved", "Inference power saved", "TOTAL saving",
                "AI spend (capex + power)", "% of AI spend cut", "AI revenue", "Net AI today",
                "Net AI with Helarctos"]


def _detail_table(ws, r0, year_col, label):
    """Per-company engine table. Returns ({col_letter: total_cell}, total_row)."""
    _section(ws, r0, f"DETAIL BY COMPANY — {label}, $B. The engine behind every number above.", len(DETAIL_HEADS))
    _thead(ws, r0 + 1, DETAIL_HEADS, height=45)
    first = r0 + 2
    for i, c in enumerate(COMPANIES):
        r = first + i
        n = c["name"]
        put(ws, r, 1, n, border=True)
        put(ws, r, 2, f"={_company_ref(n, year_col, 12)}", FMT_B, CALC_FILL, border=True)
        put(ws, r, 3, f"=Levers!$C${LV_COMPANY_ROW0 + i}", FMT_P, CALC_FILL, border=True)
        put(ws, r, 4, f"=B{r}*C{r}", FMT_B, CALC_FILL, border=True)
        put(ws, r, 5, f"=B{r}-D{r}", FMT_B, CALC_FILL, border=True)
        put(ws, r, 6, f"=D{r}*(1-1/TrainLever)", FMT_B, CALC_FILL, border=True)
        put(ws, r, 7, f"=E{r}*(1-1/ServeGPULever)", FMT_B, CALC_FILL, border=True)
        put(ws, r, 8, f"=F{r}+G{r}", FMT_B, CALC_FILL, border=True)
        put(ws, r, 9, f"={_company_ref(n, year_col, 34)}", FMT_B1, CALC_FILL, border=True)
        put(ws, r, 10, f"=IF(B{r}>0,I{r}*F{r}/B{r},0)", FMT_B1, CALC_FILL, border=True)
        put(ws, r, 11, f"=IF(B{r}>0,I{r}*G{r}/B{r},0)", FMT_B1, CALC_FILL, border=True)
        put(ws, r, 12, f"=H{r}+J{r}+K{r}", FMT_B, CALC_FILL, border=True, bold=True)
        put(ws, r, 13, f"={_company_ref(n, year_col, 10)}+I{r}", FMT_B, CALC_FILL, border=True)
        put(ws, r, 14, f"=L{r}/M{r}", FMT_P, CALC_FILL, border=True)
        put(ws, r, 15, f"={_company_ref(n, year_col, 32)}", FMT_B, CALC_FILL, border=True)
        put(ws, r, 16, f"=O{r}-M{r}", FMT_B, CALC_FILL, border=True)
        put(ws, r, 17, f"=P{r}+L{r}", FMT_B, CALC_FILL, border=True, bold=True)
    last = first + len(COMPANIES) - 1
    t = last + 1
    put(ws, t, 1, f"TOTAL ({len(COMPANIES)})", bold=True, border=True, fill=SUB_FILL)
    tot = {}
    for j in range(2, 18):
        col = chr(64 + j)
        if col == "C":
            f, fmt = f"=D{t}/B{t}", FMT_P
        elif col == "N":
            f, fmt = f"=L{t}/M{t}", FMT_P
        else:
            f = f"=SUM({col}{first}:{col}{last})"
            fmt = FMT_B1 if col in "IJK" else FMT_B
        put(ws, t, j, f, fmt, SUB_FILL, border=True, bold=True)
        tot[col] = f"{col}{t}"
    return tot, t


# Value Bridge anchors (fixed so Summary and the charts can point at them)
VB_RESULT_ROW = 40                      # Training / Inference result rows 40-41
VB_LADDER_ROW = 49                      # one-lever-at-a-time table header row
VB_DETAIL26_ROW = 74
VB_DETAIL25_ROW = VB_DETAIL26_ROW + len(COMPANIES) + 4


def build_value_bridge(ws):
    widths(ws, {"A": 52, "B": 14, "C": 12, "D": 14, "E": 14, "F": 14, "G": 14, "H": 14, "I": 14,
                "J": 13, "K": 13, "L": 13, "M": 14, "N": 11, "O": 11, "P": 12, "Q": 14})
    _title(ws, "VALUE BRIDGE — from AI spend to the Helarctos saving, step by step (FY2026)", 9)
    _para(ws, 2,
          "Read top to bottom. Teal rows are the Helarctos levers (edit them on the Levers and Inputs tabs); "
          "blue cells are live formulas.", 1, 9)

    d26, _ = _detail_table(ws, VB_DETAIL26_ROW, "C", "FY2026 (estimate)")
    d25, _ = _detail_table(ws, VB_DETAIL25_ROW, "B", "FY2025 (actual)")
    T = d26

    def row(r, label, f, fmt=FMT_B, note=None, bold=False, fill=CALC_FILL):
        put(ws, r, 1, label, bold=bold, wrap=True)
        c = put(ws, r, 2, f, fmt, fill, border=True, bold=bold)
        if fill is KPI_FILL:
            c.font = KPI_FONT
        if note:
            _para(ws, r, note, 4, 9, note=True)

    def lever(r, code, f, fmt, note, prefix=""):
        _lever_row_label(ws, r, 1, code, prefix)
        c = put(ws, r, 3, f, fmt, SCEN_FILL if LEV[code]["by_scenario"] else CALC_FILL, border=True)
        if LEV[code]["high_impact"]:
            key_cell(c)
        _para(ws, r, note, 4, 9, note=True)

    sumc = lambda cell: "=" + "+".join(_company_ref(c["name"], "C", cell) for c in COMPANIES)  # noqa: E731
    _section(ws, 4, "STEP 1 — What the six companies spend on AI this year", 9)
    row(5, "Total capex, all purposes", sumc(3), note="Company filings and guidance (company tabs, row 3).")
    row(6, "AI data-centre capex", sumc(10), note="Strips non-data-centre spend (e.g. Amazon retail logistics).")
    row(7, "…of which AI chips (GPUs, TPUs)", f"={T['B']}", bold=True,
        note="The part Helarctos shrinks. Buildings, power and networking are NOT counted (upside).")
    row(8, "Power & operations for those chips, per year", f"={T['I']}", FMT_B1)
    row(9, "AI revenue", f"={T['O']}", note="Disclosed run-rates (Microsoft, Amazon) or estimates (others).")
    row(10, "Net AI cash result today", f"={T['P']}", bold=True,
        note="AI revenue − AI capex − power. All six are losing money on AI today.")

    _section(ws, 12, "STEP 2 — Split the chip fleet by what it does", 9)
    row(13, "Training fleet — builds new models", f"={T['D']}",
        note="Per-company estimates, 25–55% (Levers tab); chip-weighted average shown.")
    put(ws, 13, 3, "=B13/B7", FMT_P, CALC_FILL, border=True)
    row(14, "Inference fleet — answers users", f"={T['E']}")
    put(ws, 14, 3, "=B14/B7", FMT_P, CALC_FILL, border=True)

    _section(ws, 16, "STEP 3 — TRAINING: a smaller model needs fewer GPU-hours → smaller training clusters", 9)
    row(17, "Training fleet capex today", "=B13")
    lever(18, "H1", "=SmallerModel", FMT_X2, "Same quality with a fraction of the parameters (projected).")
    lever(19, "H2", "=FewerTokens", FMT_X2, "A smaller model needs proportionally fewer training tokens.")
    lever(20, "H3", "=TrainSpeed", FMT_X2, "About the same speed per token at the same size — no credit taken.")
    put(ws, 21, 1, "= GPU-hours per training run fall by", bold=True)
    put(ws, 21, 3, "=TrainLever", FMT_X1, CALC_FILL, border=True, bold=True)
    _para(ws, 21, "H1 × H2 × H3", 4, 9, note=True)
    row(22, "Training fleet needed with Helarctos", "=B17/C21")
    row(23, "Training capex avoided", "=B17-B22")
    row(24, "+ power & operations those GPUs would have drawn", f"={T['J']}", FMT_B1,
        note="Power scales with the fleet.")
    row(25, "TRAINING SAVING (capex + power)", "=B23+B24", bold=True, fill=KPI_FILL,
        note='="Labs size training clusters to the GPU-hours their runs need: ~"&TEXT(TrainLever,"0")&"× fewer '
             'GPU-hours means a cluster ~"&TEXT(TrainLever,"0")&"× smaller for the same programme."')

    _section(ws, 27, "STEP 4 — INFERENCE: small memory → more conversations per GPU, each decoded faster → fewer "
                     "GPUs", 9)
    row(28, "Inference fleet capex today", "=B14")
    lever(29, "H4", "=MemoryLever", FMT_DIV, "Fixed-size state instead of a memory that grows with every token.")
    lever(30, "H5", "=MoreConversations", FMT_X0, "64 resident 262k-token conversations per GPU where a "
                                                  "transformer fits 1.")
    lever(31, "H6", "=FasterDecode", FMT_X1, "Each next token in 2.5 ms instead of ~15.6 ms (estimate).")
    put(ws, 32, 1, "Tokens per GPU, H5 × H6 with prompt processing added back")
    put(ws, 32, 3, "=ServeComputeLever", FMT_X0, CALC_FILL, border=True)
    _para(ws, 32, "Prompts are ~0.2% of a transformer's time at 262k (supporting prompt-processing speed, Levers "
                  "tab).", 4, 9, note=True)
    put(ws, 33, 1, "= Inference GPUs needed fall by (the smaller of memory and tokens per GPU)", bold=True, wrap=True)
    put(ws, 33, 3, "=ServeGPULever", FMT_X0, CALC_FILL, border=True, bold=True)
    _para(ws, 33, "A GPU is bought whole — memory and compute together — so the fleet covers whichever runs out "
                  "first. Here tokens per GPU binds, so faster decode (H6) counts in the dollars.", 4, 9, note=True)
    put(ws, 34, 1, "Inference fleet needed with Helarctos")
    put(ws, 34, 2, "=B28/C33", FMT_B1, CALC_FILL, border=True)
    row(35, "Inference capex avoided", "=B28-B34")
    row(36, "+ power & operations those GPUs would have drawn", f"={T['K']}", FMT_B1,
        note="Power scales with the fleet.")
    row(37, "INFERENCE SAVING (capex + power)", "=B35+B36", bold=True, fill=KPI_FILL)

    R = VB_RESULT_ROW
    _section(ws, R - 1, "RESULT — FY2026", 9)
    row(R, "Training (capex + power)", "=B25")
    row(R + 1, "Inference (capex + power)", "=B37")
    row(R + 2, "SPEND HELARCTOS MAKES UNNECESSARY, per year", f"=B{R}+B{R + 1}", bold=True, fill=KPI_FILL,
        note="Ties to the detail table total below.")
    row(R + 3, "…as a share of all AI spend", f"=B{R + 2}/{T['M']}", FMT_P, bold=True)
    row(R + 4, "…as a share of the AI-chip bill", "=(B23+B35)/B7", FMT_P)
    row(R + 5, "Net AI cash result with Helarctos", f"=B10+B{R + 2}", bold=True)
    row(R + 6, "Value of the yearly saving, capitalized (÷ discount rate)", f"=B{R + 2}/Inputs!$B$6/1000", FMT_T,
        note="Simple perpetuity at the Inputs discount rate (6%).")

    L = VB_LADDER_ROW
    _section(ws, L - 1, "SWITCH THE HELARCTOS LEVERS ON ONE AT A TIME (FY2026)", 9)
    _thead(ws, L, ["Step", "Training GPU-hours fall by", "Inference memory falls by",
                   "Inference compute falls by", "Chip fleet cut", "Spend cut, $B/yr", "Added by this step"],
           height=32)
    steps = [
        ("Today's transformer fleet", "1", "1", "1"),
        ("Smaller model trained on fewer tokens — training (H1–H3)", "=TrainLever", "1", "1"),
        ("+ Fixed-size memory, more conversations per GPU, faster decode — inference (H4–H6)", "=TrainLever",
         "=MemoryLever", "=ServeComputeLever"),
    ]
    TF, SF, ACC, OPX = T["D"], T["E"], T["B"], T["I"]
    for i, (lab, lt, lm, lc) in enumerate(steps):
        r = L + 1 + i
        put(ws, r, 1, lab, border=True, wrap=True, bold=(i == len(steps) - 1))
        for j, v, fmt in ((2, lt, FMT_X1), (3, lm, FMT_DIV), (4, lc, FMT_X0)):
            put(ws, r, j, v if v.startswith("=") else float(v), fmt, CALC_FILL, border=True)
        put(ws, r, 5, f"=({TF}*(1-1/B{r})+{SF}*(1-1/MIN(C{r},D{r})))/{ACC}", "0.0%", CALC_FILL, border=True)
        put(ws, r, 6, f"=E{r}*({ACC}+{OPX})", FMT_B, CALC_FILL, border=True, bold=True)
        put(ws, r, 7, f"=F{L + 1}" if i == 0 else f"=F{r}-F{r - 1}", '+"$"#,##0"B";-"$"#,##0"B";"—"',
            CALC_FILL, border=True)
    note_r = L + 1 + len(steps)
    _para(ws, note_r,
          "Training savings come from the smaller model (fewer parameters AND fewer tokens compound). Inference "
          "savings come from fixed-size memory, which lets each GPU hold many more long conversations, each "
          "decoded faster. Each "
          "fleet then shrinks by ~94–99%: bigger multiples can't add much, because a cost can only fall to zero "
          "once. That is why this workbook reports dollars, not multiples.", 1, 9, note=True)
    ch = BarChart()
    ch.type = "bar"
    ch.title = "Spend cut as each lever switches on ($B/yr, FY2026)"
    ch.add_data(Reference(ws, min_col=6, min_row=L, max_row=L + len(steps)), titles_from_data=True)
    ch.set_categories(Reference(ws, min_col=1, min_row=L + 1, max_row=L + len(steps)))
    ch.legend = None
    _color_bars(ch, LADDER_COLORS)
    ch.height, ch.width = 6.5, 19
    ws.add_chart(ch, f"A{note_r + 2}")

    xr = VB_DETAIL26_ROW - 5
    _section(ws, xr, "CROSS-CHECK against the technical Totals tab", 9)
    row(xr + 1, "Totals tab, FY2026 spend cut (memory and compute priced separately)", "=Totals!$F$13")
    row(xr + 2, "This bridge (training on GPU-hours, inference on whole GPUs)", f"=B{R + 2}")
    row(xr + 3, "Difference", f"=B{xr + 2}-B{xr + 1}",
        note="Small by design: both remove ~99% of the chip bill. Totals also honours the Inputs datacenter-"
             "scaling toggle (default 0); this bridge counts chips only.")
    ws.freeze_panes = "B3"
    return d26, d25


# ---- What Matters -----------------------------------------------------------------
WHY_IT_MATTERS = {
    "Server share of AI capex": "How much AI capex buys servers rather than buildings, power and networking. "
        "Helarctos shrinks the chip bill, so the dollars move one-for-one with it. CFO-disclosed for Microsoft "
        "and Alphabet.",
    "Accelerator share of servers": "The GPU/TPU share of server spend — same one-for-one effect. From chip and "
        "server teardowns (67–80%; a GB200 rack's GPUs are ~75–80% of its price).",
    "Smaller model for the same quality": "The biggest Helarctos-specific assumption: it drives the whole training "
        "saving (smaller model × fewer tokens). A projection from models up to 663M parameters to ~1T "
        "dense-equivalent (today's 5–6T mixture-of-experts flagships). High case: a 5T dense model.",
    "More conversations per GPU": "How many long conversations one GPU holds at once (64 vs the transformer's 1, "
        "measured). Small effect: inference GPUs already shrink by ~99.7%, so even ×16 costs under $2B.",
    "Faster decode per token": "Per-token decode speed-up (×6.3 estimate after the kernel edits; ×2.6 measured on "
        "our older kernels). With memory no longer the limit it counts in the dollars — but only ~$1B across the range.",
    "Memory per conversation": "Measured ×2,032 at 262k on our test model. Caveat: for frontier-size models with "
        "grouped-query attention the frontier-geometry estimate is ~÷208, where memory would bind again (≈ −$0.5B).",
    "Average conversation length": "Shorter conversations shrink both the memory advantage and tokens per GPU "
        "(×48 at 32k). The workbook is fixed at 262k; the app's sidebar lets you change it.",
    "Data-center share of capex": "Share of capex that goes into data centres rather than offices and other "
        "assets. From 10-K/10-Q property notes (93–98%; Amazon's AWS share 68–76%).",
    "Training speed per token": "Set to ×1 (about the same). Small effect: the smaller model already removes "
        "~94% of training GPU-hours.",
    "Electricity rate": "Only changes the power part of the saving (~4% of the total).",
    "Wall power per GPU": "Only changes the power part of the saving (~4% of the total).",
    "Training share of the chip fleet": "Small effect: both fleets shrink by 94–99%, so moving spend between them "
        "barely changes the total.",
    "Memory share of GPU cost": "No effect on the front tabs (GPUs are bought whole); under $1B on the technical "
        "Totals tab.",
}
WM_DISPLAY = {"H1+H2": "H1 + H2 · Smaller model, trained on fewer tokens ★"}


def wm_label(name, code):
    """What Matters row label: Helarctos rows carry their H-badge."""
    if not code:
        return name
    return WM_DISPLAY.get(code) or lever_label(code)


def build_what_matters(ws):
    from ai_capex_model import sensitivity_table, value_bridge
    widths(ws, {"A": 4, "B": 44, "C": 17, "D": 13, "E": 12, "F": 13, "G": 12, "H": 62})
    _title(ws, "WHAT MATTERS — which inputs move the answer, and by how much", 8)
    _, base = value_bridge(year="fy26")
    _para(ws, 2,
          f"Each row moves ONE input to a plausible low and high value, everything else at the defaults, and shows "
          f"the change in the FY2026 saving (${base['spend_cut']:,.0f}B/yr). ★ = moves it by $10B or more. "
          f"Values computed from the model at the defaults when this workbook was built.", 1, 8)
    _thead(ws, 4, ["★", "Input", "Where to edit", "Low case", "FY26 change", "High case", "FY26 change",
                   "Why it matters"], height=20)
    rows = sensitivity_table()
    groups = [
        ("Helarctos levers", "what the architecture changes", [x for x in rows if x[2]], HLEV_FILL),
        ("Market & company data", "not about Helarctos", [x for x in rows if not x[2]], GROUP_FILL),
    ]
    r = 5
    kinds = []  # per chart category: "h" / "m" / None (group row)
    for gname, gnote, items, gfill in groups:
        for col in range(1, 9):
            ws.cell(row=r, column=col).fill = gfill
        put(ws, r, 2, gname, bold=True)
        put(ws, r, 3, gnote).font = MUTED
        kinds.append(None)
        r += 1
        for name, where, code, lo_l, lo, hi_l, hi in items:
            big = max(abs(lo), abs(hi)) >= 10.0
            put(ws, r, 1, "★" if big else "", border=True, bold=True).alignment = CENTER
            lab = put(ws, r, 2, wm_label(name, code), border=True, wrap=True, bold=big)
            if code:
                lever_cell(lab)
            put(ws, r, 3, where, border=True, wrap=True)
            put(ws, r, 4, lo_l, border=True).alignment = CENTER
            c = put(ws, r, 5, lo, FMT_DELTA, CALC_FILL, border=True)
            put(ws, r, 6, hi_l, border=True).alignment = CENTER
            c2 = put(ws, r, 7, hi, FMT_DELTA, CALC_FILL, border=True)
            if big:
                key_cell(c)
                key_cell(c2)
            put(ws, r, 8, WHY_IT_MATTERS.get(name, ""), border=True, wrap=True).font = NOTE_FONT
            kinds.append("h" if code else "m")
            r += 1
    last = r - 1

    # tornado: one bar each side of zero per input; teal = Helarctos lever, gray = market data
    ch = BarChart()
    ch.type = "bar"
    ch.grouping = "clustered"
    ch.overlap = 100
    ch.gapWidth = 40
    ch.title = "Change in the FY2026 saving, $B (teal = Helarctos lever, gray = market & company data)"
    for col, title in ((5, "Low case"), (7, "High case")):
        s = Series(Reference(ws, min_col=col, min_row=5, max_row=last), title=title)
        for i, k in enumerate(kinds):
            if k:
                pt = DataPoint(idx=i)
                hexc = HELARCTOS_BAR if k == "h" else MARKET_BAR
                pt.graphicalProperties.solidFill = hexc
                pt.graphicalProperties.line.solidFill = hexc
                s.dPt.append(pt)
        _labels(s, '+"$"#,##0"B";-"$"#,##0"B";"$0"')
        ch.series.append(s)
    ch.set_categories(Reference(ws, min_col=2, min_row=5, max_row=last))
    ch.legend = None
    ch.x_axis.delete = False
    ch.y_axis.delete = False
    ch.x_axis.scaling.orientation = "maxMin"
    ch.x_axis.tickLblPos = "low"
    ch.y_axis.crosses = "max"
    ch.y_axis.numFmt = '"$"#,##0'
    ch.y_axis.majorGridlines = None
    ch.height, ch.width = 10, 24
    chart_r = last + 2
    ws.add_chart(ch, f"B{chart_r}")

    r = chart_r + 21
    _section(ws, r, "CAPITALIZED VALUE ONLY", 8)
    cap = base["spend_cut"]
    _para(ws, r + 1,
          f"★ The discount rate (Inputs B6) doesn't change the yearly saving, only its capitalized value: "
          f"${cap / 0.04 / 1000:,.1f}T at 4%, ${cap / 0.06 / 1000:,.1f}T at 6% (default), "
          f"${cap / 0.10 / 1000:,.1f}T at 10% (FY2026).", 1, 8)
    _section(ws, r + 3, "TAKEAWAY", 8)
    _para(ws, r + 4,
          "The answer is driven by how much these firms spend on AI chips (the market & company shares) more than "
          "by how large the Helarctos multiples are. The Helarctos levers move it only if they fall far below "
          "today's values — e.g. a smaller-model advantage of ×2 instead of ×4.2.", 1, 8)
    ws.freeze_panes = "A5"


# ---- Summary ----------------------------------------------------------------------
LEGEND = [
    ("H1–H6  Helarctos lever", "lever",
     "One of the six things the Helarctos architecture changes. Every other input is market, company or "
     "modelling data."),
    ("Assumption", INPUT_FILL, "A market, company or modelling assumption you can edit (not about Helarctos)."),
    ("Disclosed data", DATA_FILL, "From company filings or market data."),
    ("Formula", CALC_FILL, "Calculated live — don't overtype."),
    ("◆ Differs by scenario", SCEN_FILL,
     "The only cells that change between Current kernels and Optimized kernels (switch: Levers C4)."),
    ("★ High impact", "key", "Moves the FY2026 saving by $10B or more (What Matters tab)."),
]


def build_summary(ws, d26, d25):
    widths(ws, {"A": 50, "B": 14, "C": 14, "D": 11, "E": 15, "F": 15, "G": 15, "H": 15})
    _title(ws, "HELARCTOS × AI CAPEX — the AI spend a more efficient architecture makes unnecessary", 8)
    V = lambda cell: f"{VB}!{cell}"  # noqa: E731
    _para(ws, 2,
          f'="In FY2026 Microsoft, Alphabet, Amazon, Meta, Oracle and SpaceX will spend ~$"&TEXT({V(d26["M"])},'
          f'"#,##0")&"B on AI and earn ~$"&TEXT({V(d26["O"])},"#,##0")&"B from it. Helarctos models do the same '
          f'AI work — training and inference at the same quality — on far fewer GPUs. This page shows how much of '
          f'that spend becomes unnecessary, and which Helarctos levers drive it."', 1, 8, height=34)
    c = _para(ws, 3, f'="Scenario: "&Scenario&"  (◆ switch it on the Levers tab, cell C4)  ·  '
                     f'{ctx_label(CAMPAIGN_LANDED_20260831["context_tokens"])}-token average conversation  ·  '
                     f'$B per year unless stated"', 1, 8)
    c.font = MUTED
    c.alignment = Alignment(vertical="center")

    _section(ws, 5, "THE HEADLINE", 8)
    _thead(ws, 6, ["", "FY2026", "FY2025"])
    kpis = [
        ("AI spend: chips, data centres, power", "M", FMT_B, False),
        ("AI revenue", "O", FMT_B, False),
        ("Net AI cash result today", "P", FMT_B, False),
        ("Spend Helarctos makes unnecessary, per year", "L", FMT_B, True),
        ("…as a share of all AI spend", "N", FMT_P, True),
        ("Net AI cash result with Helarctos", "Q", FMT_B, False),
    ]
    for i, (lab, col, fmt, hi) in enumerate(kpis):
        r = 7 + i
        put(ws, r, 1, lab, bold=hi, border=True)
        for j, d in ((2, d26), (3, d25)):
            c = put(ws, r, j, f"={V(d[col])}", fmt, KPI_FILL if hi else CALC_FILL, border=True)
            if hi:
                c.font = KPI_FONT
        ws.row_dimensions[r].height = 22
    put(ws, 13, 1, "Yearly saving, capitalized at the discount rate", border=True, wrap=True)
    for j, d in ((2, d26), (3, d25)):
        put(ws, 13, j, f"={V(d['L'])}/Inputs!$B$6/1000", FMT_T, CALC_FILL, border=True)
    ws.row_dimensions[13].height = 30

    ch = BarChart()
    ch.type = "bar"
    ch.title = "Where the FY2026 saving comes from ($B/yr)"
    vbws = ws.parent[VB.strip("'")]
    ch.add_data(Reference(vbws, min_col=2, min_row=VB_RESULT_ROW, max_row=VB_RESULT_ROW + 1))
    ch.set_categories(Reference(vbws, min_col=1, min_row=VB_RESULT_ROW, max_row=VB_RESULT_ROW + 1))
    ch.legend = None
    _color_bars(ch, SOURCE_COLORS)
    ch.height, ch.width = 5.8, 11.6
    ws.add_chart(ch, "E6")

    _section(ws, 15, "WHERE THE FY2026 SAVING COMES FROM — and the Helarctos levers behind it", 8)
    _thead(ws, 16, ["Fleet / Helarctos lever", "$B per year", "Share", "Lever", "How it works", "", "", ""])
    ws.merge_cells("E16:H16")
    tr = 25
    fleets = [
        (17, "Training fleet (capex + power)", ("F", "J"), "=TrainLever", FMT_X1,
         '="H1 × H2 × H3 = ~"&TEXT(TrainLever,"0")&"× fewer GPU-hours per training run, so training clusters can '
         'be that much smaller. The GPUs never bought are capex avoided; their power is saved every year."',
         [("H1", "=SmallerModel", FMT_X2,
           f"Same quality with ~{1 / param_matching_gain(DECK_DEPLOYMENT_SCALE):.0%} of the parameters "
           f"(projected from models we trained)."),
          ("H2", "=FewerTokens", FMT_X2, "Compute-optimal training needs data in proportion to model size."),
          ("H3", "=TrainSpeed", FMT_X2, "About the same as a transformer at the same size — no credit taken.")]),
        (21, "Inference fleet (capex + power)", ("G", "K"), "=ServeGPULever", FMT_X0,
         '="Tokens per GPU rise ~"&TEXT(ServeComputeLever,"0")&"× (H5 × H6, with prompts added back). A GPU is '
         'bought whole, so the inference fleet shrinks by the smaller of memory and tokens per GPU: ~"'
         '&TEXT(ServeGPULever,"0")&"× fewer GPUs, plus the power they would draw."',
         [("H4", "=MemoryLever", FMT_DIV,
           "Fixed-size state instead of a memory that grows with every token (measured ×2,032 at 262k on our "
           "test model)."),
          ("H5", "=MoreConversations", FMT_X0,
           "64 resident 262k-token conversations per GPU where a transformer fits 1 (measured)."),
          ("H6", "=FasterDecode", FMT_X1,
           "Each next token in 2.5 ms instead of ~15.6 ms (estimate; ×2.6 measured on older kernels).")]),
    ]
    for r, lab, cols, lf, lfmt, how, levers in fleets:
        put(ws, r, 1, lab, bold=True, border=True)
        put(ws, r, 2, "=" + "+".join(V(d26[c]) for c in cols), FMT_B, KPI_FILL, border=True, bold=True)
        put(ws, r, 3, f"=B{r}/$B${tr}", FMT_P, CALC_FILL, border=True)
        put(ws, r, 4, lf, lfmt, CALC_FILL, border=True, bold=True)
        _para(ws, r, how, 5, 8, note=True)
        for i, (code, f, fmt, text) in enumerate(levers):
            rr = r + 1 + i
            _lever_row_label(ws, rr, 1, code, "    ")
            for col in (2, 3):
                ws.cell(row=rr, column=col).border = BORDER
            c = put(ws, rr, 4, f, fmt, SCEN_FILL if LEV[code]["by_scenario"] else CALC_FILL, border=True)
            if LEV[code]["high_impact"]:
                key_cell(c)
            _para(ws, rr, text, 5, 8, note=True)
    put(ws, tr, 1, "TOTAL", bold=True, border=True, fill=SUB_FILL)
    put(ws, tr, 2, "=B17+B21", FMT_B, SUB_FILL, border=True, bold=True)  # training + inference
    put(ws, tr, 3, "=C17+C21", FMT_P, SUB_FILL, border=True, bold=True)
    put(ws, tr, 4, None, fill=SUB_FILL, border=True)
    _para(ws, tr, "Step-by-step build and the one-lever-at-a-time view: Value Bridge tab.", 5, 8, note=True)

    _section(ws, 27, "BY COMPANY — FY2026, $B per year", 8)
    _thead(ws, 28, ["Company", "AI spend", "Training saving", "Inference saving", "Total saving",
                    "% of AI spend", "Net AI today", "Net AI with Helarctos"], height=32)
    base = VB_DETAIL26_ROW + 2
    for i, c in enumerate(COMPANIES + [None]):
        r = 29 + i
        src_r = base + i
        tot = c is None
        put(ws, r, 1, c["name"] if c else f"TOTAL ({len(COMPANIES)})", border=True, bold=tot,
            fill=SUB_FILL if tot else None)
        for j, cols, fmt in ((2, "M", FMT_B), (3, "FJ", FMT_B), (4, "GK", FMT_B), (5, "L", FMT_B),
                             (6, "N", FMT_P), (7, "P", FMT_B), (8, "Q", FMT_B)):
            put(ws, r, j, "=" + "+".join(f"{VB}!{col}{src_r}" for col in cols), fmt,
                SUB_FILL if tot else CALC_FILL, border=True, bold=tot or j == 5)

    r = 29 + len(COMPANIES) + 2
    _section(ws, r, "WHY ~99% OF THE CHIP BILL — NOT \"1,000×\"", 8)
    _para(ws, r + 1,
          '="Multiples don\'t multiply. A GPU is bought whole, so the inference fleet shrinks by whichever need '
          'falls least (×"&TEXT(ServeGPULever,"#,##0")&"), never by memory × tokens per GPU ("'
          '&TEXT(MemoryLever,"#,##0")&" × "&TEXT(ServeComputeLever,"#,##0")&"). Cutting a fleet "'
          '&TEXT(ServeGPULever,"#,##0")&"× already removes over 99% of it; bigger multiples only move the last '
          'fraction of a percent. So the dollars are set by how much these firms spend '
          'on chips — which is why this workbook reports dollars."', 1, 8, height=48)

    r += 3
    _section(ws, r, "HOW TO READ THIS WORKBOOK", 8)
    for k, (lab, style, text) in enumerate(LEGEND):
        rr = r + 1 + k
        c = put(ws, rr, 1, lab, border=True)
        if style == "lever":
            lever_cell(c)
        elif style == "key":
            key_cell(c)
            c.font = BOLD
        else:
            c.fill = style
            c.font = BOLD
        _para(ws, rr, text, 2, 8)
    r = r + len(LEGEND) + 2
    guide = [
        ("Summary", "This page: the headline, where the saving comes from, and by company."),
        ("Value Bridge", "Step by step from AI spend to the saving; the levers switched on one at a time."),
        ("Levers", "The six Helarctos levers, how sure we are of each, the scenario switch, and the market & "
                   "company data the front tabs use."),
        ("What Matters", "Which inputs move the answer and by how much — Helarctos levers vs market data."),
        (" / ".join(c["name"] for c in COMPANIES), "Each company's capex build from its filings."),
        ("Technical appendix", "Totals, Inputs, Sensitivity, CostLadder, ServingTraining, Evidence, Methodology — "
                               "the engineering ledger behind the levers."),
    ]
    for k, (n, d) in enumerate(guide):
        rr = r + k
        put(ws, rr, 1, n, bold=True, wrap=True)
        _para(ws, rr, d, 2, 8)
    _para(ws, r + len(guide) + 1,
          "Cash basis. FY2026 = company guidance. Only AI chips and their power are counted; buildings and power "
          "infrastructure are upside. Not investment advice.", 1, 8, note=True)
    ws.freeze_panes = "A2"


def center_rows(ws):
    """Vertically centre every cell, so labels, values and multi-line notes line up."""
    for row in ws.iter_rows():
        for c in row:
            if c.value is not None or c.fill.fill_type:
                al = c.alignment
                c.alignment = Alignment(horizontal=al.horizontal, vertical="center", wrap_text=al.wrap_text,
                                        indent=al.indent)


# ---- row-height fitting (no clipped text) ---------------------------------------
_STR_LIT = _re.compile(r'"((?:[^"]|"")*)"')


def _text_len(v):
    if not isinstance(v, str):
        return 0
    if v.startswith("="):  # text formula: count its string literals + ~6 chars per inserted value
        lits = _STR_LIT.findall(v)
        return sum(len(x) for x in lits) + 6 * v.count("TEXT(") + 16 * v.count("&Scenario")
    return len(v)


def _col_w(ws, col):
    from openpyxl.utils import get_column_letter
    w = ws.column_dimensions[get_column_letter(col)].width
    return w if w else 8.43


def fit_rows(ws):
    """Give every row enough height for its wrapped text (merged or not), and wrap
    long text that would otherwise be cut off by a neighbouring value. Never shrinks
    an explicit height."""
    merged = {}
    for mr in ws.merged_cells.ranges:
        if mr.min_row == mr.max_row:
            merged[(mr.min_row, mr.min_col)] = sum(_col_w(ws, c) for c in range(mr.min_col, mr.max_col + 1))
    covered = {(mr.min_row, c) for mr in ws.merged_cells.ranges for c in range(mr.min_col + 1, mr.max_col + 1)}
    need = {}
    for row in ws.iter_rows():
        for cell in row:
            v = cell.value
            if not isinstance(v, str) or (cell.row, cell.column) in covered:
                continue
            n = _text_len(v)
            w = merged.get((cell.row, cell.column), _col_w(ws, cell.column))
            size = (cell.font.sz or 11) if cell.font else 11
            cpl = max(1.0, w * (1.12 if size <= 10 else 1.0) * (0.9 if cell.font and cell.font.b else 1.0))
            wrap = bool(cell.alignment and cell.alignment.wrap_text)
            if not wrap and n > cpl * 1.05 and not v.startswith("="):
                right = ws.cell(row=cell.row, column=cell.column + 1).value
                if right not in (None, "") or (cell.row, cell.column) in merged:
                    al = cell.alignment
                    cell.alignment = Alignment(horizontal=al.horizontal, vertical="top", wrap_text=True)
                    wrap = True
            if wrap:
                lines = sum(max(1, -(-len(p) // int(cpl * 0.92))) for p in (v.split("\n") if not v.startswith("=")
                                                                    else [" " * n]))
                line_h = 15.0 if size <= 11 else size * 1.35
                need[cell.row] = max(need.get(cell.row, 0), lines * line_h + 3)
    for r, h in need.items():
        cur = ws.row_dimensions[r].height
        if h > (cur or 15):
            ws.row_dimensions[r].height = round(h, 1)


def main() -> None:
    wb = Workbook()
    summ = wb.active
    summ.title = "Summary"
    vb = wb.create_sheet("Value Bridge")
    lev = wb.create_sheet("Levers")
    wm = wb.create_sheet("What Matters")
    # companies single-sourced from ai_capex_model
    tabs = [c["name"] for c in COMPANIES]
    sheets = {name: wb.create_sheet(name) for name in tabs}
    # technical appendix: Totals (the previous front page) onward
    tot = wb.create_sheet("Totals")
    inp = wb.create_sheet("Inputs")
    sens = wb.create_sheet("Sensitivity")
    ladder = wb.create_sheet("CostLadder")
    servtrain = wb.create_sheet("ServingTraining")
    ev = wb.create_sheet("Evidence")
    meth = wb.create_sheet("Methodology")

    build_inputs(inp)
    for c in COMPANIES:
        build_company(
            sheets[c["name"]],
            c["name"],
            c["fy25"],
            c["fy26"],
            c["mcap"],
            c["ai_rev"],
            c["basis"],
            c.get("sources", ()),
        )
    build_totals(tot, tabs)
    build_sensitivity(sens)
    build_ladder(ladder)
    build_serving_training(servtrain)
    build_evidence(ev)
    build_methodology(meth)
    build_levers(lev, wb)
    build_what_matters(wm)
    d26, d25 = build_value_bridge(vb)
    build_summary(summ, d26, d25)

    # the "where to edit" cells quoted by the model must match this layout
    from ai_capex_model import sensitivity_table
    where = {r[0]: r[1] for r in sensitivity_table()}
    assert where["Training share of the chip fleet"] == \
        f"Levers C{LV_COMPANY_ROW0}:C{LV_COMPANY_ROW0 + len(COMPANIES) - 1}", where
    assert where["Training speed per token"] == f"Levers D{LV_LEVER_ROW0 + 2}/E{LV_LEVER_ROW0 + 2}", where
    # ★ flags on the levers must follow the >= $10B rule of the sensitivity table
    swing = {}
    for name, _, code, _, lo, _, hi in sensitivity_table():
        for c in (code.split("+") if code else []):
            swing[c] = max(swing.get(c, 0.0), abs(lo), abs(hi))
    for lv in HELARCTOS_LEVERS:
        if lv["code"] == "H2":  # tied to H1 (same value); the ★ sits on H1
            continue
        assert lv["high_impact"] == (swing.get(lv["code"], 0.0) >= 10.0), (lv["code"], swing)

    for ws in [summ, vb, lev, wm] + list(sheets.values()):
        center_rows(ws)
    for ws in wb.worksheets:
        fit_rows(ws)
    for ws in (summ, vb, lev, wm):
        ws.sheet_properties.tabColor = TAB_BRAND
    for name in tabs:
        sheets[name].sheet_properties.tabColor = TAB_COMPANY
    for ws in (tot, inp, sens, ladder, servtrain, ev, meth):
        ws.sheet_properties.tabColor = TAB_TECH
    wb.active = 0

    wb.calculation.fullCalcOnLoad = True
    out = "AI_Capex_Efficiency.xlsx"
    wb.save(out)
    print(f"wrote {out} with tabs: {wb.sheetnames}")


if __name__ == "__main__":
    main()
