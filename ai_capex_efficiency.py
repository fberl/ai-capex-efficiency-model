"""AI_Capex_Efficiency — $ value of cutting AI memory 100x and FLOPs 10x.

Layout:
  - Totals      : front page, all-company roll-up (live) + GLOBAL estimate row
  - one tab per company (one per entry in ai_capex_model.COMPANIES): full
    bottom-up build  total capex (DISCLOSED) -> infra share -> server bucket ->
    accelerator capex -> fleet/opex -> efficient version -> value, FY25 + FY26
  - Inputs      : global assumptions + the reduction engine (Amdahl cost-weighting)
  - Sensitivity : SpaceX value across reduction tiers
  - CostLadder  : own-silicon vs buy-NVIDIA vs rent $/GPU-hr
  - Evidence    : BOM split, accelerator-share data, own-silicon TCO, filing top-lines
  - Methodology : steps, caveats, sources

Engine: a GPU is ~60% memory / ~40% compute by cost. Memory x100 + FLOPs x10 ->
residual 0.6%+4% -> ~22x cost-weighted reduction, floored by compute. NOT 100x.

Every output is a LIVE FORMULA. Colors: yellow = assumption (a lever; each maps to a
slider in the Streamlit app), green = disclosed filing/market data, blue = derived.
Run:  uv run --with openpyxl python ai_capex_efficiency.py
"""

from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side

from ai_capex_model import GLOBALS, COMPANIES  # single source of truth for defaults

INPUT_FILL = PatternFill(
    "solid", fgColor="FFF2CC"
)  # yellow = ASSUMPTION (a lever / app slider)
DATA_FILL = PatternFill(
    "solid", fgColor="E2EFDA"
)  # green  = disclosed filing / market data
CALC_FILL = PatternFill("solid", fgColor="DDEBF7")  # blue   = derived formula
HEAD_FILL = PatternFill("solid", fgColor="1F4E78")
SUB_FILL = PatternFill("solid", fgColor="BDD7EE")
BOLD = Font(bold=True)
WHITE_BOLD = Font(bold=True, color="FFFFFF")
THIN = Side(style="thin", color="BBBBBB")
BORDER = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)


def safe_text(v):
    """Notes that DESCRIBE a formula start with '= ' (equals+space); strip the
    prefix so Excel stores them as text. Real formulas start with '='+non-space
    (e.g. '=B2*B3') and pass through untouched."""
    if isinstance(v, str) and v.startswith("= "):
        return v[2:]
    return v


def header(ws, row, text, span=6):
    ws.cell(row=row, column=1, value=safe_text(text)).font = WHITE_BOLD
    for col in range(1, span + 1):
        ws.cell(row=row, column=col).fill = HEAD_FILL


def put(ws, r, c, val, fmt=None, fill=None, border=False, bold=False, wrap=False):
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
    widths(inp, {"A": 40, "B": 13, "C": 8, "D": 76})
    header(
        inp,
        1,
        "GLOBAL INPUTS  —  edit yellow (assumption) cells; green = disclosed/market data",
        span=4,
    )
    G = GLOBALS  # values single-sourced from ai_capex_model; order fixes cell addresses B2..B14 (named-share = B14, used by the Totals global row)
    inputs = [
        (
            "Memory reduction factor",
            G["mem_factor"],
            "x",
            "1e-2 memory = 100x less (RNN O(1) state vs transformer O(T) KV cache).",
            INPUT_FILL,
        ),
        (
            "FLOPs reduction factor",
            G["flop_factor"],
            "x",
            "1e-1 FLOPs = 10x fewer.",
            INPUT_FILL,
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
            G["opex_reduction"],
            "x",
            "Energy ~ total FLOPs executed -> 10x floor.",
            INPUT_FILL,
        ),
        (
            "Discount rate",
            G["discount_rate"],
            "frac",
            "Perpetuity capitalization: value = annual benefit / rate (=10x at 10%).",
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
            "~1 kW TDP + node overhead x PUE 1.3.",
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
            "IPO 2026-06-12 ~$1.77T (market data; used by Sensitivity).",
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
    for label, val, unit, note, fill in inputs:
        put(inp, r, 1, label)
        put(
            inp,
            r,
            2,
            val,
            fmt=("0.0%" if unit == "frac" else "#,##0" if unit == "$" else "0.0"),
            fill=fill,
            border=True,
        )
        put(inp, r, 3, unit)
        put(inp, r, 4, note, wrap=True)
        r += 1
    header(inp, 15, "REDUCTION ENGINE (Amdahl cost-weighting) — derived", span=4)
    put(inp, 16, 1, "Compute share of GPU cost")
    put(inp, 16, 2, "=1-B4", fmt="0.0%", fill=CALC_FILL, border=True)
    put(inp, 16, 4, "1 - memory share", wrap=True)
    put(inp, 17, 1, "Memory cost fraction after reduction")
    put(inp, 17, 2, "=B4/B2", fmt="0.00%", fill=CALC_FILL, border=True)
    put(inp, 17, 4, "memory share / memory reduction", wrap=True)
    put(inp, 18, 1, "Compute cost fraction after reduction")
    put(inp, 18, 2, "=B16/B3", fmt="0.00%", fill=CALC_FILL, border=True)
    put(inp, 18, 4, "compute share / FLOPs reduction", wrap=True)
    put(inp, 19, 1, "Residual cost fraction")
    put(inp, 19, 2, "=B17+B18", fmt="0.0%", fill=CALC_FILL, border=True)
    put(inp, 19, 4, "Amdahl: floored by the least-reduced component", wrap=True)
    put(inp, 20, 1, "COST-WEIGHTED reduction factor", bold=True)
    put(inp, 20, 2, "=1/B19", fmt="0.0", fill=CALC_FILL, border=True, bold=True)
    put(inp, 20, 3, "x")
    put(
        inp,
        20,
        4,
        "1 / residual. The realistic $ reduction used everywhere.",
        wrap=True,
    )


def build_company(ws, name, c25, c26, mcap, ai_rev, basis, sources=()):
    """c25/c26 = (total_capex, infra_share, server_share, accel_share) for FY25/FY26.
    ai_rev = (revenue_FY25, revenue_FY26). sources = [(label, url), ...]."""
    widths(ws, {"A": 34, "B": 12, "C": 12, "D": 54})
    header(
        ws,
        1,
        f"{name} — accelerator capex & value (FY2025 actual + FY2026 estimate)",
        span=4,
    )
    put(ws, 2, 1, "Metric", bold=True)
    put(ws, 2, 2, "FY2025", bold=True)
    put(ws, 2, 3, "FY2026", bold=True)
    put(ws, 2, 4, "basis / source", bold=True)
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
            "strips non-AI (Amazon = AWS only)",
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
        put(ws, r, 1, lab)
        put(ws, r, 2, v25, fmt=fmt, fill=f25, border=True)
        put(ws, r, 3, v26, fmt=fmt, fill=f26, border=True)
        put(ws, r, 4, note, wrap=True)
    put(ws, 7, 1, "Market cap ($B, approx)")
    put(ws, 7, 2, mcap, fmt="#,##0", fill=DATA_FILL, border=True)
    put(ws, 7, 4, "approximate June 2026 market data -- edit", wrap=True)

    header(ws, 9, "DERIVATION", span=4)
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

    header(ws, 15, "FLEET & OPERATING COST (from accelerator capex)", span=4)
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

    header(ws, 23, "EFFICIENT VERSION & VALUE", span=4)
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
            "accel + (datacenter x dc-scale), all x (1-1/reduction)",
            False,
        ),
        (
            26,
            "Annual opex savings ($M)",
            f"=B20*(1-1/{OPXRED})",
            f"=C20*(1-1/{OPXRED})",
            "#,##0.0",
            "",
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
        put(ws, r, 1, lab, bold=bold)
        put(ws, r, 2, bf, fmt=fmt, fill=CALC_FILL, border=True, bold=bold)
        put(ws, r, 3, cf, fmt=fmt, fill=CALC_FILL, border=True, bold=bold)
        put(ws, r, 4, note, wrap=True)

    header(ws, 31, "AI ECONOMICS (cash basis: AI revenue - AI capex - AI opex)", span=4)
    rev25, rev26 = ai_rev
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
            "Spend cut, our arch ($B)",
            "=B27",
            "=C27",
            "#,##0.0",
            "= accel capex avoided + opex saved",
            False,
        ),
        (
            37,
            "Net AI WITH our arch ($B)",
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
            link = put(ws, r, 2, url)
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
        "They all spend far more on AI (capex + opex) than they earn from it -- they are losing money on AI today. Our architecture cuts the accelerator spend ~95%, shrinking the burn. The named firms are a floor; a GLOBAL estimate grosses up for the rest. Yellow assumption cells (also sliders in the app) and green disclosed-data cells live on company tabs.",
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
        "Spend cut, our arch ($B)",
        "Net AI W/ ARCH ($B)",
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

    sb = fr + 2
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
        (
            " / ".join(tabs),
            "one tab each: full build capex -> infra -> servers -> accelerator -> fleet/opex -> value",
        ),
        ("Inputs", "global assumptions + the reduction engine (cost-weighted factor)"),
        (
            "Sensitivity",
            "SpaceX value across reduction tiers (10x/30x/100x/cost-weighted)",
        ),
        ("CostLadder", "own-silicon vs buy-NVIDIA vs rent $/GPU-hr"),
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
        "Legend: yellow = assumption / lever (each maps to a slider in the Streamlit app)  ·  green = disclosed filing or market data  ·  blue = derived formula.",
        bold=True,
    )
    tot.merge_cells(start_row=lr, start_column=1, end_row=lr, end_column=8)


def build_sensitivity(sens):
    widths(sens, {"A": 30, "B": 16, "C": 16, "D": 16, "E": 18})
    header(
        sens,
        1,
        "SENSITIVITY (SpaceX) — count-based reduction tiers vs cost-weighted",
        span=5,
    )
    # base = ACCELERATOR capex (matches the SpaceX tab, which reduces only accelerator silicon
    # at the conservative dc_scale=0; NOT total capex, which would overstate the avoided spend)
    SXCAP, SXOPX = "SpaceX!$B$12", "SpaceX!$B$26"
    cols = [
        ("10x compute-bound", "10"),
        ("30x balanced", "30"),
        ("100x memory-bound", "100"),
        ("Cost-weighted", RED),
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
        "Base = SpaceX accelerator capex (SpaceX!B12), not total capex. The 'Cost-weighted' "
        "column ties to the SpaceX tab (B25/B27) at the conservative dc_scale=0.",
        wrap=True,
    )
    sens.merge_cells("A10:E10")


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
        "live: =1/(w/100+(1-w)/10)",
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
        put(ev, rr, 4, "compute term dominates -> stays far below 100x")


def build_methodology(meth):
    widths(meth, {"A": 120})
    lines = [
        ("METHODOLOGY & SOURCES", True),
        ("", False),
        (
            "Engine: GPU cost ~60% memory / ~40% compute. Memory x100 + FLOPs x10 -> residual 0.6%+4% -> ~22x cost-weighted reduction (Inputs B20).",
            False,
        ),
        (
            "  Floored by the least-reduced part (compute, 10x). Neither alone helps (mem-only ~2.5x, FLOP-only ~1.6x).",
            False,
        ),
        ("", False),
        (
            "Per company (own tab): total capex (DISCLOSED) x infra share x server share x accelerator share = accelerator capex; then fleet -> opex -> efficient -> value. FY25 + FY26.",
            False,
        ),
        (
            "  infra share strips non-AI (Amazon AWS 68%); server share CFO-disclosed (MSFT ~50%, Google 60%); accel-within-server ~67-80% from BOM.",
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
            "- Cost-weighted reduction ~22x (NOT 100x; range 17-38x over memory share 45-82%).",
            False,
        ),
        (
            "- Net AI FY25: 6 named firms spend ~$375B (capex+opex) vs ~$79B AI revenue = ~ -$295B/yr cash burn.",
            False,
        ),
        (
            "- With our architecture: spend cut ~$159B -> burn shrinks to ~ -$136B/yr (~42% of AI spend cut).",
            False,
        ),
        (
            "- Spend-cut value (named floor): FY25 ~$159B/yr (~$1.6T capitalized); FY26 r/r ~$328B/yr (~$3.3T).",
            False,
        ),
        (
            "- GLOBAL estimate (named ~80% of world AI capex): FY25 ~$2.0T, FY26 ~$4.1T capitalized. Clearly an estimate.",
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
            "- 1000x (=100x*10x) is NOT physical: cost is additive (Amdahl), not multiplicative.",
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


def main() -> None:
    wb = Workbook()
    tot = wb.active
    tot.title = "Totals"
    # companies single-sourced from ai_capex_model
    tabs = [c["name"] for c in COMPANIES]
    sheets = {name: wb.create_sheet(name) for name in tabs}
    inp = wb.create_sheet("Inputs")
    sens = wb.create_sheet("Sensitivity")
    ladder = wb.create_sheet("CostLadder")
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
    build_evidence(ev)
    build_methodology(meth)

    wb.calculation.fullCalcOnLoad = True
    out = "AI_Capex_Efficiency.xlsx"
    wb.save(out)
    print(f"wrote {out} with tabs: {wb.sheetnames}")


if __name__ == "__main__":
    main()
