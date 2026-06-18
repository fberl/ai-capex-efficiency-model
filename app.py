"""AI Capex Efficiency — interactive mirror of the AI_Capex_Efficiency workbook.

One Streamlit tab per worksheet (Totals, each company, Inputs, Sensitivity, CostLadder,
Evidence, Methodology). Each tab is the same colored grid as the Excel:
  🟡 assumption (a lever — editable)   🟢 disclosed filing/market data   🔵 derived
Edit the yellow/green cells (globals in the sidebar; per-company in the ✏️ panel on each
company tab) and every grid recomputes. All math comes from ai_capex_model.py, so the app
and the spreadsheet can't drift.

Run locally:  uv run --with streamlit --with pandas streamlit run app.py
Deploy free:  push to GitHub -> share.streamlit.io  (needs requirements.txt)
"""

import streamlit as st
import pandas as pd

from ai_capex_model import (GLOBALS, COMPANIES, reduction_factor, energy_reduction,
                            compute_company, compute_year, global_estimate)

st.set_page_config(page_title="AI Capex Efficiency", layout="wide")

# ---- spreadsheet palette (matches the .xlsx fills) -----------------------------
YEL, GRN, BLU, SUB = "#FFF2CC", "#E2EFDA", "#DDEBF7", "#BDD7EE"
CMAP = {"y": YEL, "g": GRN, "b": BLU, "s": SUB, "": ""}

st.markdown("""<style>
.xlhead{background:#1F4E78;color:#fff;padding:5px 10px;font-weight:600;
        border-radius:3px;margin:16px 0 4px;font-size:0.92rem;}
.block-container{padding-top:2.2rem;}
div[data-testid="stNumberInput"] input{padding:2px 6px;}
</style>""", unsafe_allow_html=True)


def section(title):
    st.markdown(f'<div class="xlhead">{title}</div>', unsafe_allow_html=True)


def show_table(columns, rows, widths=None, height=None):
    """rows: list of rows; each row is a list of (text, color) where color in CMAP.
    Renders a colored, Excel-like grid."""
    texts = [[c[0] for c in row] for row in rows]
    styles = [[f"background-color:{CMAP[c[1]]}" if c[1] else "" for c in row] for row in rows]
    df = pd.DataFrame(texts, columns=columns)
    smat = pd.DataFrame(styles, columns=columns)
    sty = df.style.apply(lambda _: smat, axis=None)
    cfg = {col: st.column_config.Column(width=w) for col, w in (widths or {}).items()}
    st.dataframe(sty, hide_index=True, width="stretch",
                 column_config=cfg or None, height=height or (len(rows) + 1) * 35 + 3)


# ---- formatters ----------------------------------------------------------------
def n1(v): return f"{v:,.1f}"
def n0(v): return f"{v:,.0f}"
def pct(v): return f"{v:.0%}"
def usd0(v): return f"${v:,.0f}"
def x1(v): return f"{v:.1f}×"
def md_usd(v): return f"\\${v:,.0f}"  # $-escaped for st.markdown/st.caption (Streamlit reads $…$ as LaTeX)


def fleet_breakdown(accel_b, g):
    """Intermediate fleet/energy rows (same primitives as the Excel generator)."""
    fleet = accel_b * 1e9 / g["gpu_cost"]
    mw = fleet * g["wall_power_kw"] / 1000
    mwh = mw * 24
    day = mwh * 1000 * g["elec_rate"] * (1 + g["cooling_overhead"])
    ann_m = day * 365 / 1e6
    life_b = ann_m * g["fleet_life_yr"] / 1000
    return dict(fleet=fleet, mw=mw, mwh=mwh, day=day, ann_m=ann_m, life_b=life_b)


# ---- sidebar: global assumptions (the Inputs-tab yellow cells) ------------------
def sidebar_globals():
    s = st.sidebar
    s.title("Global assumptions")
    s.caption("🟡 lever · 🟢 data — edit; every tab recomputes.")
    g = dict(GLOBALS)

    def gnum(label, key, lo, hi, step, fmt, help=None):
        st.session_state.setdefault(key, float(GLOBALS[key]))
        return s.number_input(label, min_value=float(lo), max_value=float(hi),
                              step=float(step), key=key, format=fmt, help=help)

    s.subheader("Architecture")
    g["mem_factor"] = gnum("🟡 Memory reduction (×)", "mem_factor", 1, 400, 5, "%.0f")
    g["flop_factor"] = gnum("🟡 FLOPs reduction (×)", "flop_factor", 1, 100, 1, "%.0f")
    g["mem_share"] = gnum("🟡 Memory share of GPU cost", "mem_share", 0.0, 1.0, 0.05, "%.2f")
    auto_energy = s.checkbox("Auto-derive energy reduction (= cost reduction)", value=True,
                             help="Operating energy splits between memory & compute like cost does, so the opex/energy "
                                  "reduction defaults to the cost-weighted reduction. Uncheck to set it manually.")
    if auto_energy:
        g["opex_reduction_override"] = None
    else:
        st.session_state.setdefault("opex_override", round(reduction_factor(g), 1))
        g["opex_reduction_override"] = s.number_input("🟡 Energy reduction override (×)", min_value=1.0,
                                                      max_value=200.0, step=1.0, key="opex_override", format="%.1f")
    s.caption(f"→ energy reduction = **{x1(energy_reduction(g))}** ({'derived' if auto_energy else 'manual override'})")
    s.subheader("Cost & energy")
    g["gpu_cost"] = gnum("🟡 Fully-loaded $/GPU", "gpu_cost", 5000, 150000, 1000, "%.0f")
    g["wall_power_kw"] = gnum("🟡 Wall power / GPU (kW)", "wall_power_kw", 0.3, 5.0, 0.1, "%.1f")
    g["elec_rate"] = gnum("🟡 Electricity rate ($/kWh)", "elec_rate", 0.02, 0.40, 0.01, "%.2f")
    g["cooling_overhead"] = gnum("🟡 Cooling/ops overhead", "cooling_overhead", 0.0, 1.0, 0.05, "%.2f")
    g["fleet_life_yr"] = gnum("🟡 Fleet life (yr)", "fleet_life_yr", 1, 10, 1, "%.0f")
    s.subheader("Finance & scope")
    g["discount_rate"] = gnum("🟡 Discount rate", "discount_rate", 0.02, 0.30, 0.01, "%.2f")
    g["dc_scale"] = gnum("🟡 Datacenter scaling factor", "dc_scale", 0.0, 1.0, 0.05, "%.2f",
                         help="0 = only accelerator silicon shrinks (conservative). 1 = whole DC scales. ~0.7 ≈ breakeven.")
    g["named_share_of_global"] = gnum("🟡 Named share of global AI capex", "named_share_of_global",
                                      0.3, 1.0, 0.05, "%.2f")
    g["spacex_mktcap"] = gnum("🟢 SpaceX market cap ($B)", "spacex_mktcap", 200, 4000, 10, "%.0f")

    s.metric("Cost-weighted reduction", x1(reduction_factor(g)))
    try:
        with open("AI_Capex_Efficiency.xlsx", "rb") as f:
            s.download_button("⬇ Download workbook (.xlsx)", f.read(), "AI_Capex_Efficiency.xlsx",
                              "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                              help="The full model at the default assumptions.")
    except FileNotFoundError:
        pass
    return g


# ---- per-company tab -----------------------------------------------------------
def edit_company_inputs(c):
    p = f"e_{c['name']}_"
    t25, i25, s25, a25 = c["fy25"]
    t26, i26, s26, a26 = c["fy26"]
    r25, r26 = c["ai_rev"]
    st.caption("🟡 assumption · 🟢 disclosed data — edit; the grids below recompute.")
    h = st.columns([2.4, 1, 1]); h[0].markdown("**Metric**"); h[1].markdown("**FY2025**"); h[2].markdown("**FY2026**")

    def two(label, key, d25, d26, lo, hi, step, fmt):
        cc = st.columns([2.4, 1, 1]); cc[0].write(label)
        st.session_state.setdefault(p + key + "25", float(d25))
        st.session_state.setdefault(p + key + "26", float(d26))
        v25 = cc[1].number_input(label + "25", min_value=float(lo), max_value=float(hi),
                                 step=float(step), key=p + key + "25", format=fmt, label_visibility="collapsed")
        v26 = cc[2].number_input(label + "26", min_value=float(lo), max_value=float(hi),
                                 step=float(step), key=p + key + "26", format=fmt, label_visibility="collapsed")
        return v25, v26

    t25, t26 = two("🟢/🟡 Total capex ($B)", "total", t25, t26, 0, 500, 1, "%.1f")
    i25, i26 = two("🟡 Infra / DC share", "infra", i25, i26, 0, 1, 0.01, "%.2f")
    s25, s26 = two("🟡 Server share", "server", s25, s26, 0, 1, 0.01, "%.2f")
    a25, a26 = two("🟡 Accelerator share", "accel", a25, a26, 0, 1, 0.01, "%.2f")
    r25, r26 = two("🟡 AI revenue ($B)", "rev", r25, r26, 0, 300, 0.5, "%.1f")
    st.session_state.setdefault(p + "mcap", float(c["mcap"]))
    mcap = st.number_input("🟢 Market cap ($B)", min_value=0.0, max_value=10000.0, step=10.0,
                           key=p + "mcap", format="%.0f")
    c["fy25"], c["fy26"] = (t25, i25, s25, a25), (t26, i26, s26, a26)
    c["ai_rev"], c["mcap"] = (r25, r26), mcap


def company_tab(c, g):
    name = c["name"]
    with st.expander(f"✏️  Edit {name}'s inputs (capex · shares · AI revenue)", expanded=False):
        edit_company_inputs(c)

    d25, d26 = compute_company(c, g, "fy25"), compute_company(c, g, "fy26")
    fb25, fb26 = fleet_breakdown(d25["accel"], g), fleet_breakdown(d26["accel"], g)
    t25, i25, s25, a25 = c["fy25"]; t26, i26, s26, a26 = c["fy26"]
    CB = ["Metric", "FY2025", "FY2026", "Basis / source"]
    W = {"Metric": "large", "Basis / source": "large"}

    section(f"{name} — inputs")
    show_table(CB, [
        [("Total capex ($B)", ""), (n1(t25), "g"), (n1(t26), "y"), ("FY25 disclosed; FY26 guide/est", "")],
        [("Infra / data-center share", ""), (pct(i25), "y"), (pct(i26), "y"), ("strips non-AI (Amazon = AWS)", "")],
        [("Server / short-lived share", ""), (pct(s25), "y"), (pct(s26), "y"), ("CFO-disclosed", "")],
        [("Accelerator share (within servers)", ""), (pct(a25), "y"), (pct(a26), "y"), ("BOM teardown ~67–80%", "")],
        [("Market cap ($B)", ""), (n0(c["mcap"]), "g"), ("", ""), ("approx market data", "")],
    ], widths=W)

    section("Derivation")
    show_table(CB, [
        [("AI-infra capex ($B)", ""), (n1(d25["ai_capex"]), "b"), (n1(d26["ai_capex"]), "b"), ("= total × infra", "")],
        [("Server bucket ($B)", ""), (n1(d25["ai_capex"] * s25), "b"), (n1(d26["ai_capex"] * s26), "b"), ("= infra × server", "")],
        [("ACCELERATOR capex ($B)", ""), (n1(d25["accel"]), "b"), (n1(d26["accel"]), "b"), ("= server × accel", "")],
        [("Accel % of total capex", ""), (pct(d25["accel_pct"]), "b"), (pct(d26["accel_pct"]), "b"), ("varies by company", "")],
    ], widths=W)

    section("Fleet & operating cost (from accelerator capex)")
    show_table(CB, [
        [("Fleet size (GPU-equiv)", ""), (n0(fb25["fleet"]), "b"), (n0(fb26["fleet"]), "b"), ("= accel capex / $ per GPU", "")],
        [("Total wall power (MW)", ""), (n0(fb25["mw"]), "b"), (n0(fb26["mw"]), "b"), ("= GPUs × kW", "")],
        [("Daily energy (MWh)", ""), (n0(fb25["mwh"]), "b"), (n0(fb26["mwh"]), "b"), ("", "")],
        [("Daily all-in opex ($/day)", ""), (usd0(fb25["day"]), "b"), (usd0(fb26["day"]), "b"), ("= MWh × rate × (1+overhead)", "")],
        [("Annual opex ($M)", ""), (n0(fb25["ann_m"]), "b"), (n0(fb26["ann_m"]), "b"), ("", "")],
        [("Lifetime opex ($B)", ""), (n1(fb25["life_b"]), "b"), (n1(fb26["life_b"]), "b"), ("× fleet life", "")],
    ], widths=W)

    section("Efficient version & value")
    show_table(CB, [
        [("Efficient AI capex ($B)", ""), (n1(d25["ai_capex"] - d25["capex_avoided"]), "b"), (n1(d26["ai_capex"] - d26["capex_avoided"]), "b"), ("= AI capex − avoided", "")],
        [("Capex avoided/yr ($B)", ""), (n1(d25["capex_avoided"]), "b"), (n1(d26["capex_avoided"]), "b"), ("accel(+DC×dc_scale) × (1−1/reduction)", "")],
        [("Annual opex savings ($M)", ""), (n0(d25["opex_saved"] * 1000), "b"), (n0(d26["opex_saved"] * 1000), "b"), ("", "")],
        [("Sustained annual benefit ($B/yr)", ""), (n1(d25["spend_cut"]), "b"), (n1(d26["spend_cut"]), "b"), ("= avoided + opex savings", "")],
        [("Capitalized value ($B)", ""), (n0(d25["capitalized"]), "b"), (n0(d26["capitalized"]), "b"), ("= benefit / discount rate", "")],
        [("% of market cap", ""), (f"{d25['capitalized'] / c['mcap']:.1%}", "b"), (f"{d26['capitalized'] / c['mcap']:.1%}", "b"), ("", "")],
    ], widths=W)

    section("AI economics (cash basis: AI revenue − AI capex − AI opex)")
    show_table(CB, [
        [("AI revenue ($B)", ""), (n1(c["ai_rev"][0]), "y"), (n1(c["ai_rev"][1]), "y"), ("ESTIMATE (see Methodology)", "")],
        [("AI capex ($B)", ""), (n1(d25["ai_capex"]), "b"), (n1(d26["ai_capex"]), "b"), ("full AI-infra (accel + buildings + power + net)", "")],
        [("AI opex ($B)", ""), (n1(d25["ai_opex"]), "b"), (n1(d26["ai_opex"]), "b"), ("annual power / operating", "")],
        [("Net AI NOW ($B)", ""), (n1(d25["net_now"]), "b"), (n1(d26["net_now"]), "b"), ("revenue − capex − opex (cash burn)", "")],
        [("Spend cut, our arch ($B)", ""), (n1(d25["spend_cut"]), "b"), (n1(d26["spend_cut"]), "b"), ("accel capex avoided + opex saved", "")],
        [("Net AI WITH our arch ($B)", ""), (n1(d25["net_arch"]), "b"), (n1(d26["net_arch"]), "b"), ("= net now + spend cut", "")],
        [("% AI spend reduction", ""), (pct(d25["pct_cut"]), "b"), (pct(d26["pct_cut"]), "b"), ("spend cut / total AI spend", "")],
    ], widths=W)

    if c.get("sources"):
        section("Sources & references")
        for label, url in c["sources"]:
            st.markdown(f"- [{label}]({url})")


# ---- totals tab ----------------------------------------------------------------
def econ_show(rows, tot, glob):
    cols = ["Company", "AI rev", "AI capex", "AI opex", "Net AI NOW", "Spend cut", "Net w/ ARCH", "% cut"]
    body = []
    for r in rows:
        body.append([(r["name"], ""), (n1(r["ai_rev"]), "b"), (n1(r["ai_capex"]), "b"), (n1(r["ai_opex"]), "b"),
                     (n1(r["net_now"]), "b"), (n1(r["spend_cut"]), "b"), (n1(r["net_arch"]), "b"), (pct(r["pct_cut"]), "b")])
    for d in (tot, glob):
        body.append([(d["name"], "s"), (n1(d["ai_rev"]), "s"), (n1(d["ai_capex"]), "s"), (n1(d["ai_opex"]), "s"),
                     (n1(d["net_now"]), "s"), (n1(d["spend_cut"]), "s"), (n1(d["net_arch"]), "s"), (pct(tot["pct_cut"]), "s")])
    show_table(cols, body, widths={"Company": "medium"})


def totals_tab(comps, g):
    rows25, tot25 = compute_year(g, comps, "fy25")
    rows26, tot26 = compute_year(g, comps, "fy26")
    glob25, glob26 = global_estimate(tot25, g), global_estimate(tot26, g)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Cost-weighted reduction", x1(reduction_factor(g)))
    c2.metric(f"Net AI now ({len(comps)} cos, FY25)", f"{usd0(tot25['net_now'])}B/yr")
    c3.metric("Net AI w/ our arch (FY25)", f"{usd0(tot25['net_arch'])}B/yr", delta=f"{usd0(tot25['spend_cut'])}B cut")
    c4.metric("% of AI spend cut", pct(tot25["pct_cut"]))
    verdict = ("AI flips **profitable** at these settings." if tot25["net_arch"] > 0
               else f"FY25 AI burn shrinks from **{md_usd(-tot25['net_now'])}B** to **{md_usd(-tot25['net_arch'])}B**/yr "
                    f"(spend cut **{md_usd(tot25['spend_cut'])}B**, ~{md_usd(tot25['capitalized'])}B capitalized).")
    st.markdown(f"**{verdict}**")
    st.caption(f"The {len(comps)} named firms are a floor. GLOBAL estimate (named ≈ {g['named_share_of_global']:.0%} of "
               f"world AI capex): FY25 spend cut ~{md_usd(glob25['spend_cut'])}B → ~\\${glob25['capitalized'] / 1000:.1f}T "
               f"capitalized; FY26 ~\\${glob26['capitalized'] / 1000:.1f}T. The rest is other clouds, China, neoclouds, xAI & sovereign AI.")

    section("Net AI economics — FY2025 (cash basis)")
    econ_show(rows25, tot25, glob25)
    section("Net AI economics — FY2026 (estimate)")
    econ_show(rows26, tot26, glob26)

    section("Savings breakdown & capitalized value")
    disc = g["discount_rate"]
    o25 = sum(r["opex_saved"] for r in rows25); k25 = sum(r["capex_avoided"] for r in rows25)
    o26 = sum(r["opex_saved"] for r in rows26); k26 = sum(r["capex_avoided"] for r in rows26)
    show_table(["Item", "Saved OPEX", "Avoided CAPEX (Overspend)", "Total"], [
        [("FY2025 annual ($B/yr)", ""), (n1(o25), "b"), (n1(k25), "b"), (n1(o25 + k25), "b")],
        [("FY2025 capitalized ($B)", ""), (n0(o25 / disc), "b"), (n0(k25 / disc), "b"), (n0((o25 + k25) / disc), "s")],
        [("FY2026 annual ($B/yr)", ""), (n1(o26), "b"), (n1(k26), "b"), (n1(o26 + k26), "b")],
        [("FY2026 capitalized ($B)", ""), (n0(o26 / disc), "b"), (n0(k26 / disc), "b"), (n0((o26 + k26) / disc), "s")],
        [("% reduction", ""), (pct(1 - 1 / energy_reduction(g)), "b"), (pct(1 - 1 / reduction_factor(g)), "b"), ("", "")],
    ])
    st.caption("OPEX = power saved each year (recoupable). CAPEX 'Overspend' = AI capex made unnecessary. "
               "Toggle the **Datacenter scaling factor** in the sidebar (0 = accelerator-only; ~0.7 ≈ breakeven; 1 = flips positive).")


# ---- inputs tab ----------------------------------------------------------------
def inputs_tab(g):
    section("Global inputs (edit in the sidebar ◀)")
    items = [
        ("Memory reduction factor", f"{g['mem_factor']:.0f}×", "y", "1e-2 memory = 100× less (RNN O(1) state vs transformer O(T) KV cache)"),
        ("FLOPs reduction factor", f"{g['flop_factor']:.0f}×", "y", "1e-1 FLOPs = 10× fewer"),
        ("Memory share of GPU cost", pct(g["mem_share"]), "y", "HBM + most CoWoS packaging → ~60/40 memory/compute (BOM)"),
        ("Opex / energy reduction", x1(energy_reduction(g)), "b" if g.get("opex_reduction_override") is None else "y", "DERIVED = cost-weighted reduction (energy splits memory/compute like cost); override in sidebar"),
        ("Discount rate", pct(g["discount_rate"]), "y", "perpetuity: value = annual benefit / rate"),
        ("Fully-loaded $/GPU", usd0(g["gpu_cost"]), "y", "GPU + share of server, NVLink, networking"),
        ("Wall power / GPU (kW)", f"{g['wall_power_kw']:.1f}", "y", "~1 kW TDP + node overhead × PUE 1.3"),
        ("Electricity rate ($/kWh)", f"{g['elec_rate']:.2f}", "y", "datacenter wholesale"),
        ("Cooling / ops overhead", pct(g["cooling_overhead"]), "y", "non-power running cost as fraction of electricity"),
        ("Fleet useful life (yr)", f"{g['fleet_life_yr']:.0f}", "y", "AI-GPU depreciation life"),
        ("Datacenter scaling factor", pct(g["dc_scale"]), "y", "0 = accel-only; 1 = whole DC scales; ~0.7 ≈ breakeven"),
        ("Named share of global AI capex", pct(g["named_share_of_global"]), "y", "named firms' share of worldwide AI capex (for the GLOBAL row)"),
        ("SpaceX market cap ($B)", n0(g["spacex_mktcap"]), "g", "market data (IPO 2026-06-12 ~$1.77T)"),
    ]
    show_table(["Input", "Value", "Note"], [[(a, ""), (b, c), (d, "")] for a, b, c, d in items],
               widths={"Input": "large", "Note": "large"})

    section("Reduction engine (Amdahl cost-weighting) — derived")
    cs = 1 - g["mem_share"]; mf = g["mem_share"] / g["mem_factor"]; cf = cs / g["flop_factor"]; res = mf + cf
    show_table(["Metric", "Value"], [
        [("Compute share of GPU cost", ""), (pct(cs), "b")],
        [("Memory cost fraction after reduction", ""), (f"{mf:.2%}", "b")],
        [("Compute cost fraction after reduction", ""), (f"{cf:.2%}", "b")],
        [("Residual cost fraction", ""), (pct(res), "b")],
        [("COST-WEIGHTED reduction factor", ""), (x1(1 / res), "b")],
    ])
    st.caption("Floored by the least-reduced component (compute). Neither alone helps much (mem-only ~2.5×, FLOP-only ~1.6×).")


# ---- sensitivity tab -----------------------------------------------------------
def sensitivity_tab(comps, g):
    section("Sensitivity (SpaceX) — reduction tiers vs cost-weighted")
    sx = next(c for c in comps if c["name"] == "SpaceX")
    d = compute_company(sx, g, "fy25")
    accel, opx, disc, mcap = d["accel"], d["opex_saved"] * 1000, g["discount_rate"], g["spacex_mktcap"]
    tiers = [("10× compute-bound", 10.0), ("30× balanced", 30.0), ("100× memory-bound", 100.0),
             (f"Cost-weighted {reduction_factor(g):.0f}×", reduction_factor(g))]
    cols = ["Metric"] + [t[0] for t in tiers]

    def row(label, fn, fmt):
        return [(label, "")] + [(fmt(fn(e)), "b") for _, e in tiers]

    show_table(cols, [
        row("Efficient acquisition ($B)", lambda e: accel / e, n1),
        row("Capex avoided/yr ($B)", lambda e: accel - accel / e, n1),
        row("Annual opex savings ($M)", lambda e: opx, n1),
        row("Sustained annual benefit ($B)", lambda e: (accel - accel / e) + opx / 1000, n1),
        row("Capitalized value ($B)", lambda e: ((accel - accel / e) + opx / 1000) / disc, n0),
        row("% of market cap", lambda e: ((accel - accel / e) + opx / 1000) / disc / mcap, pct),
    ])
    st.caption("Base = SpaceX accelerator capex (matches the SpaceX tab at the conservative dc_scale=0), not total capex.")


# ---- cost ladder tab -----------------------------------------------------------
def costladder_tab(g):
    section("Cost ladder — $/H100-equivalent GPU-hour (at scale)")
    lad = [
        ("Own custom silicon (TPU/Trainium)", "0.90", "1.40", "COGS + modest Broadcom/Marvell margin + power + DC. No NVIDIA margin."),
        ("Buy + operate NVIDIA (scale)", "1.50", "2.00", "NVIDIA ~84% gross margin baked into capex + power + DC."),
        ("Rent NVIDIA — neocloud / committed", "2.00", "3.50", "+ cloud provider capex recovery & margin."),
        ("Rent NVIDIA — hyperscaler on-demand", "3.00", "7.00", "+ utilization risk + flexibility premium."),
    ]
    show_table(["Procurement mode", "$/hr low", "$/hr high", "What's baked into the price"],
               [[(m, ""), (lo, "g"), (hi, "g"), (note, "")] for m, lo, hi, note in lad],
               widths={"Procurement mode": "large", "What's baked into the price": "large"})

    section("Owned-NVIDIA TCO cross-check (from inputs)")
    util = 0.85
    capx = g["gpu_cost"] / (g["fleet_life_yr"] * 8760 * util)
    powr = g["wall_power_kw"] * g["elec_rate"]; dc = 0.30
    show_table(["Metric", "$/hr", "Basis"], [
        [("Utilization", ""), (pct(util), "y"), ("assumed", "")],
        [("Capex $/hr", ""), (f"{capx:.2f}", "b"), ("$/GPU / (life × 8760h × util)", "")],
        [("Power $/hr", ""), (f"{powr:.2f}", "b"), ("wall kW × $/kWh", "")],
        [("DC/staff adder $/hr", ""), ("0.30", "y"), ("assumed", "")],
        [("Owned TCO $/hr", ""), (f"{capx + powr + dc:.2f}", "b"), ("cross-checks 'Buy + operate NVIDIA'", "")],
    ], widths={"Basis": "large"})
    st.caption("Own-silicon → buy-NVIDIA ~1.4–2× (NVIDIA margin); buy → rent ~2–3.5× (cloud margin); own → rent ~3–5×.")


# ---- evidence tab --------------------------------------------------------------
def evidence_tab(g):
    section("GPU cost split (BOM teardown = true resource cost)")
    show_table(["Component", "$ cost", "% COGS", "Note"], [
        [("H100: HBM3 memory (80GB)", ""), ("1,350", ""), ("41%", ""), ("MEMORY", "")],
        [("H100: CoWoS packaging", ""), ("750", ""), ("23%", ""), ("mostly memory (interposer hosts HBM)", "")],
        [("H100: test & assembly", ""), ("920", ""), ("28%", ""), ("shared", "")],
        [("H100: logic die (compute)", ""), ("300", ""), ("9%", ""), ("COMPUTE — cheapest part", "")],
        [("H100 total COGS", ""), ("3,320", ""), ("100%", ""), ("sells ~$28k → ~88% margin", "")],
        [("B200 total COGS", ""), ("6,400", ""), ("HBM 45%", ""), ("memory > logic; sells ~$40k → ~84% margin", "")],
    ], widths={"Note": "large"})

    section("Accelerator share of server BOM")
    show_table(["Server", "Accel share", "Note"], [
        [("8× H100 server (J.P. Morgan)", ""), ("83%", ""), ("accelerator = $200k of $240k", "")],
        [("8× A100 server (J.P. Morgan)", ""), ("71%", ""), ("GB200 rack ~76–80% (SemiAnalysis)", "")],
    ], widths={"Note": "large"})

    section("Filing data (FY2025 actuals)")
    show_table(["Company", "Total capex", "Server life", "Note"], [
        [("Microsoft (incl leases)", ""), ("~$88B", "g"), ("2–6 yr", ""), ("'half'→'two-thirds' short-lived (CFO)", "")],
        [("Alphabet", ""), ("$91.4B", "g"), ("6 yr", ""), ("60% servers / 40% DC (CFO)", "")],
        [("Amazon (cash capex)", ""), ("$128.3B", "g"), ("5 yr (cut)", ""), ("AWS 67.8% of net P&E additions", "")],
        [("Meta (incl finance leases)", ""), ("$72.2B", "g"), ("5.5 yr", ""), ("servers 'largest portion' (CFO)", "")],
        [("Oracle", ""), ("$21.2B", "g"), ("—", ""), ("~all OCI/GPU data centers; FY26 ~$50B guide", "")],
        [("SpaceX (S-1 AI capex)", ""), ("$12.7B", "g"), ("—", ""), ("~all accelerator (greenfield COLOSSUS)", "")],
    ], widths={"Note": "large"})

    section("Cost-weighted reduction vs memory share (live)")
    show_table(["Scenario", "Memory share", "Reduction"],
               [[(f"memory share = {int(w * 100)}%", ""), (pct(w), "y"),
                 (x1(1 / (w / g["mem_factor"] + (1 - w) / g["flop_factor"])), "b")]
                for w in (0.45, 0.50, 0.60, 0.70, 0.82)])
    st.caption("Compute term dominates → stays far below 100×.")


# ---- methodology tab -----------------------------------------------------------
def methodology_tab(g):
    section("Assumptions & how each value is derived")
    ek = ("derived", "b") if g.get("opex_reduction_override") is None else ("override", "y")
    rows = [
        [("Memory reduction (×)", ""), (f"{g['mem_factor']:.0f}×", "y"), ("assumption", "y"), ("Architecture claim — RNN O(1) state vs transformer O(T) KV cache → ~100× less", "")],
        [("FLOPs reduction (×)", ""), (f"{g['flop_factor']:.0f}×", "y"), ("assumption", "y"), ("Architecture claim — ~10× fewer FLOPs per token", "")],
        [("Memory share of GPU cost", ""), (pct(g["mem_share"]), "y"), ("assumption", "y"), ("BOM teardown: HBM ~41% + CoWoS ~23% (mostly memory) vs logic die ~9% → ~60/40 (Evidence tab)", "")],
        [("Cost-weighted reduction (×)", ""), (x1(reduction_factor(g)), "b"), ("derived", "b"), ("= 1 / (mem_share/mem_factor + (1−mem_share)/flop_factor). Amdahl blend.", "")],
        [("Energy / opex reduction (×)", ""), (x1(energy_reduction(g)), ek[1]), (ek[0], ek[1]), ("= cost-weighted reduction by default (energy splits memory/compute like cost); override in sidebar", "")],
        [("Discount rate", ""), (pct(g["discount_rate"]), "y"), ("assumption", "y"), ("Perpetuity capitalization rate; set to your WACC (10% → ×10)", "")],
        [("Fully-loaded $/GPU", ""), (usd0(g["gpu_cost"]), "y"), ("assumption", "y"), ("B200-class GPU (~$40k) + share of server, NVLink, networking", "")],
        [("Wall power / GPU", ""), (f"{g['wall_power_kw']:.1f} kW", "y"), ("assumption", "y"), ("≈1 kW TDP × PUE ~1.3 + node overhead", "")],
        [("Electricity rate", ""), (f"${g['elec_rate']:.2f}/kWh", "y"), ("assumption", "y"), ("Datacenter wholesale ~$0.06–0.10/kWh", "")],
        [("Cooling / ops overhead", ""), (pct(g["cooling_overhead"]), "y"), ("assumption", "y"), ("Non-power running cost as a fraction of electricity", "")],
        [("Fleet life", ""), (f"{g['fleet_life_yr']:.0f} yr", "y"), ("assumption", "y"), ("AI-GPU depreciation life; filings say 5–6 yr (we use 4, conservative)", "")],
        [("Datacenter scaling factor", ""), (pct(g["dc_scale"]), "y"), ("toggle", "y"), ("Share of non-accelerator DC that also shrinks. 0 = conservative; ~0.7 ≈ breakeven; 1 = positive", "")],
        [("Named share of global AI capex", ""), (pct(g["named_share_of_global"]), "y"), ("assumption", "y"), ("Named firms' share of worldwide AI capex; remainder grossed up pro-rata", "")],
        [("SpaceX market cap", ""), (n0(g["spacex_mktcap"]), "g"), ("data", "g"), ("Market (IPO 2026-06-12 ~$1.77T)", "")],
        [("Per-company total capex (FY25)", ""), ("disclosed", "g"), ("data", "g"), ("10-K / earnings calls — see each company tab + its Sources", "")],
        [("Per-company FY26 capex", ""), ("estimate", "y"), ("assumption", "y"), ("Management guidance midpoint — see each company tab", "")],
        [("Per-company infra / server / accel", ""), ("estimate", "y"), ("assumption", "y"), ("CFO commentary (infra/server) + BOM teardown (accel ~67–80%)", "")],
        [("Per-company AI revenue", ""), ("mixed", "y"), ("assumption", "y"), ("Disclosed run-rates where available (MSFT $37B, AMZN $15B); else estimate", "")],
    ]
    show_table(["Value / driver", "Current", "Kind", "How it's derived / source"], rows,
               widths={"Value / driver": "medium", "How it's derived / source": "large"})

    st.markdown(r"""
### Methodology & sources

**Engine.** A GPU is ~60% memory / ~40% compute by cost. Cutting memory ×100 and FLOPs ×10 leaves a
residual of ~0.6% + 4% → **~22× cost-weighted reduction** (Amdahl — floored by the least-reduced
component, compute). 1000× (=100×·10×) is *not* physical: cost is additive, not multiplicative.

**Per company.** `total capex (disclosed) × infra share × server share × accelerator share`
→ accelerator capex → fleet → energy/opex → efficient version → value (FY2025 actual + FY2026 estimate).
Infra share strips non-AI (Amazon = AWS ~68%); server share is CFO-disclosed; accelerator-within-server
is ~67–80% from BOM teardowns.

**Totals & global.** The named firms roll up live (no double-count). The **GLOBAL** row grosses the
named total up to a worldwide estimate using *Named share of global AI capex* (the rest = other clouds,
China, neoclouds, xAI, sovereign & enterprise).

**Net AI economics** are cash basis: `AI revenue − AI capex − AI opex`; with the architecture, add the
spend cut. All six firms lose money on AI today. The *Datacenter scaling factor* toggles how much of the
non-accelerator datacenter shrinks too (0 = conservative; ~0.7 ≈ breakeven; 1 = flips positive).

**Key results (defaults).** Cost-weighted reduction ~22×. FY25: ~\$370B AI capex vs ~\$79B AI revenue →
~−\$295B/yr burn; spend cut ~\$159B → burn ~−\$136B (~\$1.6T capitalized). Global est ~\$2.0T (FY25), ~\$4.1T (FY26).

**Caveats.** AI revenue is the softest input (Microsoft \$37B & Amazon \$15B run-rates disclosed; the rest
estimated; Meta's real payoff is indirect ad-uplift). Totals are disclosed; server/accelerator splits are
estimated (±15–20%). Capitalization is a simple perpetuity (benefit ÷ discount rate). Analytical estimate,
not investment advice.

**Sources.** SEC filings & earnings calls (MSFT, GOOGL, AMZN, META, ORCL 10-Ks/transcripts; SpaceX S-1);
BOM/margin teardowns (Silicon Analysts); TPU/Trainium TCO (SemiAnalysis); GPU rental pricing (Spheron).
Per-company source links are on each company tab.
""")


# ---- main ----------------------------------------------------------------------
st.title("AI Capex Efficiency")
st.caption("Interactive mirror of the workbook — the \\$ value of cutting AI memory ~100× and FLOPs ~10× "
           "(~22× cost-weighted) across the 6 largest AI-capex spenders + a global estimate. "
           "🟡 assumption · 🟢 disclosed data · 🔵 derived.")

g = sidebar_globals()
names = ["Totals"] + [c["name"] for c in COMPANIES] + ["Inputs", "Sensitivity", "CostLadder", "Evidence", "Methodology"]
T = st.tabs(names)
nco = len(COMPANIES)
comps = [dict(c) for c in COMPANIES]

# company tabs first so their edits are captured before the Totals roll-up computes
for i, c in enumerate(comps):
    with T[1 + i]:
        company_tab(c, g)
with T[0]:
    totals_tab(comps, g)
with T[1 + nco]:
    inputs_tab(g)
with T[2 + nco]:
    sensitivity_tab(comps, g)
with T[3 + nco]:
    costladder_tab(g)
with T[4 + nco]:
    evidence_tab(g)
with T[5 + nco]:
    methodology_tab(g)
