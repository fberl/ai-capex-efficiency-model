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
                            compute_company, compute_year, global_estimate,
                            SCENARIOS, DEFAULT_SCENARIO, DEPLOYMENT_SCALES,
                            DEFAULT_DEPLOYMENT_SCALE, KERNEL_OPERATING_POINTS,
                            DEFAULT_KERNEL_POINT, QUALITY_FIT_20260814,
                            PARAM_MATCHING_PCT_20260814, SERVING_STATE_20260814,
                            STEP_GAP_20260814, TRAINING_SCALING_20260814,
                            SERVING_BLEND_20260814, DECODE_THROUGHPUT_20260814,
                            param_matching_fraction, param_matching_gain,
                            serving_state_ratio, decode_throughput_ratio, compute_lever)

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

    s.subheader("Scenario")
    scale_labels = {lab: n for n, lab in DEPLOYMENT_SCALES}

    def _scenario_flop():
        """FLOPs lever implied by the current scenario + scale + kernel point."""
        name = st.session_state.get("scenario", DEFAULT_SCENARIO)
        spec = SCENARIOS[name]
        if not spec["dynamic"]:
            return spec["flop_factor"]
        n_tf = scale_labels[st.session_state.get("scale_label", _default_scale_label)]
        kf = KERNEL_OPERATING_POINTS[st.session_state.get("kernel_point", DEFAULT_KERNEL_POINT)]
        return compute_lever(n_tf, kf)

    _default_scale_label = next(lab for n, lab in DEPLOYMENT_SCALES
                                if n == DEFAULT_DEPLOYMENT_SCALE)
    if "scenario" not in st.session_state:
        st.session_state["scenario"] = DEFAULT_SCENARIO
        st.session_state["scale_label"] = _default_scale_label
        st.session_state["kernel_point"] = DEFAULT_KERNEL_POINT
        st.session_state["flop_factor"] = _scenario_flop()

    def _apply_scenario():
        st.session_state["flop_factor"] = _scenario_flop()

    s.radio("Scenario (sets the FLOPs lever)", list(SCENARIOS), key="scenario",
            on_change=_apply_scenario)
    dynamic = SCENARIOS[st.session_state["scenario"]]["dynamic"]
    if dynamic:
        s.select_slider("🟡 Deployment scale (transformer params)", options=list(scale_labels),
                        key="scale_label", on_change=_apply_scenario,
                        help="The parameter-matching curve is a projection of two OLS fits over "
                             "4 measured rungs each (47M–663M params). Everything past ~1B is "
                             "extrapolation.")
        s.selectbox("🟡 Kernel operating point (measured)", list(KERNEL_OPERATING_POINTS),
                    key="kernel_point", on_change=_apply_scenario,
                    help="The measured transformer-cost / bAttention-cost ratio at equal size. "
                         "Above 1 we are ahead; below 1 we are behind.")
        _n = scale_labels[st.session_state["scale_label"]]
        _frac = param_matching_fraction(_n)
        _kf = KERNEL_OPERATING_POINTS[st.session_state["kernel_point"]]
        s.caption(
            f"**FIT-DERIVED** — equal fitted quality at **{_frac:.1%}** of the transformer's "
            f"parameters → ×{param_matching_gain(_n):.2f} compute per token at equal quality "
            f"(ln(bpb) = a + b·ln N, 4 rungs per family, fits cross at 392M).  \n"
            f"**MEASURED** — ×{_kf:.2f} cost ratio at the selected operating point.  \n"
            f"→ FLOPs lever = ×{param_matching_gain(_n):.2f} × ×{_kf:.2f} = "
            f"**×{param_matching_gain(_n) * _kf:.2f}**. Editing the numbers below overrides it."
        )
    else:
        s.caption(SCENARIOS[st.session_state["scenario"]]["blurb"] +
                  " Editing the numbers below overrides the preset.")
    s.subheader("Architecture")
    g["mem_factor"] = gnum("🟡 Memory reduction (×)", "mem_factor", 1, 400, 5, "%.0f")
    ctx_labels = {f"{v:,}": v for v in (4096, 16384, 65536, 262144, 1048576)}
    st.session_state.setdefault("serve_ctx", "65,536")
    _ctx = ctx_labels[s.select_slider("Serving context for the memory read-out (tokens)",
                                      options=list(ctx_labels), key="serve_ctx")]
    s.caption(
        f"**MEASURED, full model** ({SERVING_STATE_20260814['config']}) — the transformer carries "
        f"{SERVING_STATE_20260814['tf_kv_mb_per_token_per_stream'] * 1000:.0f} KB of KV per token of "
        f"context per stream; our decode state is a constant "
        f"{SERVING_STATE_20260814['battn_state_mb_per_stream']:.1f} MB (h **and** M memory — the "
        f"{SERVING_STATE_20260814['battn_state_mb_per_stream_h_carry_proxy']:.2f} MB this model used to "
        f"quote was an h-carry-only proxy). At {_ctx:,} tokens that is "
        f"×{serving_state_ratio(_ctx):,.0f} less serving state "
        f"(×{serving_state_ratio(_ctx, kv_compression=8.0):,.0f} even granting the transformer "
        f"paged attention + KV ÷8). The ÷{g['mem_factor']:.0f} lever above is left far below the "
        f"measured ratio on purpose — past ~100× the memory term stops moving the Amdahl blend."
    )
    g["flop_factor"] = gnum("🟡 FLOPs reduction (×)", "flop_factor", 0.1, 200, 0.5, "%.2f",
                            help="Below 1 means compute is WORSE, which is the honest reading at "
                                 "short context — see the 'Single-GPU training step' operating point.")
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
        ("Memory reduction factor", f"{g['mem_factor']:.0f}×", "y", "MEASURED serving-state ratio is far larger (see Evidence); the lever is capped low on purpose — past ~100× it stops moving the blend"),
        ("FLOPs reduction factor", f"{g['flop_factor']:.2f}×", "y", "2026-08-14 scenario = FIT-DERIVED parameter-matching gain × MEASURED kernel cost ratio; earlier presets ×4 measured / ×10 ceiling"),
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

    section("2026-08-14 sealed scaling refit — the parameter-matching curve (FIT-DERIVED)")
    tp = QUALITY_FIT_20260814["top_pair"]
    show_table(["Transformer scale", "bAttention params for equal fitted quality", "As % of transformer", "Compute per token at equal quality"],
               [[(lab, ""), (f"{param_matching_fraction(n) * n:,.0f}", "b"),
                 (f"{param_matching_fraction(n):.1%}", "b"), (x1(param_matching_gain(n)), "b")]
                for n, lab in DEPLOYMENT_SCALES],
               widths={"Transformer scale": "medium", "bAttention params for equal fitted quality": "medium"})
    st.caption(
        f"ln(bpb) = a + b·ln(params), OLS per family — bAttention a={QUALITY_FIT_20260814['battn']['a']:.4f} "
        f"b={QUALITY_FIT_20260814['battn']['b']:.4f} over {QUALITY_FIT_20260814['battn']['rungs']} rungs; "
        f"transformer a={QUALITY_FIT_20260814['tf']['a']:.4f} b={QUALITY_FIT_20260814['tf']['b']:.4f} over "
        f"{QUALITY_FIT_20260814['tf']['rungs']} rungs. The fits cross at "
        f"{QUALITY_FIT_20260814['crossover_params'] / 1e6:.0f}M params (95% CI "
        f"{QUALITY_FIT_20260814['crossover_ci95'][0] / 1e6:.0f}M–{QUALITY_FIT_20260814['crossover_ci95'][1] / 1e6:.0f}M). "
        "**This is a projection of two fits, not a measurement** — the rungs span 47M–663M params, so every row "
        "past 1B is extrapolation. The measured top pair: bAttention reaches "
        f"{tp['battn_bpb']:.4f} bpb on {tp['battn_params']:,} params against the transformer's {tp['tf_bpb']:.4f} bpb "
        f"on {tp['tf_params']:,} — {abs(tp['param_pct']):.1f}% fewer parameters, {tp['bpb_pct']:.2f}% behind on quality. "
        f"Receipt: `{QUALITY_FIT_20260814['receipt']}`."
    )

    section("2026-08-14 serving state (MEASURED, full model) — theirs grows per token, ours does not")
    ss = SERVING_STATE_20260814
    show_table(["Context (tokens)", "Transformer KV / stream", "bAttention state / stream", "Ratio", "Granting the transformer KV ÷8"],
               [[(f"{c:,}", ""), (f"{ss['tf_kv_mb_per_token_per_stream'] * c / 1000:,.1f} GB", "b"),
                 (f"{ss['battn_state_mb_per_stream']:.1f} MB", "b"),
                 (f"×{serving_state_ratio(c):,.0f}", "b"),
                 (f"×{serving_state_ratio(c, kv_compression=8.0):,.0f}", "b")]
                for c in (4096, 16384, 65536, 262144, 1048576)])
    st.caption(
        f"{ss['config']}. {ss['tf_kv_mb_per_token_per_stream'] * 1000:.0f} KB of KV per token of context per stream, "
        f"linear in context; {ss['tf_measured_point']}. Ours is constant in context and is the FULL recurrent "
        f"state — {ss['battn_state_scope']} — so {ss['battn_measured_point']}. "
        f"Receipt: `{ss['receipt']}`. The model's memory lever stays at "
        f"÷{g['mem_factor']:.0f} — orders of magnitude below the measured ratio — because the Amdahl blend stops "
        "responding to the memory term long before that."
    )

    section("2026-08-14 decode (MEASURED) — a memory-ceiling lever, not a latency lever")
    dt = DECODE_THROUGHPUT_20260814
    show_table(["Context (tokens)", "Transformer (tokens/s per GPU)", "bAttention (tokens/s per GPU)", "Decode lever"],
               [[(f"{c:,}" + ("  ← measured cell" if c in (32768, 262144) else ""), ""),
                 (f"{dt['tf_anchor_tokens_per_s_per_gpu'] * dt['tf_anchor_ctx'] / c:,.1f}", "b"),
                 (f"{dt['battn_tokens_per_s_per_gpu']:,.1f}", "b"),
                 (f"×{decode_throughput_ratio(c):,.2f}", "b")]
                for c in (4096, 16384, 32768, 65536, 262144, 1048576)])
    st.caption(
        "**Read this one carefully — it is the number that moved.** Per generated token at a single stream the "
        "transformer is FASTER than us at 64k context (~4.9 ms of GPU-busy against our ~5.7 ms), and the old "
        "×15.9 decode lever rested on an inflated transformer per-token cost that is now retired. What the "
        "transformer cannot do is hold many long-context streams on one card: it re-reads its whole KV cache "
        "every token, so once the card is full its tokens per second falls as 1/context. Ours is context-flat. "
        f"Two measured cells pin it — at 32,768 tokens the transformer manages 8 streams and OOMs at 16 "
        f"(×{dt['measured_ratio_32k']:.2f}); at 262,144 it manages 1 and OOMs at 2 (×{dt['measured_ratio_262k']:.2f}). "
        "The 1/context law anchored on the first reproduces the second to 0.006%, so the rows between them are "
        f"arithmetic. The lever crosses 1 at ~{SERVING_BLEND_20260814['crossover_ctx']:,} tokens: **below that the "
        f"transformer serves more tokens per GPU-second than we do.** {dt['conservatism']} Receipt: `{dt['receipt']}`."
    )

    section("The counterweight (MEASURED) — where the compute cost runs AGAINST us")
    sg, tr = STEP_GAP_20260814, TRAINING_SCALING_20260814
    show_table(["Measurement", "bAttention", "Transformer", "Reading"], [
        [("Training step, one GPU, 2,048-token context", ""), (f"{sg['battn_ms_per_step']:,.0f} ms/step", "b"),
         (f"{sg['tf_ms_per_step']:,.0f} ms/step", "b"), (f"×{sg['gap_against_battn']:.2f} AGAINST us", "b")],
        [("Training scaling to 8 GPUs (own 1-GPU baseline)", ""), (f"×{tr['battn_x8_best_load']:.2f}", "b"),
         (f"×{tr['tf_x8_best_load']:.2f}", "b"), ("in our favour", "b")],
        [("Decode, per token, ONE stream at 64k context", ""), ("~5.74 ms", "b"), ("~4.92 ms", "b"),
         ("×0.86 AGAINST us", "b")],
        [("Serving blend at 64k context, 10:1 input:output", ""), ("—", ""), ("—", ""),
         (f"×{SERVING_BLEND_20260814['blend']:.2f} in our favour", "b")],
    ], widths={"Measurement": "large"})
    st.caption(
        f"The step gap is real and is carried here on purpose: {sg['config']}. {sg['note']}. "
        f"At equal quality it partly cancels — a trillion-parameter-scale deployment needs "
        f"{param_matching_fraction(1e12):.1%} of the parameters, so the same step costs "
        f"×{sg['gap_against_battn'] * param_matching_fraction(1e12):.2f} rather than ×{sg['gap_against_battn']:.2f}. "
        f"The serving blend decomposes as prefill ×{SERVING_BLEND_20260814['prefill_ratio']:.2f} and decode "
        f"×{SERVING_BLEND_20260814['decode_ratio']:.2f}, with decode "
        f"{SERVING_BLEND_20260814['tf_decode_share_of_cost']:.1%} of transformer serving cost. "
        f"**{SERVING_BLEND_20260814['caveat']}.** It supersedes a ×"
        f"{SERVING_BLEND_20260814['supersedes']['blend']:.2f} blend: "
        f"{SERVING_BLEND_20260814['supersedes']['why']}. Downside row — "
        f"{SERVING_BLEND_20260814['sensitivity_kv8']['note'][0].lower()}"
        f"{SERVING_BLEND_20260814['sensitivity_kv8']['note'][1:]} "
        f"Receipts: `{sg['receipt']}`, `{tr['receipt']}`."
    )

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
        [("Memory reduction (×)", ""), (f"{g['mem_factor']:.0f}×", "y"), ("assumption", "y"), ("RNN O(1) state vs transformer O(T) KV cache. MEASURED 2026-08-14 (d2048/L24 bf16, GH200, full model): 197 KB of KV per token of context per stream against a constant 25.4 MB state carry — ×508 at 64k context, ×2,032 at 262k. ÷100 is a deliberate cap, not the measurement", "")],
        [("FLOPs reduction (×)", ""), (f"{g['flop_factor']:.2f}×", "y"), ("assumption", "y"), ("2026-08-14 scenario = FIT-DERIVED parameter-matching gain (equal fitted quality at 23.7% of the parameters at 1T scale → ×4.22) × MEASURED kernel cost ratio at the chosen operating point. Earlier presets: ×4 measured prefill (2026-07-21), ×10 ceiling", "")],
        [("Compute cost that runs AGAINST us", ""), (f"×{STEP_GAP_20260814['gap_against_battn']:.2f}", "y"), ("measured", "b"), ("At 2,048-token context on one GPU a training step costs ×6.46 MORE (8,979 vs 1,389 ms/step, d1536/27L, fp16, 601-step protocol). Selectable as a kernel operating point in the sidebar; at equal quality it falls to ×1.53", "")],
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

**Quality-matched tier (default, 2026-08-14).** The sealed scaling refit adds a *quality* axis the
earlier tiers did not have. Both families now have four measured ladder rungs (47M–663M params) fitted
as `ln(bpb) = a + b·ln(params)`; the fits cross at **392M params**, and above the crossing bAttention
reaches the same fitted quality on fewer parameters — **84.2% at 1B, 55.2% at 10B, 36.2% at 100B,
23.7% at 1T, 15.5% at 10T**. Per-token cost is ~linear in parameters, so 23.7% of the parameters is a
**×4.22 compute-per-token advantage at equal quality** at trillion-parameter scale. That factor is
**FIT-DERIVED — a projection of two fits, not a measurement**, and every point past ~1B extrapolates
beyond the measured rungs. It multiplies the **MEASURED** cost ratio at the chosen kernel operating
point (default: the ×2.19 serving blend at 64k context, 10:1 input:output) to give the FLOPs lever.

**The decode lever was re-based on 2026-08-14, and it fell.** A full-model bf16 decode measurement on a
GH200 (d2048/24 layers, both families, same session) retired two numbers this model used to carry: a
transformer decode cost of 87.9 ms/token that a growing-KV re-planning artifact in the old harness had
inflated, and a bAttention serving-state figure of 0.15 MB/stream that was an h-carry-only proxy with no
M memory in it. The honest per-token picture is that at 64k context on a single stream the **transformer
decodes a token faster than we do** (~4.9 ms of GPU-busy against our ~5.7 ms). What survives — and what
the decode lever now means — is a **memory ceiling**: the transformer re-reads its entire KV cache for
every token it emits, so once a card is full of KV its aggregate throughput falls as 1/context, while
ours is flat. Measured on one GPU: at 262,144 tokens the transformer fits **one** stream (a second OOMs)
and serves 64 tokens/s, while bAttention holds **64** streams in 1.6 GB and serves 562 tokens/s — an
**×8.8 aggregate advantage**. At 32,768 tokens the same comparison is ×1.1, and the lever crosses 1 near
**30,000 tokens**: below that context the transformer serves more tokens per GPU-second than we do. The
64k figure (×2.20) is interpolated on that 1/context law, which reproduces the far measured cell to
0.006%. The old ×15.9 decode lever, the "×53 decode" line and the "512 concurrent streams" figure are
all retired.

**The measurement that runs against us.** At 2,048-token context on a single GPU a bAttention training
step costs **×6.46 more** than the transformer's (8,979 vs 1,389 ms/step, d1536/27 layers, fp16,
601-step protocol). It is selectable as a kernel operating point in the sidebar, where it drives the
FLOPs lever below 1. At equal quality it shrinks to ×1.53 but does not vanish. The serving claim above
excludes training for exactly this reason.

**Memory (2026-08-14, MEASURED, full model).** At d2048/24 layers in bf16 the transformer carries **197 KB
of KV per token of context per stream** (51.5 GB for one 262,144-token stream); our state carry is a
constant **25.4 MB/stream** — the *whole* recurrent cell, h and M memory, not the 0.15 MB h-carry proxy
this model used to quote. That is ×508 less serving state at 64k context, ×2,032 at 262k and ×8,128 at
1M — or ×64, ×254 and ×1,016 granting the transformer paged attention and KV ÷8. The model's memory
lever stays at **÷100** regardless: past ~100× the memory term is 0.6% of residual cost and stops moving
the Amdahl blend, so raising it would change the headline multiple without changing the money.

**Earlier tiers, kept for comparison.** ×4 measured long-context prefill (2026-07-21, parameter-matched,
bf16, 1M tokens) → ~9.4× cost-weighted; ×10 ceiling → ~22×; compute parity ×1 → ~2.5×.

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

**Key results.** FY25: ~\$370B AI capex vs ~\$79B AI revenue → ~−\$295B/yr burn. Quality-matched tier at
1T scale (~20.3×): spend cut ~\$159B → burn ~−\$136B (~\$1.6T capitalized; global est ~\$2.0T FY25,
~\$4.1T FY26). 2026-07-21 measured tier (~9.4×): cut ~\$149B → burn ~−\$146B (~\$1.5T; global ~\$1.9T /
~\$3.8T). Ceiling (~22×): cut ~\$159B → burn ~−\$136B (~\$1.6T; global ~\$2.0T / ~\$4.1T).

**Why the money barely moves — and the 2026-08-14 re-base proves it.** The cut is
`accelerator capex × (1 − 1/reduction)`, which saturates. Re-basing the decode lever on the new receipts
cut the compute lever **×50.33 → ×9.24, a factor of 5.4**, and the FY25 headline moved **\$165B → \$159B,
−3.6%**. Every plausible wiring of the levers — the parameter-matching gain alone (×4.22 → 9.9×
cost-weighted), the serving blend alone (×2.19 → 5.3×), or the two composed (×9.24 → 20.3×) — lands the
FY25 cut between ~\$135B and ~\$159B. Even the harshest downside row, granting the transformer an
idealized KV÷8 stack so the blend falls to ×0.28 and the whole compute lever to ×1.18, only takes the cut
to ~\$109B. Past a compute lever of ~10× the headline is set by the deliberately conservative ÷100 memory
cap, not by the compute receipts. Doubling the multiple is worth a few \$B; that is the honest shape of
the result and the reason the model refuses multiplicative headlines.

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
st.caption("Interactive mirror of the workbook — the \\$ value of the architecture advantage across the "
           "6 largest AI-capex spenders + a global estimate. Default scenario (**2026-08-14 sealed refit**): "
           "memory ÷100 (a deliberate cap — the measured full-model serving-state ratio is ×508 at 64k "
           "context) and a FLOPs lever built as **FIT-DERIVED** equal-quality parameter matching × "
           "**MEASURED** serving-throughput ratio. The measured factor is memory-ceiling driven — how many "
           "concurrent streams a GPU holds — not a per-token latency win. Earlier scenarios are kept in the "
           "sidebar. "
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
