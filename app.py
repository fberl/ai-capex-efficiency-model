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

from ai_capex_model import (GLOBALS, COMPANIES, MEASURED, SERVING,
                            SERVING_TRAINING_ASSUMPTIONS, PREFILL_BF16_D2048_SWEEP,
                            WORKLOAD,
                            workload_compute_advantage, reduction_factor,
                            energy_reduction, compute_company, compute_year,
                            global_estimate, serving_economics, serving_cost_curve,
                            training_throughput_ratio, QUALITY_FIT_20260814,
                            STEP_GAP_20260814, COMPUTE_LEVER_20260814,
                            DECK_DEPLOYMENT_SCALE, param_matching_fraction,
                            param_matching_gain, CEILING_FLOP_LEVER,
                            CEILING_PREFILL_SPEEDUP)

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

    s.subheader("Workload")
    st.session_state.setdefault("in_out_ratio", float(WORKLOAD["in_out_ratio"]))
    st.session_state.setdefault("context_tokens", int(WORKLOAD["context_tokens"]))
    st.session_state.setdefault("train_share", float(WORKLOAD["train_share"]))
    wl = {
        "in_out_ratio": s.number_input(
            "🟡 Input : output token ratio", min_value=0.1, max_value=1000.0, step=1.0,
            key="in_out_ratio", format="%.1f",
            help="Input tokens per generated token. ~10:1 = code/agent traces; 50-100:1 = RAG."),
        "context_tokens": int(s.number_input(
            "🟡 E[context] (tokens)", min_value=1024, max_value=1048576, step=1024,
            key="context_tokens", format="%d",
            help="Expected context length over the workload.")),
        "train_share": s.number_input(
            "🟡 Training share of accelerator cost", min_value=0.0, max_value=1.0, step=0.05,
            key="train_share", format="%.2f",
            help="Fraction of accelerator spend on TRAINING. Default 0 = serving-cost "
                 "claim; training runs against us at short sequence, so raising this "
                 "pulls the blend down."),
    }
    adv = workload_compute_advantage(wl)
    s.caption(
        f"→ prefill **×{adv['prefill_ratio']:.2f}** · decode **×{adv['decode_ratio']:.1f}** "
        f"· blended **×{adv['blended']:.2f}**  \n"
        f"Prefill is {adv['tf_prefill_share'] * 100:.1f}% of transformer serving cost here."
    )

    s.subheader("Scenario")
    scale_opts = {
        "1B": 1_000_000_000,
        "10B": 10_000_000_000,
        "100B": 100_000_000_000,
        "1T": 1_000_000_000_000,
        "10T": 10_000_000_000_000,
    }
    st.session_state.setdefault("deploy_scale", "1T")
    scale_label = s.select_slider(
        "🟡 Deployment scale (transformer params)",
        options=list(scale_opts), key="deploy_scale",
        help="Sets the equal-quality parameter ratio. The fits are measured on "
             "47M–663M rungs, so everything past ~1B is a projection.")
    _scale = scale_opts[scale_label]
    _gain = param_matching_gain(_scale)
    # The flop_factor widget floors at 1.0, so the preset must too (the raw
    # value can drop below 1 at high train_share / short context).
    _today = max(1.0, float(adv["blended"]) * _gain)
    TODAY_LABEL = "Today — quality-matched at the scale above"
    CEIL_LABEL = "Ceiling — prefill kernels mature (~×22)"
    scenarios = {TODAY_LABEL: _today, CEIL_LABEL: float(CEILING_FLOP_LEVER)}
    if "scenario" not in st.session_state:
        st.session_state["scenario"] = TODAY_LABEL
        st.session_state["flop_factor"] = scenarios[TODAY_LABEL]

    def _apply_scenario():
        st.session_state["flop_factor"] = scenarios[st.session_state["scenario"]]

    s.radio("Scenario (sets the FLOPs lever)", list(scenarios), key="scenario",
            on_change=_apply_scenario)
    # Today is computed from the sliders above, so it has to track them on every
    # rerun, not only when the radio changes. Safe to write here because the
    # flop_factor widget is created further down.
    if st.session_state.get("scenario") == TODAY_LABEL:
        st.session_state["flop_factor"] = float(_today)
    _floor_note = (" (raw ×%.2f floored at ×1)" % (adv["blended"] * _gain)
                   if adv["blended"] * _gain < 1.0 else "")
    s.caption(
        f"**Today** = measured serving blend ×{adv['blended']:.2f} × equal-quality "
        f"parameter ratio ×{_gain:.2f} at {scale_label} (fit-derived) = **×{_today:.2f}**"
        f"{_floor_note} — tracks the sliders above. **Ceiling** = measured prefill ×4.02 × "
        f"{CEILING_PREFILL_SPEEDUP:.1f} kernel-campaign speedup = **×{CEILING_FLOP_LEVER:.1f}**; "
        f"under Ceiling the FLOPs number below can be edited freely. The two differ only ~3% in "
        f"dollars **by design**: the cut is accel × (1−1/R) and saturates, so the money is mostly "
        f"captured at ~20× — the scenarios differ in the reduction (×{reduction_factor(dict(g, flop_factor=_today)):.0f} "
        f"vs ×{reduction_factor(dict(g, flop_factor=CEILING_FLOP_LEVER)):.0f}) and the residual compute "
        f"bill, not the headline. Derivation: Methodology tab."
    )
    s.subheader("Architecture")
    g["mem_factor"] = gnum("🟡 Memory reduction (×)", "mem_factor", 1, 400, 5, "%.0f")
    g["flop_factor"] = gnum("🟡 FLOPs reduction (×)", "flop_factor", 1, 100, 0.01, "%.2f")
    g["mem_share"] = gnum("🟡 Memory share of GPU cost", "mem_share", 0.0, 1.0, 0.05, "%.2f")
    auto_energy = s.checkbox("Auto-derive energy reduction (= cost reduction)", value=True,
                             help="Energy splits memory/compute like cost does. Uncheck to set manually.")
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
        [("Annual opex ($M)", ""), (n0(fb25["ann_m"]), "b"), (n0(fb26["ann_m"]), "b"), ("= MWh × rate × (1+overhead) × 365", "")],
        [("Lifetime opex ($B)", ""), (n1(fb25["life_b"]), "b"), (n1(fb26["life_b"]), "b"), ("× fleet life", "")],
    ], widths=W)

    section("Efficient version & value")
    show_table(CB, [
        [("Efficient AI capex ($B)", ""), (n1(d25["ai_capex"] - d25["capex_avoided"]), "b"), (n1(d26["ai_capex"] - d26["capex_avoided"]), "b"), ("= AI capex − avoided", "")],
        [("Capex avoided/yr ($B)", ""), (n1(d25["capex_avoided"]), "b"), (n1(d26["capex_avoided"]), "b"), ("accel(+DC×dc_scale) × (1−1/reduction)", "")],
        [("Annual opex savings ($M)", ""), (n0(d25["opex_saved"] * 1000), "b"), (n0(d26["opex_saved"] * 1000), "b"), ("", "")],
        [("Sustained annual benefit ($B/yr)", ""), (n1(d25["spend_cut"]), "b"), (n1(d26["spend_cut"]), "b"), ("= avoided + opex savings", "")],
        [("Capitalized value ($B)", ""), (n0(d25["capitalized"]), "b"), (n0(d26["capitalized"]), "b"), ("= benefit / discount rate", "")],
        [("% of market cap", ""), (f"{d25['capitalized'] / c['mcap']:.1%}" if c["mcap"] else "—", "b"), (f"{d26['capitalized'] / c['mcap']:.1%}" if c["mcap"] else "—", "b"), ("", "")],
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
        ("Memory reduction factor", f"{g['mem_factor']:.0f}×", "y", "O(1) state vs O(T) KV cache — MEASURED 2026-08-14 (full model): 64 concurrent 262k streams on one GPU in 1.62 GB against the transformer's 1 stream at 51.5 GB, a 2nd OOMs; 25.4 MB of state per stream vs 197 KB of KV per token of context — ÷100 is conservative"),
        ("FLOPs reduction factor", f"{g['flop_factor']:.2f}×", "y", "TODAY (default) = measured serving blend ×2.19 × fit-derived equal-quality parameter ratio at the sidebar's deployment scale = ×9.24 at the 1T default. CEILING = measured prefill ×4.02 (bf16 parameter-matched, d512/1M, 2026-07-21) × 5.5 kernel-campaign speedup = ×22.1"),
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
        ("SpaceX market cap ($B)", n0(g["spacex_mktcap"]), "g", "market data ~$1.84T Aug 2026 (IPO 2026-06-12 at ~$1.77T)"),
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
    section("Sensitivity (SpaceX) — Today vs Ceiling vs live cost-weighted")
    sx = next(c for c in comps if c["name"] == "SpaceX")
    d = compute_company(sx, g, "fy25")
    accel, opx, disc, mcap = d["accel"], d["opex_saved"] * 1000, g["discount_rate"], g["spacex_mktcap"]
    r_today = reduction_factor(dict(g, flop_factor=COMPUTE_LEVER_20260814))
    r_ceil = reduction_factor(dict(g, flop_factor=CEILING_FLOP_LEVER))
    tiers = [(f"Today {r_today:.0f}×", r_today), (f"Ceiling {r_ceil:.0f}×", r_ceil),
             (f"Cost-weighted (live) {reduction_factor(g):.0f}×", reduction_factor(g))]
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


# ---- serving & training tab (measured 2026-08-07/08) ---------------------------
def serving_training_tab(g):
    st.caption("The 2026-08-07/08 multi-GPU receipts (2/4/8×H100), turned into $ on this model's "
               "existing $/GPU-hour ladder. Every number below traces to a key in "
               "`bdm/docs/deck/build/derived.json` or to an assumption row in the last table. "
               "**Scope**: systems numbers are component-scope (bAttention d1536 fwd+bwd h-recurrence "
               "sub-block vs transformer d2048 forward-only layer); every speedup is each family vs "
               "its **own** 1-GPU baseline, so the scope cancels inside each ratio.")

    m1, m2, m3, m4 = st.columns(4)
    r262 = serving_economics(262144)
    m1.metric("Serving cost ratio @262k ctx", f"{r262['cost_ratio']:.1f}×",
              help="MEASURED per-GPU aggregate decode throughput at each family's own concurrency ceiling "
                   "(2026-08-14 full-model receipt). Granting the transformer an idealized KV÷8 stack: "
                   f"{serving_economics(262144, s={'tf_kv_compression': 8.0})['cost_ratio']:.1f}×.")
    m2.metric("Concurrent 262k streams / GPU", f"{MEASURED['serve_streams_per_gpu']} vs 1",
              help="MEASURED 2026-08-14: 64 bAttention streams in 1.62 GB of state on one GPU; the transformer "
                   "fits ONE 262k stream at 51.5 GB and a second OOMs the card. Aggregate throughput ×8.8.")
    m3.metric("Training throughput @8 GPUs", f"×{training_throughput_ratio():.2f}",
              delta=f"→ ×{training_throughput_ratio(deep=True):.2f} at 256 in flight",
              help="MEASURED matched load (16 in flight both families): ×5.15 fwd+bwd vs ×3.70 forward-only Ulysses best.")
    m4.metric("70B quality parity (PROJECTION)", f"×{MEASURED['parity_70B_param_multiple']:.1f} params",
              delta=f"or ×{MEASURED['parity_70B_token_multiple']:.2f} tokens (β=0.28)", delta_color="off",
              help="Projected from the measured ladder fits; 95% CI ×2.2–×7.7 params, ×1.33–×2.10 tokens. NOT a measurement.")

    section("Serving at long context — $/1M generated tokens, one 8×H100 box")
    rows = []
    for r in serving_cost_curve():
        note = ("bAttention rate extrapolated past 262k (flatness measured 4k–262k)" if r["battn_extrapolated"]
                else ("the measured decode point" if r["ctx"] == 262144 else ""))
        rows.append([
            (f"{r['ctx']:,}", ""),
            (f"${r['battn_usd_per_mtok']:.4f}", "g" if not r["battn_extrapolated"] else "y"),
            (f"${r['tf_usd_per_mtok']:.2f}", "b"),
            (f"{r['tf_streams_per_gpu']}", "b"),
            (f"×{r['cost_ratio']:.1f}", "s"),
            (note, ""),
        ])
    show_table(["Context (tokens)", "bAttention $/1M tok", "Transformer $/1M tok (mature stack)",
                "TF streams/GPU", "Cost ratio", "Note"], rows,
               widths={"Note": "large"})
    st.caption("A memory-ceiling result, not per-token: our decode is context-flat (562 tok/s/GPU — "
               "64 concurrent 262k streams in 1.62 GB) while the transformer's aggregate falls as "
               "1/context once the card is full of KV (197 KB per token of context per stream). "
               "**It is AHEAD below ~30k context**; ×2.2 at 64k, ×8.8 at 262k. Full derivation and "
               "the retired figures: Methodology tab.")

    section("Training at scale — same cluster, more steps/s")
    show_table(["Metric", "bAttention", "Transformer", "Ratio", "Status"], [
        [("8-GPU speedup, matched load (16 in flight)", ""), ("×5.15 (fwd+bwd)", "g"), ("×3.70 (fwd-only, Ulysses best)", "g"), (f"×{training_throughput_ratio():.2f}", "s"), ("MEASURED", "g")],
        [("8-GPU speedup, deep load (256 in flight)", ""), ("×6.27 — pipeline keeps filling", "g"), ("saturated by 16 in flight", "g"), (f"×{training_throughput_ratio(deep=True):.2f}", "s"), ("MEASURED", "g")],
        [("GPU-hours for the same training work", ""), ("−28% (matched) … −41% (deep)", "b"), ("baseline", ""), ("", ""), ("derived", "b")],
        [("64k-token sequence on one 80 GB GPU", ""), ("30.9 GB — fits", "g"), ("OOM (78.4 GB attempted)", "g"), ("", ""), ("MEASURED", "g")],
        [("Pipeline per-GPU peak (flat in load)", ""), ("9.05 GB", "g"), ("43.3 GB (GPipe stage)", "g"), ("×4.8", "b"), ("MEASURED", "g")],
        [("Params for equal quality at 70B", ""), ("1×", ""), ("×4.1 [2.2–7.7]", "y"), ("", ""), ("PROJECTION", "y")],
        [("Tokens for equal quality at 70B (β=0.28)", ""), ("1×", ""), ("×1.7 [1.33–2.10]", "y"), ("", ""), ("PROJECTION", "y")],
    ], widths={"Metric": "large"})
    st.caption("Speedups are each family vs its own 1-GPU baseline (scope cancels); the memory rows are "
               "component-scope and width-unmatched — not model-level claims. The parity rows are "
               "projections from the measured quality-ladder fits (crossover N* ≈ 321M ≈ the top measured "
               "rung; the sealed refit behind the compute lever crosses at 392M — different fit vintages); "
               "β = 0.28 is the Chinchilla assumption, cited as such. Trainability context: stepping "
               f"the recurrence token-by-token costs ×{MEASURED['stepped_vs_scanned']:.0f} vs the scanned "
               "training step (same geometry, same box).")

    section("The compute lever — how the workload mix sets it")
    st.caption("Training, prefill and decode point in opposite directions at today's kernel "
               "maturity, so the lever is blended over the workload rather than taken from one "
               "phase. Cost is accumulated per generated token: `in:out` input tokens through "
               "prefill plus one token through decode, using measured throughput on both sides. "
               "Sidebar controls move this table.")
    _wl_rows = []
    for _r in (1.0, 10.0, 100.0):
        for _c in (8192, 65536, 262144):
            _a = workload_compute_advantage({"in_out_ratio": _r, "context_tokens": _c,
                                             "train_share": 0.0})
            _tone = "s" if _a["blended"] >= 1.0 else "b"
            _note = "" if _a["blended"] >= 1.0 else "transformer ahead"
            if (_r, _c) == (WORKLOAD["in_out_ratio"], WORKLOAD["context_tokens"]):
                _note = "← model default (code / agent regime)"
            _wl_rows.append([
                (f"{_r:.0f}:1", ""), (f"{_c:,}", ""),
                (f"×{_a['prefill_ratio']:.2f}", "b"),
                (f"×{_a['decode_ratio']:.1f}", "g"),
                (f"×{_a['blended']:.2f}", _tone),
                (f"{_a['tf_prefill_share'] * 100:.0f}%", ""),
                (_note, ""),
            ])
    show_table(["Input:output", "E[context]", "Prefill", "Decode", "Blended",
                "Prefill share of TF cost", "Note"], _wl_rows,
               widths={"Note": "medium"})
    st.caption("Decode dominates transformer serving cost at every row, which is why the blend "
               "tracks it. Our decode cost is flat in context and the transformer's is linear, so "
               "the decode advantage is linear in E[context] and mixing contexts carries no "
               "averaging penalty. **Training is excluded**: at T=2048 blocks the transformer is "
               "5.3–9.6× faster (FLOPs are near-matched at 1.07–1.19×; the gap is achieved FLOP/s, "
               "200.9 vs 21.7 TF/s), crossing over near T≈88k. Setting a training share above ~0.06 "
               "takes the blend below 1.")

    section("The prefill lane (16-bit, parameter-matched)")
    st.caption("Measured where both families run their production 16-bit attention kernels — the "
               "lane the competition actually serves in — with parameter counts exactly matched: "
               "1 layer / batch 16, one H100, 2026-07-21. The fp32 full-model sweep is retired to "
               "the Evidence tab as a scope reference.")
    prow = []
    for T, tf_ms, ba_ms in PREFILL_BF16_D2048_SWEEP:
        ratio = tf_ms / ba_ms
        note, tone = ("transformer ahead", "b") if ratio < 1.0 else ("", "g")
        if T == 262144:
            note, tone = "width/context match to the fp32 row below", "s"
        prow.append([(f"{T:,}", ""), (f"{tf_ms:,.1f}", "b"), (f"{ba_ms:,.1f}", "g"),
                     (f"×{ratio:.2f}", tone), (note, "")])
    show_table(["Context (tokens)", "Transformer ms", "bAttention ms", "TF ÷ bAttention", "Note"],
               prow, widths={"Note": "medium"})
    st.caption(f"Shown at d2048. The lever itself is taken at **d512 / 1M context = ×"
               f"{MEASURED['prefill_bf16_tf_battn_d512_1m']:.2f}** "
               f"(d1024/1M = ×{MEASURED['prefill_bf16_tf_battn_d1024_1m']:.2f}; "
               f"d2048/262k = ×{MEASURED['prefill_bf16_tf_battn_d2048_262k']:.2f}). The ratio rises "
               f"with context and falls with width; below the crossover the transformer is faster "
               f"and the model says so. Receipts: "
               f"`experiments/paper_figures/output/matched_d{{512,1024,2048}}_h100.json`.")

    section("Assumptions — one row each (MEASURED / PROJECTION / ASSUMPTION)")
    show_table(["Assumption", "Value", "Status", "Source"],
               [[(a, ""), (b, ""), (c, "g" if c == "MEASURED" else ("y" if c in ("PROJECTION", "ASSUMPTION") else "b")), (d, "")]
                for a, b, c, d in SERVING_TRAINING_ASSUMPTIONS],
               widths={"Assumption": "medium", "Value": "large", "Source": "large"})
    st.caption("Rates unchanged from this model's CostLadder ($2.00–3.50/hr rent band; $2.50 mid). "
               "Production transformer stacks (paged attention, KV quantization) push their wall out — "
               "the 'mature' column already grants that, which is why the headline uses it as the floor.")


# ---- evidence tab --------------------------------------------------------------
def evidence_tab(g):
    section("Architecture receipts behind the two levers")
    show_table(["Lever", "Value", "Receipt", "Device & date", "Scope"], [
        [("Compute (flop_factor) — SOURCE", ""), ("×4.02 at d512 / 1M context", "s"),
         ("experiments/paper_figures/output/matched_d512_h100.json", ""),
         ("1× H100 80GB HBM3, 2026-07-21", ""),
         ("cross-family, bf16 (FlashAttention available to the transformer), params EXACTLY matched, 1 layer / batch 16", "g")],
        [("Compute — same lane, other widths", ""), ("×3.83 d1024/1M · ×2.01 d2048/262k", "b"),
         ("experiments/paper_figures/output/matched_d{1024,2048}_h100.json", ""),
         ("1× H100 80GB HBM3, 2026-07-21", ""),
         ("ratio rises with context, falls with width", "g")],
        [("Compute — fp32 lane", ""), ("×7.91 at d2048/262k", "y"),
         ("experiments/paper_figures/output/deck_speed_scaling_d2048_h100_20260803.json", ""),
         ("1× H100 80GB HBM3, 2026-08-03", ""),
         ("full model (24 layers); no fp32 FlashAttention kernel, owned arm fp16 internally, params 16.7% apart", "y")],
        [("Memory (mem_factor)", ""), ("÷100 (conservative)", "s"),
         ("bdm/docs/deck/build/derived.json — decode_multigpu, slide_decode_262k", ""),
         ("8× H100 box, 2026-08-07/08", ""),
         ("DECODE/serving state: 64 concurrent 262k streams on ONE GPU in 1.62 GB vs the transformer's 1 (a 2nd OOMs); 25.4 MB full-model state per stream vs 51.5 GB of KV. Note prefill activation peak runs the other way (≈2.5× against)", "g")],
        [("16-bit IO (training)", ""), ("×1.185 step time, −17.5% memory", "y"),
         ("experiments/paper_figures/output/receipts/io16_gate_20260809.md", ""),
         ("1× GH200 480GB, 2026-08-09", ""),
         ("same-architecture at d1536; training step time, reported on its own", "y")],
    ], widths={"Receipt": "large", "Scope": "large"})
    st.caption("The compute lever is cross-family in the bf16 lane at block scope; the memory lever "
               "is cross-family at decode/serving scope. Full model in bf16 is the one cell not yet "
               "run.")

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
        [("Memory reduction (×)", ""), (f"{g['mem_factor']:.0f}×", "y"), ("assumption", "y"), ("O(1) state vs O(T) KV cache. MEASURED 2026-08-14, full model (Serving·Training tab): 64 concurrent 262k streams on one GPU in 1.62 GB vs the transformer's 1 at 51.5 GB (a 2nd OOMs); 25.4 MB of state per stream vs 197 KB of KV per token of context — ×2,032 at 262k. ÷100 is conservative", "")],
        [("FLOPs reduction (×)", ""), (f"{g['flop_factor']:.2f}×", "y"), ("assumption", "y"), ("DEFAULT (2026-08-14, the deck's economics slide) = the MEASURED workload blend × the FIT-DERIVED equal-quality parameter ratio. Blend: cross-family bf16 prefill, parameter counts matched, one H100 — ×4.02 at d512/1M (×3.83 at d1024/1M, ×2.01 at d2048/262k; the transformer is ahead at short context) plus the measured decode lane. Parameter ratio: the sealed refit's fits (4 rungs per family) cross at 392M and give the transformer's quality on 23.7% of the params at 1T → ×4.22 — a projection of two fits, not a measurement. Ceiling = measured prefill ×4.02 × 5.5 kernel-campaign speedup = ×22.1; measured TRAINING cluster throughput ×1.39 matched load → ×1.70 deep", "")],
        [("Compute cost that runs AGAINST us", ""), (f"×{STEP_GAP_20260814['gap_against_battn']:.2f}", "y"), ("measured", "b"), (f"At 2,048-token context on one GPU a bAttention training step costs ×{STEP_GAP_20260814['gap_against_battn']:.2f} MORE ({STEP_GAP_20260814['battn_ms_per_step']:,.0f} vs {STEP_GAP_20260814['tf_ms_per_step']:,.0f} ms/step, d1536/27L, fp16, 601-step protocol). At equal quality it falls to ×{STEP_GAP_20260814['gap_against_battn'] * param_matching_fraction(DECK_DEPLOYMENT_SCALE):.2f} but does not vanish. Training is excluded from the serving claim for this reason. Receipt: dtype_trio_v4.json", "")],
        [("Memory share of GPU cost", ""), (pct(g["mem_share"]), "y"), ("assumption", "y"), ("BOM teardown: HBM ~41% + CoWoS ~23% (mostly memory) vs logic die ~9% → ~60/40 (Evidence tab)", "")],
        [("Cost-weighted reduction (×)", ""), (x1(reduction_factor(g)), "b"), ("derived", "b"), ("= 1 / (mem_share/mem_factor + (1−mem_share)/flop_factor). Amdahl blend.", "")],
        [("Energy / opex reduction (×)", ""), (x1(energy_reduction(g)), ek[1]), (ek[0], ek[1]), ("= cost-weighted reduction by default (energy splits memory/compute like cost); override in sidebar", "")],
        [("Discount rate", ""), (pct(g["discount_rate"]), "y"), ("assumption", "y"), ("Perpetuity capitalization rate; set to your WACC (6% → ×16.7)", "")],
        [("Fully-loaded $/GPU", ""), (usd0(g["gpu_cost"]), "y"), ("assumption", "y"), ("B200-class GPU (~$40k) + share of server, NVLink, networking", "")],
        [("Wall power / GPU", ""), (f"{g['wall_power_kw']:.1f} kW", "y"), ("assumption", "y"), ("≈1 kW TDP × PUE ~1.3 + node overhead", "")],
        [("Electricity rate", ""), (f"${g['elec_rate']:.2f}/kWh", "y"), ("assumption", "y"), ("Datacenter wholesale ~$0.06–0.10/kWh", "")],
        [("Cooling / ops overhead", ""), (pct(g["cooling_overhead"]), "y"), ("assumption", "y"), ("Non-power running cost as a fraction of electricity", "")],
        [("Fleet life", ""), (f"{g['fleet_life_yr']:.0f} yr", "y"), ("assumption", "y"), ("AI-GPU depreciation life; filings say 5–6 yr (we use 4, conservative)", "")],
        [("Datacenter scaling factor", ""), (pct(g["dc_scale"]), "y"), ("toggle", "y"), ("Share of non-accelerator DC that also shrinks. 0 = conservative; ~0.7 ≈ breakeven; 1 = positive", "")],
        [("Named share of global AI capex", ""), (pct(g["named_share_of_global"]), "y"), ("assumption", "y"), ("Named firms' share of worldwide AI capex; remainder grossed up pro-rata", "")],
        [("SpaceX market cap", ""), (n0(g["spacex_mktcap"]), "g"), ("data", "g"), ("Market ~$1.84T Aug 2026 (IPO 2026-06-12 at ~$1.77T)", "")],
        [("Per-company total capex (FY25)", ""), ("disclosed", "g"), ("data", "g"), ("10-K / earnings calls — see each company tab + its Sources", "")],
        [("Per-company FY26 capex", ""), ("estimate", "y"), ("assumption", "y"), ("Management guidance midpoint — see each company tab", "")],
        [("Per-company infra / server / accel", ""), ("estimate", "y"), ("assumption", "y"), ("CFO commentary (infra/server) + BOM teardown (accel ~67–80%)", "")],
        [("Per-company AI revenue", ""), ("mixed", "y"), ("assumption", "y"), ("Disclosed run-rates where available (MSFT $37B, AMZN $15B); else estimate", "")],
    ]
    show_table(["Value / driver", "Current", "Kind", "How it's derived / source"], rows,
               widths={"Value / driver": "medium", "How it's derived / source": "large"})

    st.markdown(r"""
### Methodology & sources

**Engine.** A GPU is ~60% memory / ~40% compute by cost. The cost-weighted reduction is Amdahl —
floored by the least-reduced component, compute. At the **Today** defaults (memory ×100, FLOPs ×9.24)
the residual is ~0.6% + 4.3% → **~20×**; the **Ceiling** lever (measured prefill ×4.02 × 5.5
kernel-campaign speedup = ×22.1) gives **~41×**. Multiplying the levers (100×·9.2× ≈ 900×) is *not*
physical: cost is additive, not multiplicative.

**Measured tier (default) — the workload mix.** A single compute number cannot represent training,
prefill and decode: at today's kernel maturity they point in *opposite* directions. Training is a loss
at short sequence (the transformer is ×5.3–×9.6 cheaper per step at T=2048 blocks; the crossover is
near T≈88k). Prefill crosses over near 65k context. Decode, **re-based 2026-08-14**, crosses over near
**30k context** — and only on per-GPU aggregate throughput, never per token. So the lever is computed
from the workload rather than asserted: at the default operating point — **10:1 input:output, 64k
average context, training excluded** — prefill ×1.17 and decode ×2.20 blend to **×2.19**, giving
**~5.3× cost-weighted**. Decode is ~99.7% of transformer serving cost at that point, which is why it
dominates the blend.

That blend compares the two families at **matched size**. The sealed 2026-08-14 refit adds the missing
axis — how big each family has to be for the *same quality*. Four ladder rungs per family (47M–663M
params) fitted as `ln(bpb) = a + b·ln(params)` cross at **392M**, and above the crossing bAttention
reaches the transformer fit's quality on **84.2% of the parameters at 1B, 55.2% at 10B, 36.2% at 100B,
23.7% at 1T, 15.5% at 10T**. Per-token cost is ~linear in parameters, so the default scenario multiplies
the blend by that ratio: ×2.19 MEASURED × ×4.22 **FIT-DERIVED** = ×9.24 → **~20× cost-weighted**. The
receipt's own words for the second factor: *a projection of the two fits, not a measurement* — every
point past ~1B extrapolates beyond the measured rungs.

The counterweight is measured and points the other way: at 2,048-token context on one GPU a bAttention
training step costs **×6.46 more** (8,979 vs 1,389 ms/step, d1536/27 layers, fp16, 601-step protocol).
At equal quality that becomes ×1.53 — smaller, not gone. It is selectable nowhere and hidden nowhere:
the serving claim excludes training precisely because of it.

Both phases use measured throughput, with the transformer granted an idealized mature stack (paged
attention, KV ÷8, bandwidth-floor serving) — strictly more generous than our measured lane. The memory
lever is unchanged at ÷100, grounded in serving-state math (**43–315× smaller serving memory at 1M
tokens**) and measured directly as a concurrency result (below).

**Where this goes negative, stated plainly.** At 100:1 input:output and 8k context the blend is
**×0.28** — the transformer wins. A training share above **~0.06** takes the blend below 1 (0.2 →
**×0.44**). Both are reachable in the sidebar, because a lever you cannot push until it breaks is not
a model.

**One lane: 16-bit.** The lever is measured where both families run their production 16-bit attention
kernels, parameters exactly matched — ×4.02 at d512/1M, ×2.01 at d2048/262k — the lane the competition
actually serves in. 16-bit is also the production training lane (own-baseline gate: ×1.185 step time,
−17.5% peak memory). The fp32 full-model sweep (×7.91 at 262k; no fp32 flash-attention kernel exists)
is kept on the Evidence tab as a scope reference only.

**Decode re-based 2026-08-14 (Serving·Training tab), and it fell.** A full-model bf16 measurement
(d2048/24 layers, both families, one GH200, same session) retired two numbers this model used to
carry: a transformer decode cost of 87.9 per token that a growing-KV re-planning artifact in the old
harness had inflated, and a per-stream serving state of 0.15 MB that was an h-recurrence-carry proxy
with no M memory in it. The honest reading is that **per generated token at a single stream the
transformer is faster than we are at 64k context** (~4.9 ms of GPU-busy against our ~5.7 ms flat).
What survives is a **memory ceiling**: the transformer re-reads its entire KV cache for every token
it emits, so once a card is full of KV its aggregate throughput falls as 1/context while ours is
flat. Measured on one GPU at 262,144 tokens: the transformer fits **one** stream (a second OOMs) and
serves 64 tokens/s, while bAttention holds **64** streams in 1.62 GB and serves 562 tokens/s — an
**×8.8 aggregate advantage**, and that ceiling is exact — a second stream OOMs. At 32,768 tokens the
same comparison is bounded rather than pinned, **×0.97–1.14** (near parity): 8 streams ran and 16
OOMed, so the ceiling there is only bracketed at 8–15, and whatever fits, a 32k decode step re-reads
6.44 GB of KV per stream, capping the transformer near 601 tok/s against our 585.4. The lever crosses
1 near **30,000 tokens**: below that context the transformer serves more tokens per GPU-second than we
do. Full-model state is **25.4 MB/stream** against **197 KB of KV per token of context** — ×508 at
64k, ×2,032 at 262k. Both measured cells run against us: the transformer sits at 95.6% GPU-busy at
its measured cell with no headroom, while our 64-stream cell is 9.5% GPU-busy on a 96 GB card and runs the
unoptimised Triton per-step decode path, so it is a lower bound. Granting the transformer an
idealized KV ÷8 stack divides our decode lever by 8 and puts it ahead at 64k — that row is reported,
not hidden. The retired lines are **×53 decode**, **×15.9 decode**, **512 streams at 0.15 MB** and the
**~50–500×** band. Training (2026-08-07/08) is unchanged: the same 8 GPUs do **×1.39** the steps/s at
matched load (×5.15 fwd+bwd vs ×3.70 forward-only), **×1.70** at 256 sequences in flight.
Quality-parity at 70B (×4.1 params or ×1.7 tokens) is a **projection** from the measured ladder fits,
labeled as such. Systems numbers are component-scope; every speedup is vs the family's own 1-GPU
baseline.

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

**Key results.** FY25: ~\$370B AI capex vs ~\$79B AI revenue → ~−\$295B/yr burn. **Today** (quality-matched
at 1T, ~20×, the default): spend cut ~\$159B → burn ~−\$136B (~\$2.6T capitalized at the 6% rate; global
est ~\$3.3T FY25, ~\$7.6T FY26). On FY2026 guidance the cut grows to ~\$366B. **Ceiling** (~41×): ~\$163B
FY25. The cut is `accelerator capex × (1 − 1/reduction)` and it saturates; the underlying capex, shares
and revenue never move.

**Sensitivity.** `capex_avoided ∝ (1 − 1/R)` is 0.95 at R ≈ 20, so the dollar headline is not very
sensitive to the compute lever — doubling it (Today → Ceiling) moves FY25 by ~3%. The discount rate IS
a first-order lever on the capitalized figures: they scale as 1/rate.

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
st.caption("Interactive mirror of the workbook — the \\$ value of the measured architecture advantage "
           "across the 6 largest AI-capex spenders + a global estimate. Two scenarios: **Today** "
           "(memory ÷100 + compute ×9.24 → ~20× cost-weighted) and **Ceiling** (prefill kernels "
           "mature, ×22 → ~41×). Set the workload and scenario in the sidebar; derivations live in "
           "the **Methodology** and **Serving·Training** tabs. "
           "🟡 assumption · 🟢 disclosed data · 🔵 derived.")

g = sidebar_globals()
names = ["Totals"] + [c["name"] for c in COMPANIES] + ["Inputs", "Sensitivity", "CostLadder", "Serving·Training", "Evidence", "Methodology"]
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
    serving_training_tab(g)
with T[5 + nco]:
    evidence_tab(g)
with T[6 + nco]:
    methodology_tab(g)
