"""AI Capex Efficiency — interactive mirror of the AI_Capex_Efficiency workbook.

One Streamlit tab per worksheet: the audience layer (Summary, Value Bridge, Levers, What Matters),
one tab per company, then the technical appendix (Totals, Inputs, Sensitivity, CostLadder,
Serving·Training, Evidence, Methodology). Same colours and markers as the Excel:
  teal H1-H6 = Helarctos lever · yellow = assumption · green = disclosed data · blue = formula
  ◆ purple = differs by scenario · ★ orange border = high-impact input
The sidebar separates the Helarctos levers from market & modelling data. All math comes from
ai_capex_model.py, so the app and the spreadsheet can't drift.

Run locally:  uv run --with streamlit --with pandas streamlit run app.py
Deploy free:  push to GitHub -> share.streamlit.io  (needs requirements.txt)
"""

from pathlib import Path

import importlib

import streamlit as st
import pandas as pd

import ai_capex_model

# Streamlit Cloud pulls new code into a running process without restarting it,
# so a module imported before the pull stays cached and app.py can fail with an
# ImportError against the stale copy. Reloading on every run prevents that; the
# import is cheap.
importlib.reload(ai_capex_model)

from ai_capex_model import (GLOBALS, COMPANIES, MEASURED, SERVING,
                            SERVING_TRAINING_ASSUMPTIONS, PREFILL_BF16_D2048_SWEEP,
                            WORKLOAD,
                            workload_compute_advantage, reduction_factor,
                            energy_reduction, compute_company, compute_year,
                            global_estimate, serving_economics, serving_cost_curve,
                            training_throughput_ratio, QUALITY_FIT_20260814,
                            STEP_GAP_20260814,
                            DECK_DEPLOYMENT_SCALE, param_matching_fraction,
                            param_matching_gain,
                            CEILING_PREFILL_SPEEDUP,
                            KERNEL_CAMPAIGN_20260824,
                            campaign_landed_flop_lever, CAMPAIGN_LANDED_20260831,
                            KERNEL_SPEEDUP_REALIZED_20260824,
                            KERNEL_SPEEDUP_REMAINING_TARGET,
                            serving_context_sensitivity,
                            training_cost_saving, training_step_ratio,
                            training_advantage_mix, training_context_crossover,
                            training_helps_headline_threshold,
                            headline_with_training, TRAINING_CURRICULA,
                            KV_MB_PER_TOKEN_PER_STREAM,
                            value_bridge, value_bridge_levers, savings_ladder,
                            INFRA_SHARE_BASIS, HELARCTOS_LEVERS, lever_label,
                            decode_levers, prefill_advantage, prompt_time_ratio,
                            inference_throughput, fleet_memory_lever)

st.set_page_config(page_title="AI Capex Efficiency", layout="wide")

# ---- spreadsheet palette (matches the .xlsx fills) -----------------------------
YEL, GRN, BLU, SUB = "#FFF2CC", "#E2EFDA", "#DDEBF7", "#BDD7EE"
PUR = "#E4DFEC"  # ◆ differs by scenario
TEAL = "#A8E6D2"  # a Helarctos lever (H1-H6)
KPI = "#FCE4D6"   # headline result
CMAP = {"y": YEL, "g": GRN, "b": BLU, "s": SUB, "p": PUR, "h": TEAL, "k": KPI, "": ""}


def _css(code):
    """Cell style code: one fill letter from CMAP, plus flags '*' (★ orange border,
    high-impact input) and '!' (bold). 'h' also sets the dark-teal lever text."""
    fill = CMAP.get(code.rstrip("*!"), "")
    out = [f"background-color:{fill}"] if fill else []
    if code.startswith("h"):
        out.append("color:#0B5E48;font-weight:600")
    if "*" in code:
        out.append("border:2px solid #C55A11")
    if "!" in code:
        out.append("font-weight:700")
    return ";".join(out)

st.markdown("""<style>
.xlhead{background:#1F4E78;color:#fff;padding:5px 10px;font-weight:600;
        border-radius:3px;margin:18px 0 6px;font-size:0.92rem;}
.sw{display:inline-block;padding:1px 7px;border-radius:3px;margin-right:4px;font-size:0.85rem;}
.block-container{padding-top:2.2rem;}
div[data-testid="stNumberInput"] input{padding:2px 6px;}
</style>""", unsafe_allow_html=True)


def section(title):
    st.markdown(f'<div class="xlhead">{title}</div>', unsafe_allow_html=True)


def show_table(columns, rows, widths=None, height=None, wrap=False):
    """rows: list of rows; each row is a list of (text, color) where color in CMAP.
    Renders a colored, Excel-like grid. wrap=True renders a static table so long
    text wraps instead of being cut off."""
    texts = [[c[0] for c in row] for row in rows]
    if wrap:  # st.table renders markdown: keep a leading "+ " / "- " from becoming a bullet
        texts = [["\\" + t if isinstance(t, str) and t[:2] in ("+ ", "- ", "* ") else t for t in row]
                 for row in texts]
        texts = [[t.replace("$", "\\$") if isinstance(t, str) else t for t in row] for row in texts]
    styles = [[_css(c[1]) for c in row] for row in rows]
    df = pd.DataFrame(texts, columns=columns)
    smat = pd.DataFrame(styles, columns=columns)
    sty = df.style.apply(lambda _: smat, axis=None)
    if wrap:
        st.table(sty.hide(axis="index"))
        return
    cfg = {col: st.column_config.Column(width=w) for col, w in (widths or {}).items()}
    st.dataframe(sty, hide_index=True, width="stretch",
                 column_config=cfg or None, height=height or (len(rows) + 1) * 35 + 3)


# ---- formatters ----------------------------------------------------------------
def n1(v): return f"{v:,.1f}"
def n0(v): return f"{v:,.0f}"
def pct(v): return f"{v:.0%}"
def usd0(v): return f"${v:,.0f}"
def x1(v): return f"{v:.1f}×"
def ctx_k(t): return "262k" if int(t) == 262144 else f"{int(t) // 1024}k"
def bn(v): return f"-${-v:,.0f}B" if v < 0 else f"${v:,.0f}B"  # $B, minus sign first
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


# ---- sidebar: Helarctos levers vs market & modelling data ----------------------
LEGEND_MD = ('<span class="sw" style="background:#A8E6D2;color:#0B5E48;font-weight:600">H1–H6 Helarctos lever</span>'
             '<span class="sw" style="background:#FFF2CC">assumption</span>'
             '<span class="sw" style="background:#E2EFDA">disclosed data</span>'
             '<span class="sw" style="background:#DDEBF7">formula</span>'
             '<span class="sw" style="background:#E4DFEC">◆ differs by scenario</span>'
             '<span class="sw" style="border:2px solid #C55A11">★ high impact</span>')


def sidebar_globals():
    s = st.sidebar
    s.title("Assumptions")
    s.caption("Edit any value; every tab recomputes. Only the **H1–H6** levers are about the Helarctos "
              "architecture — everything else is market, company or modelling data.")
    g = dict(GLOBALS)

    def gnum(label, key, lo, hi, step, fmt, help=None, box=None):
        st.session_state.setdefault(key, float(GLOBALS[key]))
        return (box or s).number_input(label, min_value=float(lo), max_value=float(hi),
                                       step=float(step), key=key, format=fmt, help=help)

    # Containers fix the visual order (scenario, levers, workload, market data);
    # the widgets are created in dependency order (workload -> scenario -> H4-H6).
    scen_box = s.container()
    s.subheader("Helarctos levers (H1–H6)")
    lev_box = s.container()
    wl_box = s.expander("Workload behind H5–H6 (context, input : output)")
    s.subheader("Market & modelling data")
    mkt_box = s.container()

    # App defaults (2026-09-01 user ruling): E[context] 262k and training share 35%
    # for the workload blend; the model's WORKLOAD dict keeps its own defaults.
    st.session_state.setdefault("in_out_ratio", float(WORKLOAD["in_out_ratio"]))
    st.session_state.setdefault("context_tokens", 262144)
    st.session_state.setdefault("train_share", 0.35)
    wl = {
        "in_out_ratio": wl_box.number_input(
            "Input : output token ratio", min_value=0.1, max_value=1000.0, step=1.0,
            key="in_out_ratio", format="%.1f",
            help="Input tokens per generated token. ~10:1 = code/agent traces; 50-100:1 = RAG."),
        "context_tokens": int(wl_box.number_input(
            "Average conversation length (tokens)", min_value=1024, max_value=1048576, step=1024,
            key="context_tokens", format="%d",
            help="Average conversation length over the workload. H4–H6 are quoted at 262k; changing it "
                 "resets H5 and H6 to their values at the new context.")),
        "train_share": wl_box.number_input(
            "Training share (workload blend, technical tabs)", min_value=0.0, max_value=1.0, step=0.05,
            key="train_share", format="%.2f",
            help="Only feeds the measured workload blend on the Serving·Training tab."),
    }
    adv = workload_compute_advantage(wl)
    wl_box.caption(
        f"→ measured blend: prefill **×{adv['prefill_ratio']:.2f}** · decode **×{adv['decode_ratio']:.1f}** "
        f"· blended **×{adv['blended']:.2f}** (Serving·Training tab)")

    # No model-size slider (2026-09-29 user ruling): the equal-quality ratio is
    # fixed at trillion-parameter scale and applies to TRAINING only; inference
    # cost does not scale with model size. The inference compute lever is built
    # from H5 x H6 (editable, defaults follow the workload's context) plus prompt
    # processing at the scenario's prefill speed -- the only inference input that
    # differs by scenario (2026-09-29 (4) user ruling).
    _gain = param_matching_gain(DECK_DEPLOYMENT_SCALE)
    ctx = wl["context_tokens"]
    TODAY_LABEL = "Current kernels"
    LANDED_LABEL = "Optimized kernels"
    scenarios = {TODAY_LABEL: "current", LANDED_LABEL: "mature"}
    if st.session_state.get("scenario") not in scenarios:
        st.session_state["scenario"] = TODAY_LABEL
    scen_box.radio("◆ Scenario", list(scenarios), key="scenario",
                   help="Current kernels = the software built and measured today. Optimized kernels = the "
                        "funded kernel programme lands (a TARGET).")
    pf = {k: prefill_advantage(v, ctx) for k, v in scenarios.items()}
    scen_box.caption(f"◆ Only prompt-processing speed (×{pf[TODAY_LABEL]:.1f} current vs "
                     f"×{pf[LANDED_LABEL]:.1f} optimized) and H3 (×1 in both) differ by scenario.")

    # H5 / H6 defaults track the context; a changed context resets them.
    d5, d6 = decode_levers(ctx)
    if st.session_state.get("_h56_ctx") != ctx:
        st.session_state["conv_per_gpu"], st.session_state["decode_speedup"] = float(d5), float(d6)
        st.session_state["_h56_ctx"] = ctx
    lev_box.caption(f"**H1** smaller model ×{_gain:.2f} and **H2** fewer tokens ×{_gain:.2f} (training only; "
                    f"Inputs tab) · **H3** training speed ×1 (Levers tab).")
    g["mem_factor"] = gnum("H4 · Memory per conversation, ÷", "mem_factor", 1, 20000, 10, "%.0f",
                           help="MEASURED ×2,032 at 262k on our test model. Frontier-size models with "
                                f"grouped-query attention: ~÷{fleet_memory_lever(262144):.0f} (estimate).",
                           box=lev_box)
    g["conv_per_gpu"] = lev_box.number_input(
        "H5 · More conversations per GPU, ×", min_value=1.0, max_value=1024.0, step=1.0, key="conv_per_gpu",
        format="%.0f", help="MEASURED: 64 resident 262k conversations per GPU vs the transformer's 1.")
    g["decode_speedup"] = lev_box.number_input(
        "H6 · Faster decode per token, ×", min_value=0.1, max_value=100.0, step=0.1, key="decode_speedup",
        format="%.2f", help="ESTIMATE: 15.6 ms vs 2.5 ms per token at 262k (×2.6 measured on older kernels).")
    g["flop_factor"] = max(1.0, inference_throughput(g["conv_per_gpu"], g["decode_speedup"],
                                                     pf[st.session_state["scenario"]],
                                                     prompt_time_ratio(ctx, wl["in_out_ratio"])))
    lev_box.caption(f"→ tokens per GPU (H5 × H6, with prompts) **×{g['flop_factor']:,.0f}** · inference GPUs "
                    f"fall by **×{min(g['mem_factor'], g['flop_factor']):,.0f}**")

    m = mkt_box
    g["gpu_cost"] = gnum("Fully-loaded $ per GPU", "gpu_cost", 5000, 150000, 1000, "%.0f", box=m)
    g["wall_power_kw"] = gnum("Wall power per GPU (kW)", "wall_power_kw", 0.3, 5.0, 0.1, "%.1f", box=m)
    g["elec_rate"] = gnum("Electricity ($/kWh)", "elec_rate", 0.02, 0.40, 0.01, "%.2f", box=m)
    g["discount_rate"] = gnum("Discount rate ★", "discount_rate", 0.02, 0.30, 0.01, "%.2f",
                              help="Changes only the capitalized value (≈1/rate).", box=m)
    m.caption("Capex and data-centre / server / accelerator shares: ✏️ panel on each company tab. "
              "Training shares: Levers tab.")
    with m.expander("Technical tabs only"):
        g["mem_share"] = gnum("Memory share of GPU cost", "mem_share", 0.0, 1.0, 0.05, "%.2f", box=st)
        auto_energy = st.checkbox("Auto-derive energy reduction (= cost reduction)", value=True,
                                  help="Energy splits memory/compute like cost does. Uncheck to set manually.")
        if auto_energy:
            g["opex_reduction_override"] = None
        else:
            st.session_state.setdefault("opex_override", round(reduction_factor(g), 1))
            g["opex_reduction_override"] = st.number_input("Energy reduction override (×)", min_value=1.0,
                                                           max_value=200.0, step=1.0, key="opex_override",
                                                           format="%.1f")
        st.caption(f"→ energy reduction = **{x1(energy_reduction(g))}** "
                   f"({'derived' if auto_energy else 'manual override'})")
        g["cooling_overhead"] = gnum("Cooling/ops overhead", "cooling_overhead", 0.0, 1.0, 0.05, "%.2f", box=st)
        g["fleet_life_yr"] = gnum("Fleet life (yr)", "fleet_life_yr", 1, 10, 1, "%.0f", box=st)
        g["dc_scale"] = gnum("Datacenter scaling factor", "dc_scale", 0.0, 1.0, 0.05, "%.2f", box=st,
                             help="0 = only accelerator silicon shrinks (conservative). 1 = whole DC scales.")
        g["named_share_of_global"] = gnum("Named share of global AI capex", "named_share_of_global",
                                          0.3, 1.0, 0.05, "%.2f", box=st)
        g["spacex_mktcap"] = gnum("SpaceX market cap ($B)", "spacex_mktcap", 200, 4000, 10, "%.0f", box=st)
        st.metric("Cost-weighted reduction", x1(reduction_factor(g)))
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
    r25, r26 = c["ai_rev"][0], c["ai_rev"][1]
    rev_tail = tuple(c["ai_rev"][2:])  # fy27 est (charts-only) — preserved untouched
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
    c["ai_rev"], c["mcap"] = (r25, r26) + rev_tail, mcap


def company_tab(c, g):
    name = c["name"]
    with st.expander(f"✏️  Edit {name}'s inputs (capex · shares · AI revenue)", expanded=False):
        edit_company_inputs(c)

    d25, d26 = compute_company(c, g, "fy25"), compute_company(c, g, "fy26")
    fb25, fb26 = fleet_breakdown(d25["accel"], g), fleet_breakdown(d26["accel"], g)
    t25, i25, s25, a25 = c["fy25"]; t26, i26, s26, a26 = c["fy26"]
    CB = ["Metric", "FY2025", "FY2026", "Basis / source"]
    W = {"Metric": "large", "Basis / source": "large"}

    section(f"{name} — company data (green = disclosed, yellow = estimate)")
    show_table(CB, [
        [("Total capex ($B)", ""), (n1(t25), "g"), (n1(t26), "y"), ("FY25 disclosed; FY26 guide/est", "")],
        [("Infra / data-center share ★", ""), (pct(i25), "y*"), (pct(i26), "y*"), (INFRA_SHARE_BASIS.get(name, "strips non-datacenter capex"), "")],
        [("Server / short-lived share ★", ""), (pct(s25), "y*"), (pct(s26), "y*"), ("CFO-disclosed", "")],
        [("Accelerator share within servers ★", ""), (pct(a25), "y*"), (pct(a26), "y*"), ("BOM teardown ~67–80%", "")],
        [("Market cap ($B)", ""), (n0(c["mcap"]), "g"), ("", ""), ("approx market data", "")],
    ], widths=W, wrap=True)

    section("Derivation — how much of the capex buys AI chips")
    show_table(CB, [
        [("AI-infra capex ($B)", ""), (n1(d25["ai_capex"]), "b"), (n1(d26["ai_capex"]), "b"), ("= total × infra", "")],
        [("Server bucket ($B)", ""), (n1(d25["ai_capex"] * s25), "b"), (n1(d26["ai_capex"] * s26), "b"), ("= infra × server", "")],
        [("ACCELERATOR capex ($B)", ""), (n1(d25["accel"]), "b"), (n1(d26["accel"]), "b"), ("= server × accel", "")],
        [("Accel % of total capex", ""), (pct(d25["accel_pct"]), "b"), (pct(d26["accel_pct"]), "b"), ("varies by company", "")],
    ], widths=W, wrap=True)

    section("Fleet & power (from AI-chip capex)")
    show_table(CB, [
        [("Fleet size (GPU-equiv)", ""), (n0(fb25["fleet"]), "b"), (n0(fb26["fleet"]), "b"), ("= accel capex / $ per GPU", "")],
        [("Total wall power (MW)", ""), (n0(fb25["mw"]), "b"), (n0(fb26["mw"]), "b"), ("= GPUs × kW", "")],
        [("Annual opex ($M)", ""), (n0(fb25["ann_m"]), "b"), (n0(fb26["ann_m"]), "b"), ("= MWh × rate × (1+overhead) × 365", "")],
        [("Lifetime opex ($B)", ""), (n1(fb25["life_b"]), "b"), (n1(fb26["life_b"]), "b"), ("× fleet life", "")],
    ], widths=W, wrap=True)

    section("Helarctos effect — technical engine (memory and compute priced separately)")
    show_table(CB, [
        [("Efficient AI capex ($B)", ""), (n1(d25["ai_capex"] - d25["capex_avoided"]), "b"), (n1(d26["ai_capex"] - d26["capex_avoided"]), "b"), ("= AI capex − avoided", "")],
        [("H4·H5 → Capex avoided/yr ($B)", "h"), (n1(d25["capex_avoided"]), "b"), (n1(d26["capex_avoided"]), "b"), ("Helarctos enters here: chips × (1 − 1/cost-weighted cut from H4 memory and H5 compute)", "")],
        [("H4·H5 → Annual opex savings ($M)", "h"), (n0(d25["opex_saved"] * 1000), "b"), (n0(d26["opex_saved"] * 1000), "b"), ("Helarctos enters here: power shrinks with the fleet", "")],
        [("Sustained annual benefit ($B/yr)", ""), (n1(d25["spend_cut"]), "b"), (n1(d26["spend_cut"]), "b"), ("= avoided + opex savings", "")],
        [("Capitalized value ($B)", ""), (n0(d25["capitalized"]), "b"), (n0(d26["capitalized"]), "b"), ("= benefit / discount rate", "")],
        [("% of market cap", ""), (f"{d25['capitalized'] / c['mcap']:.1%}" if c["mcap"] else "—", "b"), (f"{d26['capitalized'] / c['mcap']:.1%}" if c["mcap"] else "—", "b"), ("", "")],
    ], widths=W, wrap=True)

    section("AI economics — cash basis: AI revenue − AI capex − AI power")
    show_table(CB, [
        [("AI revenue ($B)", ""), (n1(c["ai_rev"][0]), "y"), (n1(c["ai_rev"][1]), "y"), ("ESTIMATE (see Methodology)", "")],
        [("AI capex ($B)", ""), (n1(d25["ai_capex"]), "b"), (n1(d26["ai_capex"]), "b"), ("full AI-infra (accel + buildings + power + net)", "")],
        [("AI opex ($B)", ""), (n1(d25["ai_opex"]), "b"), (n1(d26["ai_opex"]), "b"), ("annual power / operating", "")],
        [("Net AI NOW ($B)", ""), (n1(d25["net_now"]), "b"), (n1(d26["net_now"]), "b"), ("revenue − capex − opex (cash burn)", "")],
        [("Spend cut with Helarctos ($B)", ""), (n1(d25["spend_cut"]), "b"), (n1(d26["spend_cut"]), "b"), ("accel capex avoided + opex saved", "")],
        [("Net AI with Helarctos ($B)", ""), (n1(d25["net_arch"]), "b"), (n1(d26["net_arch"]), "b"), ("= net now + spend cut", "")],
        [("% AI spend reduction", ""), (pct(d25["pct_cut"]), "b"), (pct(d26["pct_cut"]), "b"), ("spend cut / total AI spend", "")],
    ], widths=W, wrap=True)

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
    _, tot27 = compute_year(g, comps, "fy27")
    glob25, glob26 = global_estimate(tot25, g), global_estimate(tot26, g)

    c1, c2, c4 = st.columns(3)
    c1.metric("Cost-weighted reduction", x1(reduction_factor(g)))
    # "Now" = FY26 (2026-09-01 user correction: the FY25 -$295B read as far too
    # small — the current-year burn is the FY26 estimate).
    c2.metric(f"Net AI now ({len(comps)} cos, FY26)", f"{usd0(tot26['net_now'])}B/yr")
    c4.metric("% of AI spend cut (FY26)", pct(tot26["pct_cut"]))
    n1c, n2c, n3c = st.columns(3)
    n1c.metric("Net AI w/ our arch (FY25)", f"{usd0(tot25['net_arch'])}B/yr", delta=f"{usd0(tot25['spend_cut'])}B cut")
    n2c.metric("Net AI w/ our arch (FY26)", f"{usd0(tot26['net_arch'])}B/yr", delta=f"{usd0(tot26['spend_cut'])}B cut")
    n3c.metric("Net AI w/ our arch (FY27 pred.)", f"{usd0(tot27['net_arch'])}B/yr", delta=f"{usd0(tot27['spend_cut'])}B cut")
    st.caption("FY27 = street-estimate capex (Morgan Stanley +57% path; Oracle's ~$70B is the only real FY27 guide) "
               "with ai_rev at the ×1.5 placeholder — prediction-grade, see per-company notes.")
    verdict = ("AI flips **profitable** at these settings." if tot26["net_arch"] > 0
               else f"FY26 AI burn shrinks from **{md_usd(-tot26['net_now'])}B** to **{md_usd(-tot26['net_arch'])}B**/yr "
                    f"(spend cut **{md_usd(tot26['spend_cut'])}B**, ~{md_usd(tot26['capitalized'])}B capitalized).")
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
    section("Global inputs (edit in the sidebar ◀) — teal = Helarctos lever")
    items = [
        ("H4 · Memory reduction factor", f"{g['mem_factor']:,.0f}×", "h", "O(1) state vs O(T) KV cache — MEASURED 2026-08-14 (full model, our test model): 64 concurrent 262k streams on one GPU in 1.62 GB against the transformer's 1 stream at 51.5 GB, a 2nd OOMs; 25.4 MB of state per stream vs 197 KB of KV per token of context = ×2,032 at 262k. Frontier-size models with grouped-query attention: ~÷208 (estimate)"),
        ("H5·H6 · FLOPs (inference compute) reduction factor", f"{g['flop_factor']:.2f}×", "b", f"= (1 + rq) / (rq/PF + 1/(H5 × H6)): H5 = {g['conv_per_gpu']:.0f} more conversations per GPU (MEASURED), H6 = ×{g['decode_speedup']:.2f} faster decode per token (ESTIMATE: 2.5 ms/token), PF = the scenario's prompt-processing speed vs the transformer (×{prefill_advantage('current'):.1f} current, ×{prefill_advantage('mature'):.1f} optimized), rq = the transformer's prompt time / decode time. Edit H5 and H6 in the sidebar"),
        ("Memory share of GPU cost", pct(g["mem_share"]), "y", "HBM + most CoWoS packaging → ~60/40 memory/compute (BOM)"),
        ("Opex / energy reduction", x1(energy_reduction(g)), "b" if g.get("opex_reduction_override") is None else "y", "DERIVED = cost-weighted reduction (energy splits memory/compute like cost); override in sidebar"),
        ("Discount rate ★", pct(g["discount_rate"]), "y", "perpetuity: value = annual benefit / rate"),
        ("Fully-loaded $/GPU", usd0(g["gpu_cost"]), "y", "GPU + share of server, NVLink, networking"),
        ("Wall power / GPU (kW)", f"{g['wall_power_kw']:.1f}", "y", "GB200 NVL72: ~1.7–1.8 kW IT per GPU × PUE ~1.3"),
        ("Electricity rate ($/kWh)", f"{g['elec_rate']:.2f}", "y", "datacenter wholesale"),
        ("Cooling / ops overhead", pct(g["cooling_overhead"]), "y", "non-power running cost as fraction of electricity"),
        ("Fleet useful life (yr)", f"{g['fleet_life_yr']:.0f}", "y", "AI-GPU depreciation life"),
        ("Datacenter scaling factor", pct(g["dc_scale"]), "y", "0 = accel-only; 1 = whole DC scales; ~0.7 ≈ breakeven"),
        ("Named share of global AI capex", pct(g["named_share_of_global"]), "y", "named firms' share of worldwide AI capex (for the GLOBAL row)"),
        ("SpaceX market cap ($B)", n0(g["spacex_mktcap"]), "g", "market data ~$1.84T Aug 2026 (IPO 2026-06-12 at ~$1.77T)"),
    ]
    show_table(["Input", "Value", "Kind", "Note"],
               [[(a, "h" if a.startswith("H") else ""), (b, c + ("*" if "★" in a else "")),
                 (f"Helarctos lever {a.split(' · ')[0]}" if a.startswith("H") else
                  ("Derived" if a.startswith("Opex") else "Market / modelling data"), "h" if a.startswith("H") else ""),
                 (d, "")] for a, b, c, d in items], wrap=True)

    section("Reduction engine (Amdahl cost-weighting) — derived")
    cs = 1 - g["mem_share"]; mf = g["mem_share"] / g["mem_factor"]; cf = cs / g["flop_factor"]; res = mf + cf
    show_table(["Metric", "Value"], [
        [("Compute share of GPU cost", ""), (pct(cs), "b")],
        [("Memory cost fraction after reduction", ""), (f"{mf:.2%}", "b")],
        [("Compute cost fraction after reduction", ""), (f"{cf:.2%}", "b")],
        [("Residual cost fraction", ""), (pct(res), "b")],
        [("COST-WEIGHTED reduction factor", ""), (x1(1 / res), "b")],
    ])
    st.caption("Floored by the least-reduced component — at the default levers, compute (tokens per GPU, ×368).")


# ---- sensitivity tab -----------------------------------------------------------
def sensitivity_tab(comps, g):
    section("Sensitivity (SpaceX) — Current vs Optimized kernels vs live cost-weighted")
    sx = next(c for c in comps if c["name"] == "SpaceX")
    d = compute_company(sx, g, "fy25")
    accel, opx, disc, mcap = d["accel"], d["opex_saved"] * 1000, g["discount_rate"], g["spacex_mktcap"]
    # Tiers mirror the sidebar picker (2026-09-01: pre-campaign floor and the
    # prefill-only ceiling are deleted — current numbers and upper bound only).
    _gn = param_matching_gain(DECK_DEPLOYMENT_SCALE)  # inference lever is size-independent
    r_today = reduction_factor(dict(g, flop_factor=campaign_landed_flop_lever(
        prefill_speedup=KERNEL_SPEEDUP_REALIZED_20260824) / _gn))
    r_ceil = reduction_factor(dict(g, flop_factor=campaign_landed_flop_lever() / _gn))
    tiers = [(f"Current kernels {r_today:.0f}×", r_today), (f"Optimized kernels {r_ceil:.0f}×", r_ceil),
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

    section("Conversation-length sensitivity — prompt share of serving cost vs average conversation length")
    ctx_rows = []
    for cs in serving_context_sensitivity():
        _t = int(cs["context_tokens"])
        lbl = ("262k" if _t == 262144 else f"{_t // 1024}k") + (" (default)" if cs["pinned"] else "")
        ctx_rows.append([
            (lbl, "b" if cs["pinned"] else ""),
            (f"{cs['tf_prefill_share']:.2%}", ""),
            (f"{cs['own_prefill_share_today']:.1%}", ""),
            (f"{cs['today_lever'] / param_matching_gain(DECK_DEPLOYMENT_SCALE):,.0f}×", "p"),
            (f"{cs['ceiling_lever'] / param_matching_gain(DECK_DEPLOYMENT_SCALE):,.0f}×", "p"),
            (f"{cs['gap_pct']:.1%}", "y"),
        ])
    show_table(["Conversation length", "TF prefill share", "Our prefill share (current)",
                "◆ Inference lever, current kernels", "◆ Inference lever, optimized kernels", "Gap"], ctx_rows)
    st.caption("Cost = box wall-clock, not FLOPs. The transformer's decode leg is KV-bandwidth-bound "
               "(measured 1/context law) while its prefill runs near peak, so prefill is <1% of its "
               "serving cost at every context — which is why the banked ×3.936 prefill (Current kernels) "
               "and the full ×7.03 (Optimized kernels) land within a few percent of each other. The "
               "default is the 262k row (2026-09-29).")


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

    section("Estimated TRAINING cost at equal quality (2026-08-29 re-anchor)")
    st.caption("Training compute is ~6·N·D FLOPs. At equal quality we need N/s parameters, "
               "where **s(N)** is the fit-derived equal-quality parameter ratio (same sealed "
               "2026-08-14 refit as the compute lever). Two token regimes fall out, differing by "
               "exactly one power of s: **compute-optimal** (Chinchilla, D ∝ N — the smaller model "
               "also trains on fewer tokens, so cost ratio = s²/r) and **fixed token budget** "
               "(data-constrained, D equal on both arms, so cost ratio = s/r). **r(T)** is the "
               "step-time ratio at matched parameters: **MEASURED** at T=2,048 (×1.72, the "
               "2026-08-29 clean-wall position of record) and T=8,192 (×1.79 at B4, "
               "artifact-flagged), **MODELED** beyond from the per-token cost form below "
               "(the B4/T32,768 ~parity cell is retired as a batch-occupancy artifact). "
               "Above 1× we are cheaper to train to the same quality.")
    _tc_ctx = (2048, 8192, 32768, 262144)
    _tc_scales = (1_000_000_000, 10_000_000_000, 100_000_000_000,
                  1_000_000_000_000, 10_000_000_000_000)
    _tc_rows = []
    for _n in _tc_scales:
        _s = param_matching_gain(_n)
        _row = [(f"{_n / 1e9:,.0f}B" if _n < 1e12 else f"{_n / 1e12:,.0f}T", ""),
                (f"×{_s:.2f}", "y")]
        for _chin in (True, False):
            for _c in _tc_ctx:
                _v = training_cost_saving(_n, _c, chinchilla=_chin)
                _row.append((f"×{_v:.2f}", "b" if _v >= 1.0 else "y"))
        _tc_rows.append(_row)
    show_table(["Transformer params", "s(N) fit",
                "Chinchilla 2k", "Chinchilla 8k", "Chinchilla 32k", "Chinchilla 256k*",
                "Fixed-D 2k", "Fixed-D 8k", "Fixed-D 32k", "Fixed-D 256k*"], _tc_rows)
    st.caption(
        "\\* 256k is **MODELED**. r(T) now comes from the per-token cost form rather than a "
        "constant ratio decay: the transformer pays `base + attn·T` per token (FlashAttention is "
        "linear per token, quadratic over the sequence) while our recurrent state is O(1), so our "
        "per-token cost is FLAT — now MEASURED, not inferred: BDM backward per-token is flat in T "
        "(3.27–3.31 µs/tok, spread 1.4%, across 8× context; tsweep d4096 n=25) while the transformer's "
        "rises +66%, and decode is flat +0.83% over 128× context. The B4/T32,768 ~parity cell stays "
        "retired (90.5% an h-grid occupancy effect: B·n_blocks = 124 < 132 SMs at B4). `base` and "
        "`attn` are fitted to the 2k and 8k cells; the flat model gives "
        f"r(32,768) = {training_step_ratio(32768):.2f} and puts the crossover at "
        f"T ≈ {training_context_crossover():,.0f} (PROJECTED for this frame; independently measured "
        f"21–24k @ B4 / ~16k @ B16 on H100 d4096). The old constant-decay curve is kept as the CONSERVATIVE bound "
        f"and is far too pessimistic past 32k (it says ×"
        f"{training_step_ratio(262144, mode='constant_decay'):.2f} at 256k where the cost model says ×"
        f"{training_step_ratio(262144):.2f}) because holding a growth ratio constant assumes the "
        "transformer's attention term stops growing. The fits cross at 392M, so below that we "
        "need MORE parameters, not fewer, and every cell past ~1B extrapolates beyond the "
        "measured 47M–663M ladder rungs. Peak training memory (×1.58 measured, ×0.88 targeted) is "
        "deliberately NOT in this model: it caps per-GPU batch density, not FLOPs. Chart: "
        "`paper/figures/vc_memo_training_cost_scaling.png`.")
    _tc_png = Path(__file__).resolve().parent / "paper/figures/vc_memo_training_cost_scaling.png"
    if _tc_png.exists():
        st.image(str(_tc_png), width="stretch")

    section("The training CURRICULUM — a context mix, not a single context")
    st.caption(
        "Quoting r at one context silently assumes **all** training happens there, and until now "
        "this model assumed 2,048 tokens for all of it — the pessimistic corner. Real training is a "
        "curriculum: a short-context bulk phase, then long-context extension phases, then "
        "long-rollout RL. The correct training term is the ratio of **costs integrated over the "
        "mix**, not the average of the per-context ratios. That distinction does the work here: our "
        "per-token cost is flat in context and the transformer's grows, so **long-context tokens "
        "dominate the transformer's bill while staying a token minority**. Token shares below are "
        "**ILLUSTRATIVE knobs** — this repo carries no citation for any lab's recipe, and none is "
        "invented here. The r(T) curve underneath them is measured/modeled as described above.")
    _cur_rows = []
    for _key, _spec in TRAINING_CURRICULA.items():
        _a = training_advantage_mix(_key)
        _mix = " · ".join(
            f"{_r['ctx'] // 1024}k: {_r['token_share'] * 100:.0f}% tok → "
            f"{_r['tf_cost_share'] * 100:.0f}% of TF cost" for _r in _a["rungs"])
        _cur_rows.append([
            (_spec["label"], ""),
            (f"×{_a['advantage']:.2f}", "b" if _a["advantage"] >= 1.0 else "y"),
            (f"×{_a['effective_step_ratio']:.2f}", "b" if _a["effective_step_ratio"] <= 1.0 else "y"),
            (_mix, ""),
            (_spec["status"], "g" if "MEASURED" in _spec["status"] else "y"),
        ])
    show_table(["Curriculum", "Training advantage (TF ÷ ours)", "Effective r", "Mix: token share → transformer cost share", "Status"],
               _cur_rows, widths={"Mix: token share → transformer cost share": "large", "Curriculum": "medium"})
    st.caption(
        f"Read the middle columns together. **Effective r crosses below 1 at every modern mix** — "
        f"training stops being a loss and becomes a contributor. But raising the *dollar headline* "
        f"is a higher bar: the blended lever is a cost-weighted harmonic mean, so training only "
        f"pushes the headline **up** once it beats the **serving** advantage of "
        f"×{training_helps_headline_threshold():.2f}, not merely ×1. Between the two, training is "
        f"profitable and still dilutive to that particular number. The right-hand column is the "
        f"crux: in the modern mix 10% of tokens sit at 256k and carry "
        f"{training_advantage_mix('modern_standard')['rungs'][-1]['tf_cost_share'] * 100:.0f}% of the "
        f"transformer's training cost.")

    section("What happens to the &#36;159B headline when training enters")
    _hl_rows = []
    for _key, _spec in TRAINING_CURRICULA.items():
        _row = [(_spec["label"], "")]
        for _ts in (0.0, 0.2, 0.4):
            _h = headline_with_training(_key, _ts)
            _row.append((f"${_h['fy25_spend_cut']:.0f}B", "b" if _h["delta_vs_serving_only"] >= 0 else "y"))
        _h2 = headline_with_training(_key, 0.2)
        _row.append(("raises it" if _h2["raises_headline"] else "dilutes it",
                     "b" if _h2["raises_headline"] else "y"))
        _hl_rows.append(_row)
    show_table(["Curriculum", "training 0% of spend", "training 20%", "training 40%", "Direction"], _hl_rows,
               widths={"Curriculum": "medium"})
    st.caption(
        "Every cell is FY2025 spend cut across the six named firms, recomputed end-to-end "
        "(`headline_with_training`). The old assumption — all training at 2k — was the only one that "
        "dragged the headline materially (to \\$147B at a 40% training share). At a modern mix the "
        "headline is essentially flat, and at long-context-heavy mixes it rises. Note how narrow the "
        "whole range is: Amdahl saturation means the dollar headline is simply not very sensitive to "
        "this lever, which is the honest shape of the result in both directions.")
    st.caption(
        "**Excluded, and it runs our way.** At 262k+ the transformer also pays sequence-parallelism "
        f"and activation-memory costs an O(1) recurrent state avoids — its KV grows at "
        f"{KV_MB_PER_TOKEN_PER_STREAM * 1024:.0f} KB per token of context per stream, which is what "
        f"forces sharding, ring/Ulysses attention and the communication overhead that comes with "
        f"them. Our own measurement shows the asymmetry in the small: from 2k to 8k the peak-memory "
        f"ratio went ×{KERNEL_CAMPAIGN_20260824['mem_ratio_2k']:.3f} → "
        f"×{KERNEL_CAMPAIGN_20260824['mem_ratio_8k']:.3f}, i.e. flat-to-falling in context while the "
        "absolute footprint the transformer must shard keeps growing. None of this is in the cost "
        "model above, which counts step time only — so the omission is **conservative**. Quantifying "
        "it needs a multi-GPU long-context training receipt we do not have.")

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
               "averaging penalty. **Training is excluded** — but by much less than it used to be. "
               "Re-based 2026-08-24/25/29 (one GH200, single layer, fwd+bwd, bf16, checkpointing off both "
               f"arms, modern transformer block): the transformer is ×{KERNEL_CAMPAIGN_20260824['step_ratio_2k']:.2f} "
               f"faster at T=2,048 and ×{KERNEL_CAMPAIGN_20260824['step_ratio_8k']:.2f} at T=8,192, "
               "the B4/T32,768 ~parity cell is retired as a batch-occupancy artifact and the flat "
               "per-token model puts the crossover at T≈13k (PROJECTED; measured 21–24k @ B4 on d4096; "
               "T≈88k pre-campaign). It read ×6.12 on the morning of 2026-08-24 (×5.3–9.6 on the older d1152 sweep). "
               "Setting a training share above ~0.30 now takes the blend below 1; that threshold was "
               "~0.06 before the kernel campaign.")

    section("The prefill lane (16-bit, parameter-matched)")
    st.caption("Measured where both families run their production 16-bit attention kernels — the "
               "lane the competition actually serves in — with parameter counts exactly matched: "
               "1 layer / batch 16, one H100, 2026-07-21. The fp32 full-model sweep is retired to "
               "the Evidence tab as a scope reference. **These are PRE-CAMPAIGN kernels** — the lane "
               "has not been re-run since the 2026-08-24..29 megakernel work banked ×3.94 on our own arm, "
               "so the lever below is conservative by construction; the remaining target is "
               "carried in the Optimized-kernels scenario, not here.")
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
        [("Memory (mem_factor)", ""), ("÷2,032 (measured, test model)", "s"),
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
        [("Memory reduction (×)", ""), (f"{g['mem_factor']:.0f}×", "y"), ("assumption", "y"), ("H4. Fixed-size state vs a growing KV cache. MEASURED 2026-08-14, full model (Serving·Training tab): 64 concurrent 262k conversations on one GPU in 1.62 GB vs the transformer's 1 at 51.5 GB (a 2nd OOMs); 25.4 MB of state per conversation vs 197 KB of KV per token — ×2,032 at 262k, used as H4. Frontier-size models with grouped-query attention would be ~×208 (estimate)", "")],
        [("FLOPs reduction (×)", ""), (f"{g['flop_factor']:.2f}×", "y"), ("assumption", "y"), ("◆ Set by the scenario (Current kernels by default) = the full-workload serving lever at a 262k-token average context × the FIT-DERIVED equal-quality parameter ratio. Serving: decode 2.5 ms/token × the measured 64-stream cell (aggregate-decode ESTIMATE, receipt pending), prefill at the BANKED " + f"×{KERNEL_SPEEDUP_REALIZED_20260824:.2f} (2k clean-wall position of record 2026-08-29). Parameter ratio: the sealed refit's fits (4 rungs per family) cross at 392M and give the transformer's quality on 23.7% of the params at 1T → ×4.22 — a projection of two fits, not a measurement. Ceiling = same at the full ×{CEILING_PREFILL_SPEEDUP:.2f} maturity factor (×{KERNEL_SPEEDUP_REALIZED_20260824:.2f} MEASURED × ×{KERNEL_SPEEDUP_REMAINING_TARGET:.2f} TARGET, the funded 8k-win gate)" + "; measured TRAINING cluster throughput ×1.39 matched load → ×1.70 deep", "")],
        [("Compute cost that runs AGAINST us", ""), (f"×{KERNEL_CAMPAIGN_20260824['step_ratio_2k']:.2f}", "y"), ("measured", "b"), (f"RE-BASED 2026-08-24 (one GH200, ONE layer, fwd+bwd, bf16, checkpointing off both arms, against a MODERN transformer block — 24Q/4KV, head_dim 256, RoPE 64, gated attention): a bAttention training step costs ×{KERNEL_CAMPAIGN_20260824['step_ratio_2k']:.2f} MORE at T=2,048 ({KERNEL_CAMPAIGN_20260824['battn_ms_2k']:.1f} vs {KERNEL_CAMPAIGN_20260824['tf_ms_2k']:.1f} ms/step) and ×{KERNEL_CAMPAIGN_20260824['step_ratio_8k']:.2f} at T=8,192 ({KERNEL_CAMPAIGN_20260824['battn_ms_8k']:.1f} vs {KERNEL_CAMPAIGN_20260824['tf_ms_8k']:.1f}); fwd ×{KERNEL_CAMPAIGN_20260824['fwd_ratio']:.2f}, bwd ×{KERNEL_CAMPAIGN_20260824['bwd_ratio']:.2f}. It read ×{KERNEL_CAMPAIGN_20260824['step_ratio_2k_precampaign']:.2f} ({KERNEL_CAMPAIGN_20260824['battn_ms_2k_precampaign']:.1f} ms) the same morning — ×{KERNEL_SPEEDUP_REALIZED_20260824:.2f} banked in a day — and ×{STEP_GAP_20260814['gap_against_battn']:.2f} on the older d1536/27L fp16 601-step receipt (dtype_trio_v4.json), which is now superseded as a headline. At equal quality the T=2,048 figure falls to ×{KERNEL_CAMPAIGN_20260824['step_ratio_2k'] * param_matching_fraction(DECK_DEPLOYMENT_SCALE):.2f}. Training is still excluded from the serving claim", "")],
        [("Training PEAK MEMORY that runs AGAINST us", ""), (f"×{KERNEL_CAMPAIGN_20260824['mem_ratio_2k']:.2f}", "y"), ("measured", "b"), (f"Same 2026-08-24 frame: ×{KERNEL_CAMPAIGN_20260824['mem_ratio_2k']:.3f} at T=2,048 ({KERNEL_CAMPAIGN_20260824['battn_peak_mib_2k']:,.1f} vs {KERNEL_CAMPAIGN_20260824['tf_peak_mib_2k']:,.1f} MiB) and ×{KERNEL_CAMPAIGN_20260824['mem_ratio_8k']:.3f} at T=8,192, down from ×{KERNEL_CAMPAIGN_20260824['mem_ratio_2k_precampaign']:.3f} the same morning. This is TRAINING peak memory — NOT the ÷100 memory lever above, which is SERVING state and is unaffected. It is not an input to the cost model; it caps per-GPU batch density", "")],
        [("Kernel program (TARGET, not a result)", ""), (f"≤{KERNEL_CAMPAIGN_20260824['target_8k_win_gate_ms']:.1f} ms", "y"), ("target", "y"), (f"The funded 8k-win gate is a T=8,192 step at or below {KERNEL_CAMPAIGN_20260824['target_8k_win_gate_ms']:.2f} ms — beating the transformer — which needs ×{KERNEL_CAMPAIGN_20260824['target_8k_gap_remaining']:.2f} more, i.e. 44% of the step still to remove. Named levers: {KERNEL_CAMPAIGN_20260824['target_levers']}. Memory target ×{KERNEL_CAMPAIGN_20260824['target_mem_ratio']:.2f} (parity or below). NO RECEIPT behind any of this — it is the program plan, and it is what the Ceiling scenario's remaining factor is", "")],
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
floored by the least-reduced component. Both scenarios price the full workload at a 262k-token
average context with the aggregate-decode estimate folded in (2.5 ms/token × 64 streams): **Current
kernels** carries prefill at the banked ×3.94 (inference lever ~×368), **Optimized kernels** adds the
remaining ×1.79 target (~×382). Inference cost does not scale with model size; the ×4.22 smaller-model
ratio applies to training only. Only those ◆ inputs differ between the scenarios. At these levers
inference is limited by tokens per GPU (H5 × H6 ≈ ×368), not by memory (H4, ×2,032); either way ~99.7%
of the inference chip bill is gone, so the scenarios land within rounding in dollars. Multiplying the levers is *not* physical: cost is
additive, not multiplicative. The ×7.03 maturity factor was a flat ×5.5 assumption until 2026-08-24;
it is now **×3.94 measured** (already banked; 2k clean-wall position of record 2026-08-29, 101.737 vs
59.268 ms) × **×1.79 target** (the funded 8k-win gate).

**The measured workload mix (background).** A single compute number cannot represent training,
prefill and decode: at today's kernel maturity they point in *opposite* directions. Training is still a
loss at short sequence, but a much smaller one since the 2026-08-24..29 kernel re-base: the transformer is
**×1.72 cheaper per step at T=2,048** (the 2026-08-29 clean-wall position of record) and **×1.79 at
T=8,192** (B4, artifact-flagged); the B4/T32,768 ~parity cell is retired as a batch-occupancy artifact
and the flat per-token model puts the crossover at **T≈13k** (PROJECTED; measured 21–24k @ B4 on d4096,
from T≈88k pre-campaign). (It read ×6.12 on the morning of 2026-08-24,
and ×5.3–×9.6 on the older d1152 sweep.) Prefill crosses over near 65k context. Decode, **re-based
2026-08-14**, crosses over near
**30k context** — and only on per-GPU aggregate throughput, never per token. So the lever is computed
from the workload rather than asserted: at the default operating point — **10:1 input:output, 64k
average context, training excluded** — prefill ×1.17 and decode ×2.20 blend to **×2.19**, giving
**~5.3× cost-weighted**. Decode is ~99.7% of transformer serving cost at that point, which is why it
dominates the blend.

That blend compares the two families at **matched size**. The sealed 2026-08-14 refit adds the missing
axis — how big each family has to be for the *same quality*. Four ladder rungs per family (47M–663M
params) fitted as `ln(bpb) = a + b·ln(params)` cross at **392M**, and above the crossing bAttention
reaches the transformer fit's quality on **84.2% of the parameters at 1B, 55.2% at 10B, 36.2% at 100B,
23.7% at 1T, 15.5% at 10T**. Per-token cost is ~linear in parameters, so both scenarios multiply the
serving lever by that equal-quality ratio (×4.22 at 1T). The receipt's own words for that factor:
*a projection of the two fits, not a measurement* — every point past ~1B extrapolates beyond the
measured rungs. (The pre-campaign composite ×2.19 × ×4.22 = ×9.24 was the app's Today lever until
2026-09-01; it is retired — measured on kernels that no longer exist.)

The counterweight is measured and points the other way, though it moved a long way over 2026-08-24..29: on
one GH200, one layer, fwd+bwd, bf16, checkpointing off both arms, against a *modern* transformer block
(24 query / 4 KV heads, head_dim 256, RoPE 64, gated attention), a bAttention training step costs
**×1.72 more at T=2,048** (clean-wall 101.737 ± 0.609 vs 59.268 ± 0.381 ms, n=8 — the 2026-08-29
position of record) and **×1.79 at T=8,192** (148.4 vs 83.1 ms, B4, artifact-flagged) — forward ×1.82,
backward ×2.35 (the 2026-08-24 split; the re-anchored 2k cell has no new split). It cost 400.4 ms on
the morning of 2026-08-24 (×6.12), so the megakernel campaign banked **×3.94** (×2.86 of it in a day);
the older ×6.46 figure (8,979 vs 1,389 ms/step, d1536/27 layers, fp16, 601-step protocol) is a
pre-campaign receipt and is retired as a headline. At equal quality the T=2,048 figure becomes ×0.41 —
i.e. a win — but at *matched size and short context* it is still a loss, so the serving
claim continues to exclude training. Training **peak memory** runs against us the same way and by a
similar amount: **×1.58 at T=2,048** (17,087.5 vs 10,823.6 MiB), ×1.57 at T=8,192, from ×2.83 that
morning. That is *training* memory and is a different quantity from the H4 memory lever above, which
is *serving* state.

**The remaining gap is a funded program, and its numbers are TARGETS.** The 8k-win gate is a T=8,192
step at or below **83.08 ms** — beating the transformer — which needs ×1.79 more, 44% of the step.
Named levers: recompute removal (~−12.8 ms), backward-fold + occupancy redesign (~3× headroom in the
dominant fused kernel), and copies elimination (24.8% of the step). Memory target ×0.88, i.e. parity or
below. None of that has a receipt yet, and it is exactly the factor sitting inside the Optimized-kernels scenario.

Both phases use measured throughput, with the transformer granted an idealized mature stack (paged
attention, KV ÷8, bandwidth-floor serving) — strictly more generous than our measured lane. The memory
lever (H4) is the measured ×2,032 at a 262k conversation, consistent with serving-state math (**43–315×
smaller serving memory at 1M tokens** for frontier geometries) and measured directly as a concurrency
result (below).

**Where this goes negative, stated plainly.** At 100:1 input:output and 8k context the blend is
**×0.28** — the transformer wins. A training share above **~0.30** takes the blend below 1 (0.2 →
**×1.23**); that threshold was ~0.06 (and 0.2 gave ×0.44) before the 2026-08-24 kernel re-base.
Both are reachable in the sidebar, because a lever you cannot push until it breaks is not
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
Infra (data-center) share comes from the 10-K/10-Q property & equipment and segment notes (Amazon = AWS, 68–76%); server share is CFO-disclosed; accelerator-within-server
is ~67–80% from BOM teardowns.

**Totals & global.** The named firms roll up live (no double-count). The **GLOBAL** row grosses the
named total up to a worldwide estimate using *Named share of global AI capex* (the rest = other clouds,
China, neoclouds, xAI, sovereign & enterprise).

**Net AI economics** are cash basis: `AI revenue − AI capex − AI opex`; with the architecture, add the
spend cut. All six firms lose money on AI today. The *Datacenter scaling factor* toggles how much of the
non-accelerator datacenter shrinks too (0 = conservative; ~0.7 ≈ breakeven; 1 = flips positive).

**Key results.** FY25: ~\$357B AI capex vs ~\$79B AI revenue → ~−\$284B/yr burn. At the current-kernels
levers (262k conversations; technical-tab reduction ~×724) the named spend cut is **~\$163B FY25**
(~\$2.7T capitalized at the 6% rate; global est ~\$3.4T FY25) and **~\$385B on FY2026 guidance**
(~\$6.4T / global ~\$8.0T). The front tabs, which price whole GPUs, read ~\$377B for FY2026. Data-center shares are from the 10-K/10-Q property & equipment notes
(2026-09-29). Current and Optimized kernels are within rounding of each other in dollars: the cut is `accelerator capex × (1 − 1/reduction)` and it saturates; the underlying
capex, shares and revenue never move.

**Sensitivity.** `capex_avoided ∝ (1 − 1/R)` is 0.999 at R ≈ 700, so the dollar headline is essentially
insensitive to the compute lever at these levels. The discount rate IS a first-order lever on the
capitalized figures: they scale as 1/rate.

**Caveats.** AI revenue is the softest input (Microsoft \$37B & Amazon \$15B run-rates disclosed; the rest
estimated; Meta's real payoff is indirect ad-uplift). Totals are disclosed; server/accelerator splits are
estimated (±15–20%). Capitalization is a simple perpetuity (benefit ÷ discount rate). Analytical estimate,
not investment advice.

**Sources.** SEC filings & earnings calls (MSFT, GOOGL, AMZN, META, ORCL 10-Ks/transcripts; SpaceX S-1);
BOM/margin teardowns (Silicon Analysts); TPU/Trainium TCO (SemiAnalysis); GPU rental pricing (Spheron).
Per-company source links are on each company tab.
""")


# ---- audience layer: Summary / Value Bridge / Levers / What Matters (mirrors the workbook) ----
LEV = {x["code"]: x for x in HELARCTOS_LEVERS}
_LEVER_TEXT = {  # code: (how sure are we?, what it means, saves money in)
    "H1": ("PROJECTED — quality trends measured on models we trained (47M–663M parameters), extended to "
           "~1T dense-equivalent",
           f"A Helarctos model matches a ~1T dense-equivalent transformer — about what today's 5–6T-total "
           f"mixture-of-experts flagships amount to — with "
           f"~{param_matching_fraction(DECK_DEPLOYMENT_SCALE):.0%} of the parameters: "
           f"{param_matching_gain(DECK_DEPLOYMENT_SCALE):.1f}× fewer numbers to store, update and run.",
           "Training (inference cost doesn't depend on model size)"),
    "H2": ("PROJECTED — standard compute-optimal scaling",
           f"Frontier labs train on data in proportion to model size, so a {param_matching_gain(DECK_DEPLOYMENT_SCALE):.1f}× "
           f"smaller model reaches its best quality on ~{param_matching_gain(DECK_DEPLOYMENT_SCALE):.1f}× fewer tokens.",
           "Training"),
    "H3": ("ASSUMPTION — about the same as a transformer; no credit taken",
           "At the same size Helarctos trains at about the same speed per token, so no speed-up is counted.",
           "Training"),
    "H4": ("MEASURED ×2,032 at 262k (on our test model). Frontier-size models with grouped-query attention: "
           f"~÷{fleet_memory_lever(262144):.0f} (estimate), where memory would bind again (about −$0.5B)",
           "A transformer's memory (the KV cache) grows with every token of every live conversation — 51.6 GB "
           "for one 262k-token conversation. Helarctos keeps a fixed-size state (25.4 MB).",
           "Inference GPUs (the memory limit; not binding at 262k)"),
    "H5": ("MEASURED — 64 conversations in 1.6 GB on one GPU (a grid cap); the transformer's 2nd OOMs",
           "The small memory (H4) lets one GPU hold 64 resident 262k-token conversations at once, where a "
           "transformer fits 1.",
           "Inference GPUs (tokens per GPU)"),
    "H6": ("ESTIMATE — 2.5 ms per token after the kernel edits (receipt pending); MEASURED ×2.6 on our older "
           "kernels",
           "Each conversation's next token takes 2.5 ms instead of the transformer's 15.6 ms at 262k: no growing "
           "cache to re-read for every token.",
           "Inference GPUs (tokens per GPU)"),
}


def lever_fmt(code, v):
    return f"÷{v:,.0f}" if code == "H4" else (f"×{v:,.0f}" if code == "H5" else f"×{v:.2f}")  # H6: ×6.25


def lever_value_code(code):
    """Style code for a lever's value cell: purple if it differs by scenario, ★ border if high impact."""
    return ("p" if LEV[code]["by_scenario"] else "b") + ("*" if LEV[code]["high_impact"] else "")


# One color per bar (categorical slots, validated for adjacent-pair CVD separation;
# two sit below 3:1 on white, so every bar carries its value label).
SOURCE_COLORS = ["#2a78d6", "#eb6834"]  # training, inference (each incl. its power)
LADDER_COLORS = ["#a3a29c", "#2a78d6", "#eb6834", "#1baf7a"]  # baseline gray, then one hue per lever step
HELARCTOS_BAR, MARKET_BAR = "#1baf7a", "#a3a29c"


def _colored_bars(labels, values, colors, horizontal=False):
    import altair as alt
    df = pd.DataFrame({"label": labels, "value": values})
    cat = alt.X if horizontal else alt.Y
    lab = (alt.Y if horizontal else alt.X)("label:N", sort=None, title=None,
                                           axis=alt.Axis(labelLimit=420, labelAngle=0))
    val = cat("value:Q", title="$B per year", axis=alt.Axis(grid=True, gridOpacity=0.3))
    color = alt.Color("label:N", sort=None, legend=None,
                      scale=alt.Scale(domain=labels, range=colors[:len(labels)]))
    bars = alt.Chart(df).mark_bar(cornerRadiusEnd=4, size=26).encode(
        lab, val, color, tooltip=[alt.Tooltip("label:N", title=""),
                                  alt.Tooltip("value:Q", title="$B/yr", format=",.0f")])
    text = alt.Chart(df).mark_text(align="left" if horizontal else "center",
                                   baseline="middle" if horizontal else "bottom",
                                   dx=4 if horizontal else 0, dy=0 if horizontal else -4,
                                   color="#52514e").encode(lab, val, text=alt.Text("value:Q", format="$,.0f"))
    st.altair_chart((bars + text).properties(height=max(60 * len(labels), 160) if horizontal else 320),
                    width="stretch")


def _train_shares():
    out = {}
    for c in COMPANIES:
        k = f"ts_{c['name']}"
        st.session_state.setdefault(
            k, float(CAMPAIGN_LANDED_20260831["train_share_by_company"].get(
                c["name"], CAMPAIGN_LANDED_20260831["train_share"])))
        out[c["name"]] = float(st.session_state[k])
    return out


def _bridge_levers(g):
    kernels = "mature" if st.session_state.get("scenario") == "Optimized kernels" else "current"
    return value_bridge_levers(g, kernels=kernels)


def summary_tab(comps, g):
    lv, ts = _bridge_levers(g), _train_shares()
    rows26, t26 = value_bridge(g, comps, "fy26", lv, ts)
    _, t25 = value_bridge(g, comps, "fy25", lv, ts)
    spend26 = t26["ai_capex"] + t26["ai_opex"]
    st.markdown(
        f"In FY2026 Microsoft, Alphabet, Amazon, Meta, Oracle and SpaceX will spend **{md_usd(spend26)}B** on AI "
        f"and earn **{md_usd(t26['ai_rev'])}B** from it. Helarctos models do the same AI work — training and "
        f"inference at the same quality — on far fewer GPUs. This page shows how much of that spend becomes "
        f"unnecessary, and which Helarctos levers drive it.")
    st.caption(f"Scenario: **{st.session_state.get('scenario')}** (◆ switch it in the sidebar) · "
               f"{ctx_k(st.session_state.get('context_tokens', 262144))}-token average conversation · "
               f"$B per year unless stated")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Spend Helarctos makes unnecessary", f"{bn(t26['spend_cut'])}/yr",
              delta=f"{pct(t26['pct_cut'])} of AI spend")
    k2.metric("AI spend (FY26)", f"{bn(spend26)}")
    k3.metric("Net AI cash result with Helarctos", bn(t26["net_with"]).replace("$", "\\$"),
              delta=f"today: {bn(t26['net_now'])}", delta_color="off")
    k4.metric(f"Capitalized value (at {g['discount_rate']:.0%})", f"${t26['spend_cut'] / g['discount_rate'] / 1000:.1f}T")

    section("The headline")
    show_table(["", "FY2026", "FY2025"], [
        [("AI spend: chips, data centres, power", ""), (bn(t26["ai_capex"] + t26["ai_opex"]), "b"),
         (bn(t25["ai_capex"] + t25["ai_opex"]), "b")],
        [("AI revenue", ""), (bn(t26["ai_rev"]), "b"), (bn(t25["ai_rev"]), "b")],
        [("Net AI cash result today", ""), (bn(t26["net_now"]), "b"), (bn(t25["net_now"]), "b")],
        [("Spend Helarctos makes unnecessary, per year", "!"), (bn(t26["spend_cut"]), "k!"),
         (bn(t25["spend_cut"]), "k!")],
        [("…as a share of all AI spend", "!"), (pct(t26["pct_cut"]), "k!"), (pct(t25["pct_cut"]), "k!")],
        [("Net AI cash result with Helarctos", ""), (bn(t26["net_with"]), "b"),
         (bn(t25["net_with"]), "b")],
        [("Yearly saving, capitalized at the discount rate", ""),
         (f"${t26['spend_cut'] / g['discount_rate'] / 1000:.1f}T", "b"),
         (f"${t25['spend_cut'] / g['discount_rate'] / 1000:.1f}T", "b")],
    ], wrap=True)

    section("Where the FY2026 saving comes from — and the Helarctos levers behind it")
    tot = t26["spend_cut"]
    lever_rows = {"Training": ["H1", "H2", "H3"], "Inference": ["H4", "H5", "H6"]}
    lever_text = {"H1": "Same quality with ~24% of the parameters (projected from models we trained).",
                  "H2": "Compute-optimal training needs data in proportion to model size.",
                  "H3": "About the same as a transformer at the same size — no credit taken.",
                  "H4": "Fixed-size state instead of a memory that grows with every token (measured ×2,032 at "
                        "262k on our test model).",
                  "H5": "64 resident 262k-token conversations per GPU where a transformer fits 1 (measured).",
                  "H6": "Each next token in 2.5 ms instead of ~15.6 ms (estimate; ×2.6 measured on older kernels)."}
    body = []
    for fleet, val, lever, how in (
        ("Training", t26["training_total"], f"×{lv['train_lever']:.1f}",
         f"H1 × H2 × H3 = ~{lv['train_lever']:.0f}× fewer GPU-hours per training run, so training clusters can be "
         f"that much smaller. The GPUs never bought are capex avoided; their power is saved every year."),
        ("Inference", t26["inference_total"], f"×{lv['serving_gpu_lever']:.0f}",
         f"Tokens per GPU rise ~{lv['serving_compute_lever']:.0f}× (H5 × H6, with prompts added back). A GPU is "
         f"bought whole, so the inference fleet shrinks by the smaller of memory and tokens per GPU: "
         f"~{lv['serving_gpu_lever']:.0f}× fewer GPUs, plus the power they would draw."),
    ):
        body.append([(f"{fleet} fleet (capex + power)", "!"), (bn(val), "k!"), (pct(val / tot), "b"),
                     (lever, "b!"), (how, "")])
        for code in lever_rows[fleet]:
            body.append([("↳ " + lever_label(code), "h"), ("", ""), ("", ""),
                         (lever_fmt(code, lv[LEV[code]["key"]]), lever_value_code(code)),
                         (lever_text[code], "")])
    body.append([("TOTAL", "s!"), (bn(tot), "s!"), ("100%", "s!"), ("", "s"),
                 ("Step-by-step build and the one-lever-at-a-time view: Value Bridge tab.", "s")])
    show_table(["Fleet / Helarctos lever", "$B per year", "Share", "Lever", "How it works"], body, wrap=True)
    _colored_bars(["Training", "Inference"], [t26["training_total"], t26["inference_total"]], SOURCE_COLORS,
                  horizontal=True)

    section("By company — FY2026, $B per year")
    show_table(["Company", "AI spend", "Training saving", "Inference saving", "Total saving",
                "% of AI spend", "Net AI today", "Net AI with Helarctos"],
               [[(r["name"], "s!" if r is t26 else ""), (bn(r["ai_capex"] + r["ai_opex"]), "s" if r is t26 else "b"),
                 (bn(r["training_total"]), "s" if r is t26 else "b"), (bn(r["inference_total"]), "s" if r is t26 else "b"),
                 (bn(r["spend_cut"]), "s!" if r is t26 else "b!"), (pct(r["pct_cut"]), "s" if r is t26 else "b"),
                 (bn(r["net_now"]), "s" if r is t26 else "b"), (bn(r["net_with"]), "s!" if r is t26 else "b")]
                for r in rows26 + [t26]], wrap=True)

    section('Why ~99% of the chip bill — not "1,000×"')
    st.markdown(f"Multiples don't multiply. A GPU is bought whole, so the inference fleet shrinks by whichever "
                f"need falls least (×{lv['serving_gpu_lever']:,.0f}), never by memory × tokens per GPU "
                f"({lv['memory']:,.0f} × {lv['serving_compute_lever']:,.0f}). Cutting a fleet "
                f"{lv['serving_gpu_lever']:,.0f}× already removes over 99% of it; bigger multiples only move the last "
                f"fraction of a percent. So the dollars are set by how much these firms "
                f"spend on chips — which is why this model reports dollars.")

    section("How to read this app")
    st.markdown(LEGEND_MD, unsafe_allow_html=True)
    st.caption("Teal = one of the six things the Helarctos architecture changes (H1–H6); every other input is "
               "market, company or modelling data. ◆ = the only values that change between Current kernels and "
               "Optimized kernels. ★ = moves the FY2026 saving by \\$10B or more (What Matters tab). Cash basis; "
               "FY2026 = company guidance; only AI chips and their power are counted (buildings and power "
               "infrastructure are upside). Not investment advice.")


def value_bridge_tab(comps, g):
    lv, ts = _bridge_levers(g), _train_shares()
    rows, t = value_bridge(g, comps, "fy26", lv, ts)
    st.caption("Read top to bottom. Teal rows are the Helarctos levers (edit them in the sidebar and on the "
               "Levers tab); blue cells are formulas.")
    section("Step 1 — What the six companies spend on AI this year (FY2026)")
    show_table(["Item", "$B", "Note"], [
        [("AI data-centre capex", ""), (bn(t["ai_capex"]), "b"), ("company filings and guidance", "")],
        [("…of which AI chips (GPUs, TPUs)", "!"), (bn(t["accel"]), "b!"),
         ("the part Helarctos shrinks; buildings, power and networking are not counted (upside)", "")],
        [("Power & operations for those chips, per year", ""), (f"${t['ai_opex']:,.1f}B", "b"), ("", "")],
        [("AI revenue", ""), (bn(t["ai_rev"]), "b"),
         ("disclosed run-rates (Microsoft, Amazon) or estimates (others)", "")],
        [("Net AI cash result today", "!"), (bn(t["net_now"]), "b!"), ("revenue − capex − power", "")],
    ], wrap=True)
    section("Step 2 — Split the chip fleet by what it does")
    show_table(["Fleet", "$B", "Share", "Note"], [
        [("Training fleet — builds new models", ""), (bn(t["train_fleet"]), "b"), (pct(t["train_share"]), "b"),
         ("per-company estimates, 25–55% (Levers tab); chip-weighted average", "")],
        [("Inference fleet — answers users", ""), (bn(t["serve_fleet"]), "b"),
         (pct(1 - t["train_share"]), "b"), ("", "")],
    ], wrap=True)

    def lever_row(code, note):
        return [(lever_label(code), "h"), ("", ""), (lever_fmt(code, lv[LEV[code]["key"]]), lever_value_code(code)),
                (note, "")]

    section("Step 3 — Training: a smaller model needs fewer GPU-hours → smaller training clusters")
    show_table(["Item", "$B", "Lever", "Note"], [
        [("Training fleet capex today", ""), (bn(t["train_fleet"]), "b"), ("", ""), ("", "")],
        lever_row("H1", "same quality with a fraction of the parameters (projected)"),
        lever_row("H2", "a smaller model needs proportionally fewer training tokens"),
        lever_row("H3", "about the same speed per token at the same size — no credit taken"),
        [("GPU-hours per training run fall by", "!"), ("", ""), (f"×{lv['train_lever']:.1f}", "b!"), ("H1 × H2 × H3", "")],
        [("Training fleet needed with Helarctos", ""), (bn(t["train_fleet"] / lv["train_lever"]), "b"), ("", ""), ("", "")],
        [("Training capex avoided", ""), (bn(t["train_saved"]), "b"), ("", ""), ("", "")],
        [("+ power & operations those GPUs would have drawn", ""), (f"${t['train_power_saved']:,.1f}B", "b"),
         ("", ""), ("power scales with the fleet", "")],
        [("TRAINING SAVING (capex + power)", "!"), (bn(t["training_total"]), "k!"), ("", ""),
         (f"labs size training clusters to GPU-hours: ~{lv['train_lever']:.0f}× fewer GPU-hours means a cluster "
          f"~{lv['train_lever']:.0f}× smaller for the same programme", "")],
    ], wrap=True)
    section("Step 4 — Inference: small memory → more conversations per GPU, each decoded faster → fewer GPUs")
    show_table(["Item", "$B", "Lever", "Note"], [
        [("Inference fleet capex today", ""), (bn(t["serve_fleet"]), "b"), ("", ""), ("", "")],
        lever_row("H4", "fixed-size state instead of a memory that grows with every token"),
        lever_row("H5", "64 resident 262k-token conversations per GPU where a transformer fits 1"),
        lever_row("H6", "each next token in 2.5 ms instead of ~15.6 ms (estimate)"),
        [("Tokens per GPU, H5 × H6 with prompt processing added back", ""), ("", ""),
         (f"×{lv['serving_compute_lever']:,.0f}", "b"),
         (f"prompts are ~0.2% of a transformer's time at 262k (prompt-processing speed ×{lv['prefill_speed']:.1f}, "
          f"the only inference input that differs by scenario)", "")],
        [("Inference GPUs needed fall by (the smaller of memory and tokens per GPU)", "!"), ("", ""),
         (f"×{lv['serving_gpu_lever']:,.0f}", "b!"),
         ("a GPU is bought whole — memory and compute together — so the fleet covers whichever runs out first; "
          "here tokens per GPU binds, so faster decode (H6) counts in the dollars", "")],
        [("Inference fleet needed with Helarctos", ""), (bn(t["serve_fleet"] / lv["serving_gpu_lever"]), "b"), ("", ""), ("", "")],
        [("Inference capex avoided", ""), (bn(t["serve_saved"]), "b"), ("", ""), ("", "")],
        [("+ power & operations those GPUs would have drawn", ""), (f"${t['infer_power_saved']:,.1f}B", "b"),
         ("", ""), ("power scales with the fleet", "")],
        [("INFERENCE SAVING (capex + power)", "!"), (bn(t["inference_total"]), "k!"), ("", ""), ("", "")],
    ], wrap=True)
    section("Result — FY2026")
    show_table(["Item", "$B"], [
        [("Training (capex + power)", ""), (bn(t["training_total"]), "b")],
        [("Inference (capex + power)", ""), (bn(t["inference_total"]), "b")],
        [("SPEND HELARCTOS MAKES UNNECESSARY, per year", "!"), (bn(t["spend_cut"]), "k!")],
        [("…as a share of all AI spend", "!"), (pct(t["pct_cut"]), "b!")],
        [("…as a share of the AI-chip bill", ""), (pct(t["capex_avoided"] / t["accel"]), "b")],
        [("Net AI cash result with Helarctos", "!"), (bn(t["net_with"]), "b!")],
        [("Value of the yearly saving, capitalized (÷ discount rate)", ""),
         (f"${t['spend_cut'] / g['discount_rate'] / 1000:.1f}T", "b")],
    ], wrap=True)
    section("Switch the Helarctos levers on one at a time (FY2026)")
    lad = savings_ladder(g, comps, "fy26", lv, ts)
    labels = [s_["step"].replace(" (training)", " — training (H1–H3)").replace(" (inference)", " — inference (H4–H6)")
              for s_ in lad]
    show_table(["Step", "Training GPU-hours fall by", "Inference memory falls by", "Inference compute falls by",
                "Chip fleet cut", "Spend cut, $B/yr", "Added by this step"],
               [[(labels[i], "!" if i == len(lad) - 1 else ""), (f"×{s_['train_lever']:.1f}", "b"),
                 (f"÷{s_['memory']:.0f}", "b"), (f"×{s_['serving_compute_lever']:,.0f}", "b"),
                 (f"{s_['fleet_cut']:.1%}", "b"), (bn(s_["spend_cut"]), "b!"),
                 (f"+${s_['increment']:,.0f}B" if i else "—", "b")]
                for i, s_ in enumerate(lad)], wrap=True)
    _colored_bars([f"{i}. {lab}" for i, lab in enumerate(labels)], [s_["spend_cut"] for s_ in lad],
                  LADDER_COLORS, horizontal=True)
    st.caption("Training savings come from the smaller model (fewer parameters AND fewer tokens compound). "
               "Inference savings come from fixed-size memory, which lets each GPU hold many more long "
               "conversations, each decoded faster. Each fleet then shrinks by ~94–99%: bigger multiples can't add much, because a "
               "cost can only fall to zero once. That is why this model reports dollars, not multiples.")
    _, tt = compute_year(g, comps, "fy26")
    st.caption(f"Cross-check: the technical Totals tab prices memory and compute separately — "
               f"{md_usd(tt['spend_cut'])}B FY26 vs {md_usd(t['spend_cut'])}B here. Small by design: both remove "
               f"~99% of the chip bill.")


def _scenario_levers(g):
    """Lever sets for both scenarios at the sidebar workload (◆ = differs)."""
    w = {"context_tokens": int(st.session_state.get("context_tokens", 262144)),
         "in_out_ratio": float(st.session_state.get("in_out_ratio", WORKLOAD["in_out_ratio"]))}
    rq = prompt_time_ratio(w["context_tokens"], w["in_out_ratio"])
    cur, opt = (max(1.0, inference_throughput(g["conv_per_gpu"], g["decode_speedup"],
                                              prefill_advantage(k, w["context_tokens"]), rq))
                for k in ("current", "mature"))
    return (value_bridge_levers(dict(g, flop_factor=cur), kernels="current"),
            value_bridge_levers(dict(g, flop_factor=opt), kernels="mature"))


def levers_tab(comps, g):
    lv = _bridge_levers(g)
    cur, opt = _scenario_levers(g)
    st.caption("Only the six teal levers (H1–H6) are about the Helarctos architecture. Everything else is "
               "market or company data, listed further down. ◆ purple = differs between the two scenarios · "
               "★ orange border = high-impact input (What Matters tab).")
    section(f"Helarctos levers — the six things the architecture changes · active scenario: "
            f"{st.session_state.get('scenario')} (sidebar)")
    rows = []
    for x in HELARCTOS_LEVERS:
        code, k = x["code"], x["key"]
        status, what, where = _LEVER_TEXT[code]
        differs = x["by_scenario"]
        rows.append([(code, "h"), (lever_label(code).split(" · ", 1)[1], "h"),
                     (lever_fmt(code, lv[k]), lever_value_code(code)),
                     (lever_fmt(code, cur[k]) if differs else "same", "p" if differs else ""),
                     (lever_fmt(code, opt[k]) if differs else "same", "p" if differs else ""),
                     (status, ""), (what, ""), (where, "")])
    rows.append([("", ""), ("Supporting: prompt-processing speed vs the transformer ◆", ""),
                 (f"×{lv['prefill_speed']:.1f}", "p"), (f"×{cur['prefill_speed']:.1f}", "p"),
                 (f"×{opt['prefill_speed']:.1f}", "p"),
                 ("MEASURED ×3.94 kernel speed-up banked; the full ×7.03 is a TARGET", ""),
                 ("A Helarctos kernel property, not a headline lever: prompts are ~0.2% of a transformer's time at "
                  "262k, so it barely moves the dollars.", ""), ("Inference GPUs (tokens per GPU)", "")])
    show_table(["#", "Helarctos lever", "Active value", "◆ Current kernels", "◆ Optimized kernels",
                "How sure are we?", "What it means", "Saves money in"], rows, wrap=True)
    section("How the levers combine — one number per fleet")
    show_table(["Fleet lever", "Value", "Built from"], [
        [("Training: GPU-hours per training run fall by", "!"), (f"×{lv['train_lever']:.1f}", "b!"),
         ("H1 × H2 × H3 — training clusters are sized to GPU-hours", "")],
        [("Inference: memory per conversation falls by", "!"), (f"÷{lv['memory']:,.0f}", "b!"), ("H4", "")],
        [("Inference: decode tokens per GPU rise by", "!"),
         (f"×{lv['conversations_per_gpu'] * lv['faster_decode']:,.0f}", "b!"),
         ("H5 × H6: more conversations per GPU, each decoded faster", "")],
        [("Inference: tokens per GPU, including prompt processing", "!"),
         (f"×{lv['serving_compute_lever']:,.0f}", "b!"),
         ("H5 × H6 with prompts added back at the supporting prompt-processing speed (inference cost doesn't "
          "depend on model size)", "")],
        [("Inference: GPUs needed fall by", "!"), (f"×{lv['serving_gpu_lever']:,.0f}", "b!"),
         (f"the smaller of memory (H4) and tokens per GPU — a GPU is bought whole, so the fleet covers whichever "
          f"runs out first (here tokens per GPU, so faster decode counts); multiplying them "
          f"({lv['memory']:,.0f} × {lv['serving_compute_lever']:,.0f}) would be wrong", "")],
        [("For reference: technical tabs' blended cut", ""), (f"×{reduction_factor(g):,.0f}", "b"),
         ("the appendix prices memory (~60% of a GPU's cost) and compute separately; same chip bill, within ~2%", "")],
    ], wrap=True)
    section("Market & company data — the other inputs (not about Helarctos)")
    _, t = value_bridge(g, comps, "fy26", lv, _train_shares())
    show_table(["Input", "Value", "Where to edit", "Note"], [
        [("AI data-centre capex, FY2026 (six companies)", ""), (bn(t["ai_capex"]), "b"),
         ("✏️ panel on each company tab ★", ""), ("disclosed / guided capex × data-centre share (10-K/10-Q notes)", "")],
        [("…share that buys AI chips", ""), (pct(t["accel"] / t["ai_capex"]), "b"),
         ("✏️ panel on each company tab ★", ""), ("server share (CFO-disclosed) × accelerator share of servers", "")],
        [("Training share of the chip fleet (chip-weighted)", ""), (pct(t["train_share"]), "b"),
         ("per company, below", ""), ("analyst estimates; no company discloses this", "")],
        [("Fully-loaded cost per GPU", ""), (usd0(g["gpu_cost"]), "b"), ("sidebar", ""),
         ("sizes the fleet for the power calculation", "")],
        [("Wall power per GPU (kW)", ""), (f"{g['wall_power_kw']:.1f}", "b"), ("sidebar", ""),
         ("power is ~4% of the saving", "")],
        [("Electricity ($/kWh)", ""), (f"${g['elec_rate']:.2f}", "b"), ("sidebar", ""), ("power is ~4% of the saving", "")],
        [("Discount rate ★ (capitalized value only)", ""), (pct(g["discount_rate"]), "b*"), ("sidebar", ""),
         ("changes only the capitalized value, never the yearly saving", "")],
    ], wrap=True)
    section("Training share of each company's AI-chip fleet — edit (market estimate)")
    st.caption("Estimates: no company discloses this. Analysts put training at 30–45% of AI compute in 2026 "
               "(Gartner, Deloitte); the rest is inference, i.e. serving users.")
    _train_shares()
    cols = st.columns(len(COMPANIES))
    for col, c in zip(cols, COMPANIES):
        col.number_input(c["name"], min_value=0.0, max_value=1.0, step=0.05, format="%.2f",
                         key=f"ts_{c['name']}")
    section("Fine print")
    st.markdown("- Only AI chips and the power they draw are counted. Buildings, power infrastructure and "
                "networking around them would shrink too — upside, not in these numbers.\n"
                "- Training is priced on GPU-hours only; no memory credit is taken on training clusters.\n"
                "- H4 (×2,032) is measured on our test model; for frontier-size models with grouped-query attention "
                "the estimate is ~×208, where memory would bind again (about −\\$0.5B on FY2026). H6 is an estimate "
                "until the aggregate-decode receipt lands (×2.6 measured on our older kernels).\n"
                "- H1 is a projection: quality trends measured up to 663M parameters, extended to ~1T "
                "dense-equivalent — about what today's 5–6T-total mixture-of-experts flagships (Grok 5 at 6T, "
                "Kimi K3 at 2.8T) amount to.\n"
                "- The saving is spend no longer needed for the same AI output; firms will likely reinvest it. "
                "Cash basis; capitalized = yearly saving ÷ discount rate.")


WM_WHY = {
    "Server share of AI capex": "How much AI capex buys servers rather than buildings, power and networking; the "
        "dollars move one-for-one with it. CFO-disclosed for Microsoft and Alphabet.",
    "Accelerator share of servers": "The GPU/TPU share of server spend — same one-for-one effect. From teardowns "
        "(67–80%).",
    "Smaller model for the same quality": "The biggest Helarctos-specific assumption: it drives the whole training "
        "saving. A projection from models up to 663M parameters to ~1T dense-equivalent (today's 5–6T "
        "mixture-of-experts flagships). High case: a 5T dense model.",
    "More conversations per GPU": "64 vs the transformer's 1 (measured). Small effect: inference GPUs already "
        "shrink by ~99.7%, so even ×16 costs under $2B.",
    "Faster decode per token": "×6.3 estimate after the kernel edits (×2.6 measured on older kernels). It now counts "
        "in the dollars, but only ~$1B across the range.",
    "Memory per conversation": "Measured ×2,032 at 262k on our test model. Frontier-size models with grouped-query "
        "attention: ~÷208 (estimate), where memory would bind again (≈ −$0.5B).",
    "Average conversation length": "Shorter conversations shrink both the memory advantage and tokens per GPU "
        "(×48 at 32k); longer ones grow them (×1,411 tokens per GPU at 1M, where one transformer conversation no "
        "longer fits a GPU). Set it in the sidebar.",
    "Data-center share of capex": "Share of capex that goes into data centres (10-K/10-Q property notes, 93–98%; "
        "Amazon's AWS share 68–76%).",
    "Training speed per token": "Set to ×1. Small effect: the smaller model already removes ~94% of training "
        "GPU-hours.",
    "Electricity rate": "Only changes the power part of the saving (~4% of the total).",
    "Wall power per GPU": "Only changes the power part of the saving (~4% of the total).",
    "Training share of the chip fleet": "Small effect: both fleets shrink by 94–99%.",
    "Memory share of GPU cost": "No effect on the front tabs (GPUs are bought whole).",
}


def wm_label(name, code):
    if not code:
        return name
    return "H1 + H2 · Smaller model, trained on fewer tokens ★" if code == "H1+H2" else lever_label(code)


def what_matters_tab(comps, g):
    import altair as alt
    from ai_capex_model import sensitivity_table
    _, t = value_bridge(g, comps, "fy26", _bridge_levers(g), _train_shares())
    st.caption("Each row moves ONE input to a plausible low and high value, everything else as set, and shows "
               f"the change in the FY2026 saving ({md_usd(t['spend_cut'])}B/yr). ★ = moves it by \\$10B or more.")
    rows = sensitivity_table(g, comps)
    section(f"What moves the FY2026 saving ({bn(t['spend_cut'])}/yr) — Helarctos levers vs market data")
    body, chart = [], []
    for gname, items, gcode in (("Helarctos levers — what the architecture changes", [r for r in rows if r[2]], "h"),
                                ("Market & company data — not about Helarctos", [r for r in rows if not r[2]], "s")):
        body.append([("", gcode), (gname, gcode + "!"), ("", gcode), ("", gcode), ("", gcode), ("", gcode),
                     ("", gcode), ("", gcode)])
        for name, where, code, lo_l, lo, hi_l, hi in items:
            big = max(abs(lo), abs(hi)) >= 10.0
            lab = wm_label(name, code)
            body.append([("★" if big else "", ""), (lab, "h" if code else ("!" if big else "")), (where, ""),
                         (lo_l, ""), (f"{lo:+,.1f}", "b*" if big else "b"), (hi_l, ""),
                         (f"{hi:+,.1f}", "b*" if big else "b"), (WM_WHY.get(name, ""), "")])
            chart.append({"input": lab, "lo": lo, "hi": hi,
                          "kind": "Helarctos lever" if code else "Market & company data"})
    show_table(["★", "Input", "Where to edit (workbook)", "Low case", "FY26 change, low ($B)", "High case",
                "FY26 change, high ($B)", "Why it matters"], body, wrap=True)
    df = pd.DataFrame(chart)
    order = list(df["input"])
    long = pd.concat([df.assign(v=df["lo"], case="Low case"), df.assign(v=df["hi"], case="High case")])
    y = alt.Y("input:N", sort=order, title=None, axis=alt.Axis(labelLimit=420))
    color = alt.Color("kind:N", title=None, legend=alt.Legend(orient="top"),
                      scale=alt.Scale(domain=["Helarctos lever", "Market & company data"],
                                      range=[HELARCTOS_BAR, MARKET_BAR]))
    bars = alt.Chart(long).mark_bar(size=18).encode(
        y, alt.X("v:Q", title="Change in the FY2026 saving, $B (left = low case, right = high case)"), color,
        tooltip=["input", "case", alt.Tooltip("v:Q", format="+,.1f")])
    lbl = alt.Chart(long).transform_filter("datum.v != 0").mark_text(
        dx=alt.expr("datum.v < 0 ? -4 : 4"), align=alt.expr("datum.v < 0 ? 'right' : 'left'"),
        color="#52514e").encode(y, alt.X("v:Q"), text=alt.Text("v:Q", format="+$,.0f"))
    zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color="#52514e").encode(x="x:Q")
    st.altair_chart((bars + lbl + zero).properties(height=32 * len(df) + 40), width="stretch")
    st.caption(f"★ The discount rate changes only the capitalized value: "
               f"~\\${t['spend_cut'] / 0.04 / 1000:.1f}T at 4%, ~\\${t['spend_cut'] / 0.06 / 1000:.1f}T at 6%, "
               f"~\\${t['spend_cut'] / 0.10 / 1000:.1f}T at 10%. **Takeaway:** the answer is driven by how much "
               "these firms spend on AI chips (the market & company shares) more than by how large the Helarctos "
               "multiples are; the Helarctos levers move it only if they fall far below today's values.")


# ---- main ----------------------------------------------------------------------
st.title("AI Capex Efficiency")
st.caption("How much AI data-centre spend the Helarctos architecture makes unnecessary for the six biggest AI "
           "spenders — a tab-for-tab mirror of the workbook. Set the scenario and the levers in the sidebar.")
st.markdown(LEGEND_MD, unsafe_allow_html=True)

g = sidebar_globals()
FRONT = ["Summary", "Value Bridge", "Levers", "What Matters"]
names = FRONT + [c["name"] for c in COMPANIES] + ["Totals", "Inputs", "Sensitivity", "CostLadder", "Serving·Training", "Evidence", "Methodology"]
T = st.tabs(names)
nco = len(COMPANIES)
nf = len(FRONT)
comps = [dict(c) for c in COMPANIES]

# company tabs first so their edits are captured before the roll-ups compute
for i, c in enumerate(comps):
    with T[nf + i]:
        company_tab(c, g)
with T[2]:
    levers_tab(comps, g)
with T[0]:
    summary_tab(comps, g)
with T[1]:
    value_bridge_tab(comps, g)
with T[3]:
    what_matters_tab(comps, g)
with T[nf + nco]:
    totals_tab(comps, g)
with T[nf + nco + 1]:
    inputs_tab(g)
with T[nf + nco + 2]:
    sensitivity_tab(comps, g)
with T[nf + nco + 3]:
    costladder_tab(g)
with T[nf + nco + 4]:
    serving_training_tab(g)
with T[nf + nco + 5]:
    evidence_tab(g)
with T[nf + nco + 6]:
    methodology_tab(g)
