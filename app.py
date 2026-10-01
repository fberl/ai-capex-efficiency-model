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
                            CAMPAIGN_LANDED_20260831,
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
                            inference_throughput, fleet_memory_lever,
                            deck_layer_levers, DECK_LAYER_CARD_20261001,
                            TRAIN_SPEED_BY_SCENARIO, INFERENCE_SIZE_FACTOR,
                            INFERENCE_LEVER_EQUAL_SIZE, INFERENCE_LEVER_CURRENT,
                            INFERENCE_LEVER_OPTIMIZED, H5_CONVERSATIONS, H6_DECODE,
                            BRIDGE_FOLLOW_20261001, BRIDGE_FOLLOW_HELD, headline_family,
                            DECODE_GEOMETRY_OF_RECORD)

GEOM = DECODE_GEOMETRY_OF_RECORD  # the ~90-layer grouped-query frontier geometry the decode levers are quoted at
GEOM_TXT = (f"a ~{GEOM['layers']}-layer frontier geometry with grouped-query attention "
            f"({GEOM['kv_bytes_per_token_per_layer'] // 1024} KB of cache per token per layer)")
_FFN_EST = "bytes-bound" in DECK_LAYER_CARD_20261001["status"]
FFN_TXT = ("our feed-forward block added as a bytes-bound estimate" if _FFN_EST else
           "both per-layer steps measured including their feed-forward blocks — ours as written and uncompiled, "
           "so the ratio is a floor")


def h6_meaning(h5, h6):
    """Direction-aware plain-language reading of H6 (per-step decode ratio)."""
    if h6 >= 1.0:
        return (f"One decode step for all {h5:,.0f} resident conversations takes less time than the transformer's "
                f"step for its 1 (×{h6:.2f}): no growing cache to re-read for every token.")
    return (f"Per decode step the transformer's single long-context stream is still faster than our "
            f"{h5:,.0f}-stream step (×{h6:.2f}, i.e. our step is {1 / h6:.2f}× longer); the gain is H5 — "
            f"{h5:,.0f} conversations advance per step instead of 1.")

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


SIDEBAR_HEAD = None  # the sidebar's single headline metric; filled in main() once the company edits are read


def sidebar_globals():
    global SIDEBAR_HEAD
    s = st.sidebar
    s.title("Assumptions")
    SIDEBAR_HEAD = s.container()
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

    # No model-size slider: the equal-quality ratio is fixed at trillion-parameter
    # scale. 2026-10-01 user ruling: it applies to INFERENCE as well as training
    # (a smaller equal-quality model costs proportionally less per token), so the
    # inference lever = tokens per GPU at equal size (H5 x H6, editable, defaults
    # from the 2026-10-01 per-layer decode receipt at the workload's context, plus
    # prompt processing at the scenario's prefill speed) x the size factor.
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
    kern = scenarios[st.session_state["scenario"]]
    h3 = TRAIN_SPEED_BY_SCENARIO[kern]
    scen_box.caption(f"◆ Prompt-processing speed (×{pf[TODAY_LABEL]:.1f} current vs ×{pf[LANDED_LABEL]:.1f} "
                     f"optimized) and H3 training speed per token (×{TRAIN_SPEED_BY_SCENARIO['current']:.2f} "
                     f"MEASURED vs ×{TRAIN_SPEED_BY_SCENARIO['mature']:.2f} TARGET) differ by scenario.")
    # 2026-10-01: the servers around the chips and the datacenters FOLLOW the
    # fleet by default (they scale down when no longer necessary); holding the
    # datacenters (contracted leases to FY33) is the sensitivity.
    st.session_state.setdefault("hold_dc", False)
    scen_box.checkbox("Hold datacenters (contracted leases to FY33)", key="hold_dc",
                      help="Default (unchecked): the servers around the chips AND the datacenter buckets "
                           "(buildings, power & cooling, network) shrink with the chip fleet. Checked: the "
                           "datacenters are held at today's spend — leases are contracted to FY33 — and only "
                           "chips, servers and power fall. Front tabs only (Summary, Value Bridge, Levers).")
    g["follow"] = BRIDGE_FOLLOW_HELD if st.session_state["hold_dc"] else BRIDGE_FOLLOW_20261001

    # H5 / H6 defaults track the context; a changed context resets them.
    d5, d6 = decode_levers(ctx)
    if st.session_state.get("_h56_ctx") != ctx:
        st.session_state["conv_per_gpu"], st.session_state["decode_speedup"] = float(d5), float(d6)
        st.session_state["_h56_ctx"] = ctx
    lev_box.caption(f"**H1** smaller model ×{_gain:.2f} and **H2** fewer tokens ×{_gain:.2f} — H1 also multiplies "
                    f"inference: a {_gain:.2f}× smaller equal-quality model costs ~{_gain:.2f}× less per token "
                    f"(Inputs tab) · **H3** training speed per token ×{h3:.2f} "
                    f"({'MEASURED, modern 8k/64k/256k curriculum' if kern == 'current' else 'TARGET, fused kernels'}; "
                    f"Levers tab).")
    g["mem_factor"] = gnum("H4 · Memory per conversation, ÷", "mem_factor", 1, 20000, 10, "%.0f",
                           help="MEASURED ×2,032 at 262k on our test model. Frontier-size models with "
                                f"grouped-query attention: ~÷{fleet_memory_lever(262144):.0f} (estimate).",
                           box=lev_box)
    g["conv_per_gpu"] = lev_box.number_input(
        "H5 · More conversations per GPU, ×", min_value=1.0, max_value=1024.0, step=1.0, key="conv_per_gpu",
        format="%.0f", help=f"MEASURED per layer (2026-10-01 decode receipt), quoted at {GEOM_TXT}: our "
                            f"{H5_CONVERSATIONS:.0f} resident streams (1 MB of state per layer each) vs the "
                            f"transformer's largest measured batch whose full-model cache fits the card — 1 at 262k. "
                            f"Defaults track the conversation length.")
    g["decode_speedup"] = lev_box.number_input(
        "H6 · Decode step vs the transformer, per layer, ×", min_value=0.1, max_value=100.0, step=0.1, key="decode_speedup",
        format="%.2f", help=f"MEASURED per layer (2026-10-01 decode receipt): the transformer's per-step decode "
                            f"time at its feasible batch over ours at {H5_CONVERSATIONS:.0f} streams ({FFN_TXT}) — "
                            f"×{H6_DECODE:.2f} at 262k at "
                            f"{GEOM_TXT} (its step scaled to the grouped-query cache bytes it re-reads; "
                            f"×{deck_layer_levers(262144)['h6']:.2f} on the receipt's 24-layer comparator). "
                            + ("Below 1: the transformer's single stream steps faster; the lever is H5." if H6_DECODE < 1 else ""))
    # tokens per GPU at EQUAL size, at the sidebar workload, for both prefill
    # scenarios; the inference lever = that x the equal-quality size factor
    # (2026-10-01 ruling). g['flop_factor'] carries the ACTIVE scenario (the
    # technical tabs read it); the front tabs read the current/optimized pair so
    # the scenario is applied exactly once (see _bridge_levers).
    rq = prompt_time_ratio(ctx, wl["in_out_ratio"])
    eq = {k: max(1.0, inference_throughput(g["conv_per_gpu"], g["decode_speedup"], pf[k], rq)) for k in scenarios}
    g["flop_factor_current"] = eq[TODAY_LABEL] * INFERENCE_SIZE_FACTOR
    g["flop_factor_optimized"] = eq[LANDED_LABEL] * INFERENCE_SIZE_FACTOR
    g["flop_factor"] = eq[st.session_state["scenario"]] * INFERENCE_SIZE_FACTOR
    _lvc = deck_layer_levers(ctx, geometry=GEOM)
    lev_box.caption(f"→ tokens per GPU at equal size (H5 × H6, with prompts) **×{eq[st.session_state['scenario']]:,.0f}** "
                    f"× the **×{INFERENCE_SIZE_FACTOR:.2f}** equal-quality size factor (H1) = inference lever "
                    f"**×{g['flop_factor']:,.0f}** · inference GPUs fall by "
                    f"**×{min(g['mem_factor'], g['flop_factor']):,.0f}** (the smaller of memory and that)"
                    + (" · H5/H6 defaults at this context are modelled from the receipt's measured contexts "
                       "(4k, 32k, 128k, 262k)" if _lvc["modelled"] else "")
                    + (" · one transformer conversation no longer fits a GPU here (sharding cost ignored — "
                       "conservative for us)" if _lvc["sharded"] else ""))

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
        g["dc_scale"] = gnum("Datacenter scaling factor (legacy Totals engine only)", "dc_scale", 0.0, 1.0, 0.05,
                             "%.2f", box=st,
                             help="Technical Totals / company tabs only. 0 = only accelerator silicon shrinks; "
                                  "1 = the whole datacenter scales. The front tabs let servers and datacenters "
                                  "follow the fleet by default (checkbox above).")
        g["named_share_of_global"] = gnum("Named share of global AI capex", "named_share_of_global",
                                          0.3, 1.0, 0.05, "%.2f", box=st)
        g["spacex_mktcap"] = gnum("SpaceX market cap ($B)", "spacex_mktcap", 200, 4000, 10, "%.0f", box=st)
        st.metric("Technical ledger (legacy): cost-weighted reduction", x1(reduction_factor(g)),
                  help="Accelerators + power only, Amdahl-weighted — the technical Totals / company tabs. The "
                       "headline is the value bridge on the Summary tab.")
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
    h = st.columns([2.4, 1, 1]); h[0].markdown("**Metric**"); h[1].markdown("**FY2026**"); h[2].markdown("**FY2025** (last year)")

    def two(label, key, d25, d26, lo, hi, step, fmt):
        cc = st.columns([2.4, 1, 1]); cc[0].write(label)
        st.session_state.setdefault(p + key + "25", float(d25))
        st.session_state.setdefault(p + key + "26", float(d26))
        v26 = cc[1].number_input(label + "26", min_value=float(lo), max_value=float(hi),
                                 step=float(step), key=p + key + "26", format=fmt, label_visibility="collapsed")
        v25 = cc[2].number_input(label + "25", min_value=float(lo), max_value=float(hi),
                                 step=float(step), key=p + key + "25", format=fmt, label_visibility="collapsed")
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
    CB = ["Metric", "FY2026", "FY2025 (last year)", "Basis / source"]
    W = {"Metric": "large", "Basis / source": "large"}

    section(f"{name} — company data (green = disclosed, yellow = estimate)")
    show_table(CB, [
        [("Total capex ($B)", ""), (n1(t26), "y"), (n1(t25), "g"), ("FY2026 per company guidance / actual (see Sources); FY2025 disclosed", "")],
        [("Infra / data-center share ★", ""), (pct(i26), "y*"), (pct(i25), "y*"), (INFRA_SHARE_BASIS.get(name, "strips non-datacenter capex"), "")],
        [("Server / short-lived share ★", ""), (pct(s26), "y*"), (pct(s25), "y*"), ("CFO-disclosed", "")],
        [("Accelerator share within servers ★", ""), (pct(a26), "y*"), (pct(a25), "y*"), ("BOM teardown ~67–80%", "")],
        [("Market cap ($B)", ""), (n0(c["mcap"]), "g"), ("", ""), ("approx market data", "")],
    ], widths=W, wrap=True)

    section("Derivation — how much of the capex buys AI chips")
    show_table(CB, [
        [("AI-infra capex ($B)", ""), (n1(d26["ai_capex"]), "b"), (n1(d25["ai_capex"]), "b"), ("= total × infra", "")],
        [("Server bucket ($B)", ""), (n1(d26["ai_capex"] * s26), "b"), (n1(d25["ai_capex"] * s25), "b"), ("= infra × server", "")],
        [("ACCELERATOR capex ($B)", ""), (n1(d26["accel"]), "b"), (n1(d25["accel"]), "b"), ("= server × accel", "")],
        [("Accel % of total capex", ""), (pct(d26["accel_pct"]), "b"), (pct(d25["accel_pct"]), "b"), ("varies by company", "")],
    ], widths=W, wrap=True)

    section("Fleet & power (from AI-chip capex)")
    show_table(CB, [
        [("Fleet size (GPU-equiv)", ""), (n0(fb26["fleet"]), "b"), (n0(fb25["fleet"]), "b"), ("= accel capex / $ per GPU", "")],
        [("Total wall power (MW)", ""), (n0(fb26["mw"]), "b"), (n0(fb25["mw"]), "b"), ("= GPUs × kW", "")],
        [("Annual opex ($M)", ""), (n0(fb26["ann_m"]), "b"), (n0(fb25["ann_m"]), "b"), ("= MWh × rate × (1+overhead) × 365", "")],
        [("Lifetime opex ($B)", ""), (n1(fb26["life_b"]), "b"), (n1(fb25["life_b"]), "b"), ("× fleet life", "")],
    ], widths=W, wrap=True)

    section("Helarctos effect — technical engine (memory and compute priced separately)")
    show_table(CB, [
        [("Efficient AI capex ($B)", ""), (n1(d26["ai_capex"] - d26["capex_avoided"]), "b"), (n1(d25["ai_capex"] - d25["capex_avoided"]), "b"), ("= AI capex − avoided", "")],
        [("H4·H5 → Capex avoided/yr ($B)", "h"), (n1(d26["capex_avoided"]), "b"), (n1(d25["capex_avoided"]), "b"), ("Helarctos enters here: chips × (1 − 1/cost-weighted cut from H4 memory and H5 compute)", "")],
        [("H4·H5 → Annual opex savings ($M)", "h"), (n0(d26["opex_saved"] * 1000), "b"), (n0(d25["opex_saved"] * 1000), "b"), ("Helarctos enters here: power shrinks with the fleet", "")],
        [("Sustained annual benefit ($B/yr)", ""), (n1(d26["spend_cut"]), "b"), (n1(d25["spend_cut"]), "b"), ("= avoided + opex savings", "")],
        [("Capitalized value ($B)", ""), (n0(d26["capitalized"]), "b"), (n0(d25["capitalized"]), "b"), ("= benefit / discount rate", "")],
        [("% of market cap", ""), (f"{d26['capitalized'] / c['mcap']:.1%}" if c["mcap"] else "—", "b"), (f"{d25['capitalized'] / c['mcap']:.1%}" if c["mcap"] else "—", "b"), ("", "")],
    ], widths=W, wrap=True)

    section("AI economics — cash basis: AI revenue − AI capex − AI power")
    show_table(CB, [
        [("AI revenue ($B)", ""), (n1(c["ai_rev"][1]), "y"), (n1(c["ai_rev"][0]), "y"), ("ESTIMATE (see Methodology); FY2026 first", "")],
        [("AI capex ($B)", ""), (n1(d26["ai_capex"]), "b"), (n1(d25["ai_capex"]), "b"), ("full AI-infra (accel + buildings + power + net)", "")],
        [("AI opex ($B)", ""), (n1(d26["ai_opex"]), "b"), (n1(d25["ai_opex"]), "b"), ("annual power / operating", "")],
        [("Net AI NOW ($B)", ""), (n1(d26["net_now"]), "b"), (n1(d25["net_now"]), "b"), ("revenue − capex − opex (cash burn)", "")],
        [("Spend cut with Helarctos ($B)", ""), (n1(d26["spend_cut"]), "b"), (n1(d25["spend_cut"]), "b"), ("accel capex avoided + opex saved", "")],
        [("Net AI with Helarctos ($B)", ""), (n1(d26["net_arch"]), "b"), (n1(d25["net_arch"]), "b"), ("= net now + spend cut", "")],
        [("% AI spend reduction", ""), (pct(d26["pct_cut"]), "b"), (pct(d25["pct_cut"]), "b"), ("spend cut / total AI spend", "")],
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

    st.caption("**Technical ledger (legacy): accelerators + power only, Amdahl-weighted.** This engine prices memory "
               "and compute separately and counts only the chips (plus the legacy datacenter-scaling share, default 0). "
               "The headline is the value bridge on the Summary tab, which also lets the servers and data centres "
               "sized by the GPUs follow the fleet.")
    c1, c2, c4 = st.columns(3)
    c1.metric("Cost-weighted reduction (legacy ledger)", x1(reduction_factor(g)))
    # "Now" = FY26 (2026-09-01 user correction: the FY25 -$295B read as far too
    # small — the current-year burn is the FY26 estimate).
    c2.metric(f"Net AI now ({len(comps)} cos, FY2026)", f"{bn(tot26['net_now'])}/yr")
    c4.metric("% of AI spend cut (FY2026)", pct(tot26["pct_cut"]))
    n2c, n3c = st.columns(2)
    n2c.metric("Net AI w/ our arch (FY2026, technical ledger)", f"{bn(tot26['net_arch'])}/yr", delta=f"{bn(tot26['spend_cut'])} cut")
    n3c.metric("Net AI w/ our arch (FY2027, street estimate)", f"{bn(tot27['net_arch'])}/yr", delta=f"{bn(tot27['spend_cut'])} cut")
    st.caption("FY2027 = street-estimate capex (Morgan Stanley +57% path; Oracle's ~$70B is the only real FY27 guide) "
               "with ai_rev at the ×1.5 placeholder — prediction-grade, see per-company notes. FY2026 is the base year "
               "(the capex is set; guidance vs actual is noted per company). FY2025 (last year) is below for reference.")
    verdict = ("AI flips **profitable** at these settings." if tot26["net_arch"] > 0
               else f"FY26 AI burn shrinks from **{md_usd(-tot26['net_now'])}B** to **{md_usd(-tot26['net_arch'])}B**/yr "
                    f"(spend cut **{md_usd(tot26['spend_cut'])}B**, ~{md_usd(tot26['capitalized'])}B capitalized).")
    st.markdown(f"Technical ledger (accelerators + power only, Amdahl-weighted, legacy): {verdict}")
    st.caption(f"The {len(comps)} named firms are a floor. GLOBAL estimate (named ≈ {g['named_share_of_global']:.0%} of "
               f"world AI capex): FY2026 spend cut ~{md_usd(glob26['spend_cut'])}B → ~\\${glob26['capitalized'] / 1000:.1f}T "
               f"capitalized (FY2025: ~\\${glob25['capitalized'] / 1000:.1f}T). The rest is other clouds, China, neoclouds, xAI & sovereign AI.")

    section("Net AI economics — FY2026 (cash basis)")
    econ_show(rows26, tot26, glob26)
    section("Net AI economics — FY2025, last year (cash basis)")
    econ_show(rows25, tot25, glob25)

    section("Savings breakdown & capitalized value")
    disc = g["discount_rate"]
    o25 = sum(r["opex_saved"] for r in rows25); k25 = sum(r["capex_avoided"] for r in rows25)
    o26 = sum(r["opex_saved"] for r in rows26); k26 = sum(r["capex_avoided"] for r in rows26)
    show_table(["Item", "Saved OPEX", "Avoided CAPEX (Overspend)", "Total"], [
        [("FY2026 annual ($B/yr)", ""), (n1(o26), "b"), (n1(k26), "b"), (n1(o26 + k26), "b")],
        [("FY2026 capitalized ($B)", ""), (n0(o26 / disc), "b"), (n0(k26 / disc), "b"), (n0((o26 + k26) / disc), "s")],
        [("FY2025 annual, last year ($B/yr)", ""), (n1(o25), "b"), (n1(k25), "b"), (n1(o25 + k25), "b")],
        [("FY2025 capitalized ($B)", ""), (n0(o25 / disc), "b"), (n0(k25 / disc), "b"), (n0((o25 + k25) / disc), "s")],
        [("% reduction", ""), (pct(1 - 1 / energy_reduction(g)), "b"), (pct(1 - 1 / reduction_factor(g)), "b"), ("", "")],
    ])
    st.caption("OPEX = power saved each year (recoupable). CAPEX 'Overspend' = AI capex made unnecessary. "
               "This legacy engine counts accelerators plus the **Datacenter scaling factor** share of the rest "
               "(sidebar, technical inputs; 0 = accelerator-only, 1 = whole DC). The front tabs (Summary, "
               "Value Bridge) instead let the servers and datacenters sized by the GPUs follow the fleet by "
               "default, which is why they read higher.")


# ---- inputs tab ----------------------------------------------------------------
def inputs_tab(g):
    section("Global inputs (edit in the sidebar ◀) — teal = Helarctos lever")
    items = [
        ("H4 · Memory reduction factor", f"{g['mem_factor']:,.0f}×", "h", "O(1) state vs O(T) KV cache — MEASURED 2026-08-14 (full model, our test model): 64 concurrent 262k streams on one GPU in 1.62 GB against the transformer's 1 stream at 51.5 GB, a 2nd OOMs; 25.4 MB of state per stream vs 197 KB of KV per token of context = ×2,032 at 262k. Frontier-size models with grouped-query attention: ~÷208 (estimate)"),
        ("H1·H5·H6 · Inference compute lever (FLOPs reduction factor)", f"{g['flop_factor']:,.0f}×", "b", f"= tokens per GPU at equal size, (1 + rq) / (rq/PF + 1/(H5 × H6)) = ×{g['flop_factor'] / INFERENCE_SIZE_FACTOR:,.0f}, × the ×{INFERENCE_SIZE_FACTOR:.2f} equal-quality size factor (H1; a smaller equal-quality model costs proportionally less per token, in inference as in training — 2026-10-01). H5 = {g['conv_per_gpu']:.0f} conversations per GPU and H6 = ×{g['decode_speedup']:.2f} per-step decode, both MEASURED per layer on the 2026-10-01 decode receipt ({FFN_TXT}); PF = the scenario's prompt-processing speed vs the transformer (×{prefill_advantage('current'):.1f} current, ×{prefill_advantage('mature'):.1f} optimized); rq = the transformer's prompt time / decode time. Edit H5 and H6 in the sidebar"),
        ("Memory share of GPU cost", pct(g["mem_share"]), "y", "HBM + most CoWoS packaging → ~60/40 memory/compute (BOM)"),
        ("Opex / energy reduction", x1(energy_reduction(g)), "b" if g.get("opex_reduction_override") is None else "y", "DERIVED = cost-weighted reduction (energy splits memory/compute like cost); override in sidebar"),
        ("Discount rate ★", pct(g["discount_rate"]), "y", "perpetuity: value = annual benefit / rate"),
        ("Fully-loaded $/GPU", usd0(g["gpu_cost"]), "y", "GPU + share of server, NVLink, networking"),
        ("Wall power / GPU (kW)", f"{g['wall_power_kw']:.1f}", "y", "GB200 NVL72: ~1.7–1.8 kW IT per GPU × PUE ~1.3"),
        ("Electricity rate ($/kWh)", f"{g['elec_rate']:.2f}", "y", "datacenter wholesale"),
        ("Cooling / ops overhead", pct(g["cooling_overhead"]), "y", "non-power running cost as fraction of electricity"),
        ("Fleet useful life (yr)", f"{g['fleet_life_yr']:.0f}", "y", "AI-GPU depreciation life"),
        ("Datacenter scaling factor (legacy Totals engine only)", pct(g["dc_scale"]), "y", "technical Totals / company tabs: 0 = accel-only; 1 = whole DC scales. The front tabs let servers and datacenters follow the fleet by default (sidebar checkbox)"),
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
    _bind = "compute (the inference lever" if g["flop_factor"] < g["mem_factor"] else "memory (H4"
    st.caption(f"Floored by the least-reduced component — at these levers, {_bind} "
               f"×{min(g['flop_factor'], g['mem_factor']):,.0f}; memory ×{g['mem_factor']:,.0f} vs inference "
               f"lever ×{g['flop_factor']:,.0f}).")


# ---- sensitivity tab -----------------------------------------------------------
def sensitivity_tab(comps, g):
    section("Sensitivity (SpaceX, FY2026) — Current vs Optimized kernels vs live cost-weighted")
    sx = next(c for c in comps if c["name"] == "SpaceX")
    d = compute_company(sx, g, "fy26")
    accel, opx, disc, mcap = d["accel"], d["opex_saved"] * 1000, g["discount_rate"], g["spacex_mktcap"]
    # Tiers mirror the sidebar picker (2026-09-01: pre-campaign floor and the
    # prefill-only ceiling are deleted — current numbers and upper bound only).
    # the model's two inference levers (equal-size tokens per GPU x the size factor, 2026-10-01)
    r_today = reduction_factor(dict(g, flop_factor=INFERENCE_LEVER_CURRENT))
    r_ceil = reduction_factor(dict(g, flop_factor=INFERENCE_LEVER_OPTIMIZED))
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

    section("Conversation-length sensitivity — the inference lever from the 2026-10-01 per-layer decode receipt")
    rc_rows = []
    for _c in (4096, 32768, 65536, 131072, 262144, 1048576):
        _l = deck_layer_levers(_c, geometry=GEOM)
        _l24 = deck_layer_levers(_c)
        _eq = inference_throughput(_l["h5"], _l["h6"], prefill_advantage("current", _c), prompt_time_ratio(_c))
        _tone = "g" if not _l["modelled"] else "y"
        rc_rows.append([
            (ctx_k(_c) + (" (default)" if _c == CAMPAIGN_LANDED_20260831["context_tokens"] else ""),
             "b" if _c == CAMPAIGN_LANDED_20260831["context_tokens"] else ""),
            (f"{_l['tf_streams']}" + (" (does not fit: sharded)" if _l["sharded"] else ""), _tone),
            (f"×{_l['h5']:,.0f}", _tone), (f"×{_l['h6']:.2f}", _tone), (f"×{_l['ratio']:,.0f}", _tone),
            (f"×{_eq:,.0f}", "b"), (f"×{_eq * INFERENCE_SIZE_FACTOR:,.0f}", "b"),
            (f"×{_l24['ratio']:,.0f}", ""),
            ("modelled (linear fit in context over the measured cells)" if _l["modelled"] else "measured cell", ""),
        ])
    show_table(["Conversation length", "Transformer streams that fit a GPU", "H5 conversations per GPU",
                "H6 per-step decode", "H5 × H6", "Tokens per GPU at equal size (current kernels)",
                "Inference lever (× size factor)", "H5 × H6 on the 24-layer comparator", "Basis"], rc_rows,
               widths={"Basis": "medium"})
    st.caption(f"Quoted at the geometry of record: {GEOM_TXT}, anchored on disclosed frontier depths "
               f"({GEOM['anchor']}). The transformer runs the largest measured batch whose full-model cache fits a "
               f"{GEOM['hbm_gb']:.0f} GB card at that context; at and above 32k its step is scaled to the cache bytes "
               f"it re-reads (grouped-query vs the receipt's multi-head arm). We run {H5_CONVERSATIONS:.0f} streams at "
               f"{DECK_LAYER_CARD_20261001['own_state_mb_per_stream']:.1f} MB per layer each, context-independent. "
               f"The per-layer step ratio is depth-invariant, so the ratio is the full-model ratio; {FFN_TXT}. "
               f"Contexts the receipt did not run are fitted linearly in "
               f"context and marked. The last column is the receipt's own 24-layer d2048 comparator for reference.")

    section("Conversation-length sensitivity — LEGACY ESTIMATE BASIS (superseded)")
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
                "◆ Tokens per GPU at equal size, current kernels (legacy estimate)",
                "◆ Tokens per GPU at equal size, optimized kernels (legacy estimate)", "Gap"], ctx_rows)
    st.caption("LEGACY ESTIMATE BASIS: this table still runs on the retired aggregate-decode estimate "
               "(a 2.5 ms-per-token step at 64 streams), kept only to show how prompt processing's share of "
               "serving cost moves with context; the table above it is the receipt-based lever the model uses. "
               "Cost = box wall-clock, not FLOPs. The transformer's decode leg is KV-bandwidth-bound (measured "
               "1/context law) while its prefill runs near peak, so prefill is <1% of its serving cost at every "
               "context — which is why the banked ×3.936 prefill (Current kernels) and the full ×7.03 (Optimized "
               "kernels) land within a few percent of each other.")


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
               "`internal measurement archive` or to an assumption row in the last table. "
               "**Scope**: systems numbers are component-scope (bAttention d1536 fwd+bwd internal-dynamics "
               "sub-block vs transformer d2048 forward-only layer); every speedup is each family vs "
               "its **own** 1-GPU baseline, so the scope cancels inside each ratio.")

    m1, m2, m3, m4 = st.columns(4)
    r262 = serving_economics(262144)
    m1.metric("Serving cost ratio @262k ctx", f"{r262['cost_ratio']:.1f}×",
              help="MEASURED per-GPU aggregate decode throughput at each family's own concurrency ceiling "
                   "(2026-08-14 full-model receipt). Granting the transformer an idealized KV÷8 stack: "
                   f"{serving_economics(262144, s={'tf_kv_compression': 8.0})['cost_ratio']:.1f}×.")
    m2.metric("Concurrent 262k streams / GPU", f"{MEASURED['serve_streams_per_gpu']} vs 1",
              help="MEASURED 2026-08-14 (full model): 64 bAttention streams in 1.62 GB of state on one GPU; the "
                   "transformer fits ONE 262k stream at 51.5 GB and a second OOMs the card. Aggregate throughput "
                   f"×8.8. The 2026-10-01 per-layer decode receipt supersedes this cell as the source of H5 "
                   f"({H5_CONVERSATIONS:.0f} streams vs 1) and H6 (×{H6_DECODE:.2f} per step).")
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
               f"the internal dynamics token-by-token costs ×{MEASURED['stepped_vs_scanned']:.0f} vs the "
               "production training step (same geometry, same box).")

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
        "constant ratio decay: the transformer pays `base + attn·T` per token (the transformer's fused attention kernel is "
        "linear per token, quadratic over the sequence) while our internal state is fixed-size, so our "
        "per-token cost is FLAT — now MEASURED, not inferred: our backward pass per-token is flat in T "
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

    section("What happens to the technical-ledger FY2026 headline when training enters the blend")
    _hl_rows = []
    _base26 = compute_year(g, COMPANIES, "fy26")[1]["spend_cut"]
    for _key, _spec in TRAINING_CURRICULA.items():
        _row = [(_spec["label"], "")]
        for _ts in (0.0, 0.2, 0.4):
            _h = headline_with_training(_key, _ts)
            _cut26 = compute_year(dict(g, flop_factor=_h["flop_factor"]), COMPANIES, "fy26")[1]["spend_cut"]
            _row.append((f"${_cut26:.0f}B", "b" if _cut26 - _base26 >= -1e-9 else "y"))
        _h2 = headline_with_training(_key, 0.2)
        _row.append(("raises it" if _h2["raises_headline"] else "dilutes it",
                     "b" if _h2["raises_headline"] else "y"))
        _hl_rows.append(_row)
    show_table(["Curriculum", "training 0% of spend", "training 20%", "training 40%", "Direction"], _hl_rows,
               widths={"Curriculum": "medium"})
    st.caption(
        "Every cell is the FY2026 technical-ledger spend cut across the six named firms, recomputed end-to-end "
        "(the blended lever from `headline_with_training` applied to FY2026). The old assumption — all training "
        "at 2k — was the only one that dragged the headline materially. At a modern mix the "
        "headline is essentially flat, and at long-context-heavy mixes it rises. Note how narrow the "
        "whole range is: Amdahl saturation means the dollar headline is simply not very sensitive to "
        "this lever, which is the honest shape of the result in both directions.")
    st.caption(
        "**Excluded, and it runs our way.** At 262k+ the transformer also pays sequence-parallelism "
        f"and activation-memory costs a fixed-size internal state avoids — its KV grows at "
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
               "has not been re-run since the 2026-08-24..29 fused kernel work banked ×3.94 on our own arm, "
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
         ("cross-family, bf16 (the transformer's fused attention kernel available to the transformer), params EXACTLY matched, 1 layer / batch 16", "g")],
        [("Compute — same lane, other widths", ""), ("×3.83 d1024/1M · ×2.01 d2048/262k", "b"),
         ("experiments/paper_figures/output/matched_d{1024,2048}_h100.json", ""),
         ("1× H100 80GB HBM3, 2026-07-21", ""),
         ("ratio rises with context, falls with width", "g")],
        [("Compute — fp32 lane", ""), ("×7.91 at d2048/262k", "y"),
         ("experiments/paper_figures/output/deck_speed_scaling_d2048_h100_20260803.json", ""),
         ("1× H100 80GB HBM3, 2026-08-03", ""),
         ("full model (24 layers); no fp32 the transformer's fused attention kernel kernel, owned arm fp16 internally, params 16.7% apart", "y")],
        [("Memory (mem_factor)", ""), ("÷2,032 (measured, test model)", "s"),
         ("internal measurement archive — decode_multigpu, slide_decode_262k", ""),
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
        [("Inference compute lever (FLOPs reduction, ×)", ""), (f"{g['flop_factor']:,.0f}×", "y"), ("measured × projection", "y"), (f"◆ Set by the scenario (Current kernels by default). RE-BASED 2026-10-01 = tokens per GPU at EQUAL model size (×{g['flop_factor'] / INFERENCE_SIZE_FACTOR:,.0f}: H5 = {g['conv_per_gpu']:.0f} conversations per GPU × H6 = ×{g['decode_speedup']:.2f} per-step decode, both MEASURED per layer on the 2026-10-01 decode receipt — the transformer at the largest measured batch whose full-model cache fits the card at that context, us at {H5_CONVERSATIONS:.0f} streams ({FFN_TXT}); prompt processing added back at the BANKED ×{KERNEL_SPEEDUP_REALIZED_20260824:.2f} prefill speed-up) × the FIT-DERIVED equal-quality parameter ratio ×{INFERENCE_SIZE_FACTOR:.2f} at 1T (reinstated 2026-10-01: a smaller equal-quality model costs proportionally less per token, in inference as in training). Parameter ratio: the sealed refit's fits (4 rungs per family) cross at 392M and give the transformer's quality on 23.7% of the params at 1T — a projection of two fits, not a measurement. Optimized kernels = prefill at the full ×{CEILING_PREFILL_SPEEDUP:.2f} (×{KERNEL_SPEEDUP_REALIZED_20260824:.2f} MEASURED × ×{KERNEL_SPEEDUP_REMAINING_TARGET:.2f} TARGET) → ×{INFERENCE_LEVER_OPTIMIZED:,.0f}. The former 2.5 ms-per-token × 64-stream aggregate-decode ESTIMATE is superseded", "")],
        [("Compute cost that runs AGAINST us", ""), (f"×{KERNEL_CAMPAIGN_20260824['step_ratio_2k']:.2f}", "y"), ("measured", "b"), (f"RE-BASED 2026-08-24 (one GH200, ONE layer, fwd+bwd, bf16, checkpointing off both arms, against a MODERN transformer block — 24Q/4KV, head_dim 256, RoPE 64, gated attention): a bAttention training step costs ×{KERNEL_CAMPAIGN_20260824['step_ratio_2k']:.2f} MORE at T=2,048 ({KERNEL_CAMPAIGN_20260824['battn_ms_2k']:.1f} vs {KERNEL_CAMPAIGN_20260824['tf_ms_2k']:.1f} ms/step) and ×{KERNEL_CAMPAIGN_20260824['step_ratio_8k']:.2f} at T=8,192 ({KERNEL_CAMPAIGN_20260824['battn_ms_8k']:.1f} vs {KERNEL_CAMPAIGN_20260824['tf_ms_8k']:.1f}); fwd ×{KERNEL_CAMPAIGN_20260824['fwd_ratio']:.2f}, bwd ×{KERNEL_CAMPAIGN_20260824['bwd_ratio']:.2f}. It read ×{KERNEL_CAMPAIGN_20260824['step_ratio_2k_precampaign']:.2f} ({KERNEL_CAMPAIGN_20260824['battn_ms_2k_precampaign']:.1f} ms) the same morning — ×{KERNEL_SPEEDUP_REALIZED_20260824:.2f} banked in a day — and ×{STEP_GAP_20260814['gap_against_battn']:.2f} on the older d1536/27L fp16 601-step receipt (internal measurement archive), which is now superseded as a headline. At equal quality the T=2,048 figure falls to ×{KERNEL_CAMPAIGN_20260824['step_ratio_2k'] * param_matching_fraction(DECK_DEPLOYMENT_SCALE):.2f}. Training is still excluded from the serving claim", "")],
        [("Training PEAK MEMORY that runs AGAINST us", ""), (f"×{KERNEL_CAMPAIGN_20260824['mem_ratio_2k']:.2f}", "y"), ("measured", "b"), (f"Same 2026-08-24 frame: ×{KERNEL_CAMPAIGN_20260824['mem_ratio_2k']:.3f} at T=2,048 ({KERNEL_CAMPAIGN_20260824['battn_peak_mib_2k']:,.1f} vs {KERNEL_CAMPAIGN_20260824['tf_peak_mib_2k']:,.1f} MiB) and ×{KERNEL_CAMPAIGN_20260824['mem_ratio_8k']:.3f} at T=8,192, down from ×{KERNEL_CAMPAIGN_20260824['mem_ratio_2k_precampaign']:.3f} the same morning. This is TRAINING peak memory — NOT the ÷100 memory lever above, which is SERVING state and is unaffected. It is not an input to the cost model; it caps per-GPU batch density", "")],
        [("Kernel program (TARGET, not a result)", ""), (f"≤{KERNEL_CAMPAIGN_20260824['target_8k_win_gate_ms']:.1f} ms", "y"), ("target", "y"), (f"The funded 8k-win gate is a T=8,192 step at or below {KERNEL_CAMPAIGN_20260824['target_8k_win_gate_ms']:.2f} ms — beating the transformer — which needs ×{KERNEL_CAMPAIGN_20260824['target_8k_gap_remaining']:.2f} more, i.e. 44% of the step still to remove. Named levers: {KERNEL_CAMPAIGN_20260824['target_levers']}. Memory target ×{KERNEL_CAMPAIGN_20260824['target_mem_ratio']:.2f} (at or below the transformer's). NO RECEIPT behind any of this — it is the program plan, and it is what the Ceiling scenario's remaining factor is", "")],
        [("Memory share of GPU cost", ""), (pct(g["mem_share"]), "y"), ("assumption", "y"), ("BOM teardown: HBM ~41% + CoWoS ~23% (mostly memory) vs logic die ~9% → ~60/40 (Evidence tab)", "")],
        [("Cost-weighted reduction (×)", ""), (x1(reduction_factor(g)), "b"), ("derived", "b"), ("= 1 / (mem_share/mem_factor + (1−mem_share)/flop_factor). Amdahl blend.", "")],
        [("Energy / opex reduction (×)", ""), (x1(energy_reduction(g)), ek[1]), (ek[0], ek[1]), ("= cost-weighted reduction by default (energy splits memory/compute like cost); override in sidebar", "")],
        [("Discount rate", ""), (pct(g["discount_rate"]), "y"), ("assumption", "y"), ("Perpetuity capitalization rate; set to your WACC (6% → ×16.7)", "")],
        [("Fully-loaded $/GPU", ""), (usd0(g["gpu_cost"]), "y"), ("assumption", "y"), ("B200-class GPU (~$40k) + share of server, NVLink, networking", "")],
        [("Wall power / GPU", ""), (f"{g['wall_power_kw']:.1f} kW", "y"), ("assumption", "y"), ("≈1 kW TDP × PUE ~1.3 + node overhead", "")],
        [("Electricity rate", ""), (f"${g['elec_rate']:.2f}/kWh", "y"), ("assumption", "y"), ("Datacenter wholesale ~$0.06–0.10/kWh", "")],
        [("Cooling / ops overhead", ""), (pct(g["cooling_overhead"]), "y"), ("assumption", "y"), ("Non-power running cost as a fraction of electricity", "")],
        [("Fleet life", ""), (f"{g['fleet_life_yr']:.0f} yr", "y"), ("assumption", "y"), ("AI-GPU depreciation life; filings say 5–6 yr (we use 4, conservative)", "")],
        [("Datacenter scaling factor (legacy Totals engine only)", ""), (pct(g["dc_scale"]), "y"), ("toggle", "y"), ("Technical tabs only: share of the non-accelerator datacenter that also shrinks (0 = accelerator-only). The front tabs let the servers around the chips and the datacenters follow the fleet by default (2026-10-01); holding the datacenters (leases contracted to FY33) is their sensitivity", "")],
        [("Named share of global AI capex", ""), (pct(g["named_share_of_global"]), "y"), ("assumption", "y"), ("Named firms' share of worldwide AI capex; remainder grossed up pro-rata", "")],
        [("SpaceX market cap", ""), (n0(g["spacex_mktcap"]), "g"), ("data", "g"), ("Market ~$1.84T Aug 2026 (IPO 2026-06-12 at ~$1.77T)", "")],
        [("Per-company total capex (FY25)", ""), ("disclosed", "g"), ("data", "g"), ("10-K / earnings calls — see each company tab + its Sources", "")],
        [("Per-company FY26 capex", ""), ("estimate", "y"), ("assumption", "y"), ("Management guidance midpoint — see each company tab", "")],
        [("Per-company infra / server / accel", ""), ("estimate", "y"), ("assumption", "y"), ("CFO commentary (infra/server) + BOM teardown (accel ~67–80%)", "")],
        [("Per-company AI revenue", ""), ("mixed", "y"), ("assumption", "y"), ("Disclosed run-rates where available (MSFT $37B, AMZN $15B); else estimate", "")],
    ]
    show_table(["Value / driver", "Current", "Kind", "How it's derived / source"], rows,
               widths={"Value / driver": "medium", "How it's derived / source": "large"})

    _hf = headline_family()
    st.markdown(rf"""
### Methodology & sources

**Engine.** A GPU is ~60% memory / ~40% compute by cost. The cost-weighted reduction is Amdahl —
floored by the least-reduced component. Both scenarios price serving at a 262k-token average
conversation. RE-BASED 2026-10-01: the inference lever = tokens per GPU at EQUAL model size
(H5 {H5_CONVERSATIONS:.0f} conversations per GPU × H6 ×{H6_DECODE:.2f} per-step decode, both MEASURED per layer
on the 2026-10-01 decode receipt, prompt processing added back; ×{INFERENCE_LEVER_EQUAL_SIZE:,.0f}) × the
×{INFERENCE_SIZE_FACTOR:.2f} equal-quality size factor — a {INFERENCE_SIZE_FACTOR:.2f}× smaller equal-quality
model costs ~{INFERENCE_SIZE_FACTOR:.2f}× less per token, applied to inference as to training (this
reverses the 2026-09-29 reading that inference cost did not scale with model size). **Current
kernels** carries prefill at the banked ×3.94 (inference lever ~×{INFERENCE_LEVER_CURRENT:,.0f}), **Optimized
kernels** adds the remaining ×1.79 target (~×{INFERENCE_LEVER_OPTIMIZED:,.0f}) and the fused-kernel H3 training
speed (×{TRAIN_SPEED_BY_SCENARIO['mature']:.2f} TARGET vs ×{TRAIN_SPEED_BY_SCENARIO['current']:.2f} MEASURED).
Only those ◆ inputs differ between the scenarios. At these levers inference is limited by the compute
lever (×{INFERENCE_LEVER_CURRENT:,.0f}), not by memory (H4, ×2,032); either way over 99.9% of the
inference chip bill is gone, so the scenarios land within rounding in dollars. Multiplying the levers
is *not* physical: cost is additive, not multiplicative. The ×7.03 maturity factor was a flat ×5.5
assumption until 2026-08-24; it is now **×3.94 measured** (already banked; 2k clean-wall position of
record 2026-08-29, 101.737 vs 59.268 ms) × **×1.79 target** (the funded 8k-win gate).

**Decode receipt (2026-10-01).** One layer, bf16, graph-captured one-token step on a {GEOM['hbm_gb']:.1f} GB card: the
transformer at the largest measured batch whose full-model cache fits at that context, us at 256 streams
(1 MB of state per layer each, context-independent, a grid cap rather than a ceiling); {FFN_TXT}. The per-layer step ratio is
depth-invariant, so the ratio is the full-model ratio — but the transformer's cache capacity and
cache-read-bound step time are not, so the levers are quoted at {GEOM_TXT} (anchored on
{GEOM['anchor']}) rather than the receipt's 24-layer d2048 comparator. H5 × H6 at the geometry of
record: ×{deck_layer_levers(4096, geometry=GEOM)['ratio']:.1f} at 4k, ×{deck_layer_levers(32768, geometry=GEOM)['ratio']:.0f} at 32k, ×{deck_layer_levers(131072, geometry=GEOM)['ratio']:.0f} at 128k, ×{deck_layer_levers(262144, geometry=GEOM)['ratio']:.0f} at 262k
(the comparator reads ×{deck_layer_levers(262144)['ratio']:.0f} at 262k). At 262k the transformer fits ONE stream (90 GiB
of cache) and its single-stream step is {'faster' if H6_DECODE < 1 else 'slower'} than our 256-stream step (H6
×{H6_DECODE:.2f}); the lever is the 256 conversations advancing per step (H5). This replaces the 2.5 ms-per-token
× 64-stream aggregate-decode ESTIMATE that the model carried from 2026-08-31 to 2026-09-30.

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
the morning of 2026-08-24 (×6.12), so the fused kernel campaign banked **×3.94** (×2.86 of it in a day);
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
dominant fused kernel), and copies elimination (24.8% of the step). Memory target ×0.88, i.e. at or
below the transformer's. None of that has a receipt yet, and it is exactly the factor sitting inside the Optimized-kernels scenario.

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
−17.5% peak memory). The fp32 full-model sweep (×7.91 at 262k; no fp32 fused-attention kernel exists)
is kept on the Evidence tab as a scope reference only.

**Decode re-based 2026-08-14 (Serving·Training tab), and it fell.** A full-model bf16 measurement
(d2048/24 layers, both families, one GH200, same session) retired two numbers this model used to
carry: a transformer decode cost of 87.9 per token that a growing-KV re-planning artifact in the old
harness had inflated, and a per-stream serving state of 0.15 MB that was a carry-only proxy
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
unoptimised unoptimised per-step decode path, so it is a lower bound. Granting the transformer an
idealized KV ÷8 stack divides our decode lever by 8 and puts it ahead at 64k — that row is reported,
not hidden. The retired lines are **×53 decode**, **×15.9 decode**, **512 streams at 0.15 MB** and the
**~50–500×** band. Training (2026-08-07/08) is unchanged: the same 8 GPUs do **×1.39** the steps/s at
matched load (×5.15 fwd+bwd vs ×3.70 forward-only), **×1.70** at 256 sequences in flight.
Quality-parity at 70B (×4.1 params or ×1.7 tokens) is a **projection** from the measured ladder fits,
labeled as such. Systems numbers are component-scope; every speedup is vs the family's own 1-GPU
baseline.

**Per company.** `total capex (disclosed) × infra share × server share × accelerator share`
→ accelerator capex → fleet → energy/opex → efficient version → value (FY2026, the base year; FY2025 last year).
Infra (data-center) share comes from the 10-K/10-Q property & equipment and segment notes (Amazon = AWS, 68–76%); server share is CFO-disclosed; accelerator-within-server
is ~67–80% from BOM teardowns.

**Totals & global.** The named firms roll up live (no double-count). The **GLOBAL** row grosses the
named total up to a worldwide estimate using *Named share of global AI capex* (the rest = other clouds,
China, neoclouds, xAI, sovereign & enterprise).

**Net AI economics** are cash basis: `AI revenue − AI capex − AI opex`; with the architecture, add the
spend cut. All six firms lose money on AI today. "AI spend" is property & equipment additions (chips,
the servers around them, data-centre buildings, power & cooling, network) plus electricity — no labor.
On the technical tabs the *Datacenter scaling factor* (legacy) sets how much of the non-accelerator
datacenter shrinks too; on the front tabs the servers and datacenters sized by the GPUs FOLLOW the
blended fleet cut by default (2026-10-01 ruling: they scale down when no longer necessary), with the
datacenters HELD (leases contracted to FY33) as the sidebar sensitivity.

**Key results (FY2026 base year).** The headline is the value bridge (training on GPU-hours, inference on
whole GPUs, the servers and datacenters sized by those GPUs following the fleet):
**~\${_hf['fy26_bridge_spend_cut']:,.0f}B for FY2026** (net AI turns ~+\${_hf['fy26_bridge_net_with']:,.0f}B;
~\${_hf['fy26_bridge_spend_cut_held']:,.0f}B with the datacenters held to their FY33 leases,
~\${_hf['fy26_bridge_spend_cut_mature']:,.0f}B with the fused-kernel targets; last year, FY2025:
~\${_hf['fy25_bridge_spend_cut']:,.0f}B). The technical ledger (accelerators + power only, memory and compute
priced separately; cost-weighted reduction ~×{_hf['today_reduction']:,.0f}) reads
**~\${_hf['fy26_spend_cut']:,.0f}B for FY2026** (~\${_hf['fy26_capitalized'] / 1000:.1f}T capitalized at the 6% rate;
global est ~\${_hf['global_fy26_capitalized'] / 1000:.1f}T) and ~\${_hf['fy25_spend_cut']:,.0f}B for FY2025
(~\${_hf['fy25_capitalized'] / 1000:.1f}T; global ~\${_hf['global_fy25_capitalized'] / 1000:.1f}T; FY25 burn ~−\$284B/yr
on ~\$357B of AI capex against ~\$79B of AI revenue).
Data-center shares are from the 10-K/10-Q property & equipment notes (2026-09-29). Current and
Optimized kernels are within rounding of each other in dollars: the cut is `fleet × (1 − 1/lever)`
and it saturates; the underlying capex, shares and revenue never move.

**Sensitivity.** `capex_avoided ∝ (1 − 1/R)` is 0.999 at R ≈ 1,000, so the dollar headline is essentially
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
           "Training AND inference: a smaller equal-quality model costs proportionally less per token "
           "(the inference size factor, reinstated 2026-10-01)"),
    "H2": ("PROJECTED — standard compute-optimal scaling",
           f"Frontier labs train on data in proportion to model size, so a {param_matching_gain(DECK_DEPLOYMENT_SCALE):.1f}× "
           f"smaller model reaches its best quality on ~{param_matching_gain(DECK_DEPLOYMENT_SCALE):.1f}× fewer tokens.",
           "Training"),
    "H3": (f"MEASURED ×{TRAIN_SPEED_BY_SCENARIO['current']:.2f} — step costs measured at 2k/8k and modelled "
           f"beyond, weighted over a modern 8k/64k/256k curriculum (the curriculum shares are illustrative); "
           f"×{TRAIN_SPEED_BY_SCENARIO['mature']:.2f} is the fused-kernel TARGET",
           "Per token at the same model size, Helarctos trains faster over a modern long-context mix: our "
           "cost per token is flat in context while the transformer's grows, so the long-context phases "
           "dominate its bill.",
           "Training"),
    "H4": ("MEASURED ×2,032 at 262k (on our test model). Frontier-size models with grouped-query attention: "
           f"~÷{fleet_memory_lever(262144):.0f} (estimate), where memory would bind again (about −$0.5B)",
           "A transformer's memory (the KV cache) grows with every token of every live conversation — 51.6 GB "
           "for one 262k-token conversation. Helarctos keeps a fixed-size state (25.4 MB).",
           "Inference GPUs (the memory limit; not binding at 262k)"),
    "H5": (f"MEASURED per layer — 2026-10-01 decode receipt at {GEOM_TXT}: {H5_CONVERSATIONS:.0f} resident streams "
           f"at 1 MB of state per layer each (a grid cap, not a ceiling) vs the transformer's largest measured batch "
           f"whose full-model cache fits the card (1 at 262k — 90 GiB of cache)",
           f"The small memory (H4) lets one GPU hold {H5_CONVERSATIONS:.0f} resident 262k-token conversations at "
           f"once, where a transformer fits 1.",
           "Inference GPUs (tokens per GPU)"),
    "H6": (f"MEASURED per layer — 2026-10-01 decode receipt: ×{H6_DECODE:.2f} per decode step at 262k at "
           f"{GEOM_TXT} (×{deck_layer_levers(262144)['h6']:.2f} on the receipt's 24-layer comparator); {FFN_TXT}",
           h6_meaning(H5_CONVERSATIONS, H6_DECODE),
           "Inference GPUs (tokens per GPU)"),
}


def lever_fmt(code, v):
    return f"÷{v:,.0f}" if code == "H4" else (f"×{v:,.0f}" if code == "H5" else f"×{v:.2f}")  # H6: measured per layer at the geometry of record


def lever_value_code(code):
    """Style code for a lever's value cell: purple if it differs by scenario, ★ border if high impact."""
    return ("p" if LEV[code]["by_scenario"] else "b") + ("*" if LEV[code]["high_impact"] else "")


# One color per bar (categorical slots, validated for adjacent-pair CVD separation;
# two sit below 3:1 on white, so every bar carries its value label).
SOURCE_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#8e6bd6"]  # training, inference (each incl. its power), servers, datacenters
LADDER_COLORS = ["#a3a29c", "#2a78d6", "#eb6834", "#1baf7a", "#8e6bd6"]  # baseline gray, then one hue per step
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


def _stacked_bar(title, labels, values, colors):
    """One horizontal bar = the total; segments = the parts (they are one number, not alternatives)."""
    import altair as alt
    df = pd.DataFrame({"segment": labels, "value": values, "bar": ["FY2026"] * len(labels)})
    df["label"] = [f"{lab}: ${v:,.0f}B" for lab, v in zip(labels, values)]
    color = alt.Color("segment:N", sort=labels, title=None, legend=alt.Legend(orient="bottom"),
                      scale=alt.Scale(domain=labels, range=colors[:len(labels)]))
    bars = alt.Chart(df).mark_bar(size=34).encode(
        alt.X("value:Q", stack="zero", title="$B per year", axis=alt.Axis(grid=True, gridOpacity=0.3)),
        alt.Y("bar:N", title=None, axis=None), color,
        order=alt.Order("segment:N", sort="ascending"),
        tooltip=[alt.Tooltip("segment:N", title=""), alt.Tooltip("value:Q", title="$B/yr", format=",.0f")])
    text = alt.Chart(df).mark_text(color="white", fontWeight="bold").encode(
        alt.X("value:Q", stack="center"), alt.Y("bar:N", axis=None), text="label:N",
        order=alt.Order("segment:N", sort="ascending"))
    st.altair_chart((bars + text).properties(height=110, title=title), width="stretch")


def _train_shares():
    out = {}
    for c in COMPANIES:
        k = f"ts_{c['name']}"
        st.session_state.setdefault(
            k, float(CAMPAIGN_LANDED_20260831["train_share_by_company"].get(
                c["name"], CAMPAIGN_LANDED_20260831["train_share"])))
        out[c["name"]] = float(st.session_state[k])
    return out


def _follow(g):
    return g.get("follow", BRIDGE_FOLLOW_20261001)


def _held(g):
    return _follow(g)["datacenter"] < 1.0


def _bridge_levers(g):
    """The six audience levers at the sidebar workload for the ACTIVE scenario.
    The sidebar already prices tokens per GPU for both prefill scenarios (so the
    scenario's prefill speed is applied exactly once), so the serving lever is
    read from that pair rather than re-scaled by the model's default-context ratio."""
    kernels = "mature" if st.session_state.get("scenario") == "Optimized kernels" else "current"
    lv = value_bridge_levers(dict(g, flop_factor=g.get("flop_factor_current", g["flop_factor"])), kernels=kernels)
    tp = float(g.get("flop_factor_optimized" if kernels == "mature" else "flop_factor_current", g["flop_factor"]))
    lv["serving_throughput"] = lv["serving_compute_lever"] = tp
    lv["serving_gpu_lever"] = min(lv["memory"], tp)
    return lv


def summary_tab(comps, g):
    lv, ts, fol = _bridge_levers(g), _train_shares(), _follow(g)
    held = _held(g)
    rows26, t26 = value_bridge(g, comps, "fy26", lv, ts, follow=fol)
    _, t25 = value_bridge(g, comps, "fy25", lv, ts, follow=fol)
    spend26 = t26["ai_capex"] + t26["ai_opex"]
    st.markdown(
        f"In FY2026 Microsoft, Alphabet, Amazon, Meta, Oracle and SpaceX will spend **{md_usd(spend26)}B** on AI, "
        f"with AI revenue of {md_usd(t26['ai_rev'])}B. Helarctos models do the same AI work — training and "
        f"inference at the same quality — on far fewer GPUs, and the servers and data centres sized by those GPUs "
        f"shrink with them. This page shows how much of that spend becomes unnecessary, and which Helarctos "
        f"levers drive it. *AI spend* here is property & equipment additions (chips, the servers around them, "
        f"data-centre buildings, power & cooling, network) plus electricity — no labor.")
    st.caption(f"Scenario: **{st.session_state.get('scenario')}** (◆ switch it in the sidebar) · "
               f"{ctx_k(st.session_state.get('context_tokens', 262144))}-token average conversation · "
               + ("data centres **HELD** at today's spend (contracted leases to FY33 — the sensitivity)"
                  if held else "servers and data centres **follow** the fleet (headline basis)")
               + " · $B per year unless stated")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Spend Helarctos makes unnecessary (FY2026)", f"{bn(t26['spend_cut'])}/yr",
              delta=f"{pct(t26['pct_cut'])} of AI spend")
    k2.metric("AI spend (FY26)", f"{bn(spend26)}")
    k3.metric("Net AI cash result with Helarctos", bn(t26["net_with"]),
              delta=f"today: {bn(t26['net_now'])}", delta_color="off")
    k4.metric(f"Capitalized value (at {g['discount_rate']:.0%})", f"${t26['spend_cut'] / g['discount_rate'] / 1000:.1f}T")

    # what the scenario switch changes — both scenarios side by side, at the active datacenter switch
    cur, opt = _scenario_levers(g)
    _, tc = value_bridge(g, comps, "fy26", cur, ts, follow=fol)
    _, to = value_bridge(g, comps, "fy26", opt, ts, follow=fol)
    gpus = lambda d: d * 1e9 / g["gpu_cost"]  # noqa: E731
    section("Current kernels → Optimized kernels: what changes, and what it does to the bill")
    show_table(["", "Current kernels (measured)", "Optimized kernels (funded programme, TARGET)"], [
        [("H3 · training speed per token, same size", "h"), (f"×{cur['train_speed']:.2f}", "p"), (f"×{opt['train_speed']:.2f}", "p")],
        [("Training GPU-hour lever (H1 × H2 × H3)", ""), (f"×{cur['train_lever']:.1f}", "b"), (f"×{opt['train_lever']:.1f}", "b")],
        [("Prompt-processing speed vs the transformer (supporting)", ""), (f"×{cur['prefill_speed']:.1f}", "p"), (f"×{opt['prefill_speed']:.1f}", "p")],
        [("Inference compute lever (tokens per GPU at equal size × size factor)", ""),
         (f"×{cur['serving_compute_lever']:,.0f}", "b"), (f"×{opt['serving_compute_lever']:,.0f}", "b")],
        [("Training-fleet GPUs after (FY2026, GPU-equivalents)", ""),
         (n0(gpus(tc["train_fleet"] - tc["train_saved"])), "b"), (n0(gpus(to["train_fleet"] - to["train_saved"])), "b")],
        [("Inference-fleet GPUs after (FY2026, GPU-equivalents)", ""),
         (n0(gpus(tc["serve_fleet"] - tc["serve_saved"])), "b"), (n0(gpus(to["serve_fleet"] - to["serve_saved"])), "b")],
        [("Spend made unnecessary, FY2026", "!"), (bn(tc["spend_cut"]), "b!"), (bn(to["spend_cut"]), "b!")],
    ], wrap=True)
    _act = to if st.session_state.get("scenario") == "Optimized kernels" else tc
    shrink = 1.0 / max(1e-9, 1.0 - _act["fleet_cut"])
    st.caption(f"The bill barely moves between scenarios because the chip fleet already shrinks ~{shrink:,.0f}× "
               f"(1 ÷ (1 − the {_act['fleet_cut']:.1%} fleet cut)): once that is a few hundred ×, kernel maturity changes "
               f"the multiple, not the dollars — the remaining bill is the servers and buildings that cannot shrink "
               f"below one per site plus power. Active scenario: **{st.session_state.get('scenario')}**.")

    section("The headline")
    dc_lab = "…data centres: buildings, power & cooling, network — HELD (leases to FY33)" if held else \
        "…data centres: buildings, power & cooling, network — follow the fleet"
    show_table(["", "FY2026", "FY2025 (last year)"], [
        [("AI spend: chips, servers, data centres, power (P&E additions + electricity; no labor)", ""),
         (bn(t26["ai_capex"] + t26["ai_opex"]), "b"), (bn(t25["ai_capex"] + t25["ai_opex"]), "b")],
        [("…of which AI chips (GPUs, TPUs)", ""), (bn(t26["accel"]), "b"), (bn(t25["accel"]), "b")],
        [("…of which the servers around them", ""), (bn(t26["servers"]), "b"), (bn(t25["servers"]), "b")],
        [("…of which data centres (buildings, power & cooling, network)", ""), (bn(t26["datacenter"]), "b"),
         (bn(t25["datacenter"]), "b")],
        [("…of which electricity for the chips, per year", ""), (f"${t26['power']:,.1f}B", "b"), (f"${t25['power']:,.1f}B", "b")],
        [("AI revenue", ""), (bn(t26["ai_rev"]), "b"), (bn(t25["ai_rev"]), "b")],
        [("Net AI cash result today", ""), (bn(t26["net_now"]), "b"), (bn(t25["net_now"]), "b")],
        [("Spend Helarctos makes unnecessary, per year", "!"), (bn(t26["spend_cut"]), "k!"),
         (bn(t25["spend_cut"]), "k!")],
        [("…AI chips and their power", ""), (bn(t26["accel_saved"] + t26["opex_saved"]), "b"),
         (bn(t25["accel_saved"] + t25["opex_saved"]), "b")],
        [("…the servers around those chips — follow the fleet", ""), (bn(t26["servers_saved"]), "b"),
         (bn(t25["servers_saved"]), "b")],
        [(dc_lab, ""), (bn(t26["datacenter_saved"]), "b"), (bn(t25["datacenter_saved"]), "b")],
        [("…as a share of all AI spend", ""), (pct(t26["pct_cut"]), "b!"), (pct(t25["pct_cut"]), "b!")],
        [("Net AI cash result with Helarctos", ""), (bn(t26["net_with"]), "b"),
         (bn(t25["net_with"]), "b")],
        [("Yearly saving, capitalized at the discount rate", ""),
         (f"${t26['spend_cut'] / g['discount_rate'] / 1000:.1f}T", "b"),
         (f"${t25['spend_cut'] / g['discount_rate'] / 1000:.1f}T", "b")],
    ], wrap=True)

    tot = t26["spend_cut"]
    section(f"FY2026 saving by fleet — adds to {bn(tot)}")
    st.caption("A subordinate view of the one headline above. Each fleet's figure includes its chips, their power, "
               "and its share of the servers and data centres that follow the fleet — attributed by how much of the "
               "chip saving each fleet produced (the model's attribution).")
    lever_rows = {"Training": ["H1", "H2", "H3"], "Inference": ["H4", "H5", "H6"]}
    lever_text = {"H1": f"Same quality with ~{param_matching_fraction(DECK_DEPLOYMENT_SCALE):.0%} of the parameters "
                        f"(projected from models we trained) — also multiplies inference: a "
                        f"{lv['smaller_model']:.2f}× smaller model costs ~{lv['smaller_model']:.2f}× less per token.",
                  "H2": "Compute-optimal training needs data in proportion to model size.",
                  "H3": f"×{lv['train_speed']:.2f} per token at the same size over a modern 8k/64k/256k curriculum "
                        f"({'MEASURED step costs; curriculum shares illustrative' if st.session_state.get('scenario') != 'Optimized kernels' else 'fused-kernel TARGET'}).",
                  "H4": "Fixed-size state instead of a memory that grows with every token (measured ×2,032 at "
                        "262k on our test model).",
                  "H5": f"{lv['conversations_per_gpu']:,.0f} resident conversations per GPU where a transformer fits 1 at 262k "
                        f"(measured per layer, 2026-10-01 decode receipt).",
                  "H6": f"×{lv['faster_decode']:.2f} per-step decode speed vs the transformer, measured per layer "
                        f"({FFN_TXT}). "
                        + h6_meaning(lv['conversations_per_gpu'], lv['faster_decode'])}
    train_amt = t26["training_total"]
    infer_amt = t26["inference_total"]
    followed = t26["servers_saved"] + t26["datacenter_saved"]
    body = []
    for fleet, val, lever, how in (
        ("Training", train_amt, f"×{lv['train_lever']:.1f}",
         f"H1 × H2 × H3 = ~{lv['train_lever']:.0f}× fewer GPU-hours per training run, so training clusters can be "
         f"that much smaller. The GPUs never bought are capex avoided; their power is saved every year."),
        ("Inference", infer_amt, f"×{lv['serving_gpu_lever']:,.0f}",
         f"Tokens per GPU at equal size rise ~{lv['serving_compute_lever'] / lv['size_factor']:,.0f}× (H5 × H6, with "
         f"prompts added back) × the ×{lv['size_factor']:.2f} smaller equal-quality model (H1) = ×{lv['serving_compute_lever']:,.0f}. "
         f"A GPU is bought whole, so the inference fleet shrinks by the smaller of memory (×{lv['memory']:,.0f}) and "
         f"that: ~{lv['serving_gpu_lever']:,.0f}× fewer GPUs, plus the power they would draw."),
    ):
        body.append([(f"{fleet} fleet (chips + power + its share of the followed servers and data centres)", ""),
                     (bn(val), "b"), (pct(val / tot), "b"), (lever, "b"), (how, "")])
        for code in lever_rows[fleet]:
            body.append([("↳ " + lever_label(code), "h"), ("", ""), ("", ""),
                         (lever_fmt(code, lv[LEV[code]["key"]]), lever_value_code(code)),
                         (lever_text[code], "")])
    body.append([("↳ of which: the servers and data centres that follow the fleet (inside both fleets above)", ""),
                 (bn(followed), "b"), (pct(followed / tot), "b"), (f"−{t26['fleet_cut']:.1%}", "b"),
                 (f"Servers {bn(t26['servers_saved'])} + data centres {bn(t26['datacenter_saved'])}"
                  + (" (HELD: leases contracted to FY33 — untick the sidebar switch to let them follow)" if held else
                     f": both are sized by the GPUs, so they fall with the blended chip-fleet cut ({t26['fleet_cut']:.1%}); "
                     f"tick the sidebar switch to hold the data centres (leases to FY33)")
                  + ". Attributed to training / inference by each fleet's share of the chip saving.", "")])
    body.append([("TOTAL — the headline", "s!"), (bn(tot), "s!"), ("100%", "s!"), ("", "s"),
                 ("Step-by-step build and the one-lever-at-a-time view: Value Bridge tab.", "s")])
    show_table(["Fleet / Helarctos lever", "$B per year", "Share", "Lever", "How it works"], body, wrap=True)
    _stacked_bar(f"FY2026 saving {bn(tot)}/yr — training vs inference (each incl. power and its share of the "
                 f"followed servers and data centres)",
                 ["Training", "Inference"], [train_amt, infer_amt], SOURCE_COLORS[:2])

    section("By company — FY2026, $B per year")
    show_table(["Company", "AI spend", "Training saving", "Inference saving", "Servers & data centres",
                "Total saving", "% of AI spend", "Net AI today", "Net AI with Helarctos"],
               [[(r["name"], "s!" if r is t26 else ""), (bn(r["ai_capex"] + r["ai_opex"]), "s" if r is t26 else "b"),
                 (bn(r["train_saved"] + r["train_power_saved"]), "s" if r is t26 else "b"),
                 (bn(r["serve_saved"] + r["infer_power_saved"]), "s" if r is t26 else "b"),
                 (bn(r["servers_saved"] + r["datacenter_saved"]), "s" if r is t26 else "b"),
                 (bn(r["spend_cut"]), "s!" if r is t26 else "b!"), (pct(r["pct_cut"]), "s" if r is t26 else "b"),
                 (bn(r["net_now"]), "s" if r is t26 else "b"), (bn(r["net_with"]), "s!" if r is t26 else "b")]
                for r in rows26 + [t26]], wrap=True)

    section('Why ~99% of the chip bill — not "1,000×"')
    st.markdown(f"Multiples don't multiply. A GPU is bought whole, so the inference fleet shrinks by whichever "
                f"need falls least (×{lv['serving_gpu_lever']:,.0f}), never by memory × tokens per GPU "
                f"({lv['memory']:,.0f} × {lv['serving_compute_lever']:,.0f}). The compute side is itself a product "
                f"that does *not* multiply the dollars further: tokens per GPU at equal size "
                f"(×{lv['serving_compute_lever'] / lv['size_factor']:,.0f}, H5 × H6 with prompts) × the "
                f"×{lv['size_factor']:.2f} smaller equal-quality model (H1 — it costs proportionally less per token in "
                f"inference as in training). Cutting a fleet {lv['serving_gpu_lever']:,.0f}× already removes over 99% "
                f"of it; bigger multiples only move the last fraction of a percent. So the dollars are set by how much "
                f"these firms spend on chips, and on the servers and data centres sized by those chips — which is why "
                f"this model reports dollars.")

    section("How to read this app")
    st.markdown(LEGEND_MD, unsafe_allow_html=True)
    st.caption("Teal = one of the six things the Helarctos architecture changes (H1–H6); every other input is "
               "market, company or modelling data. ◆ = the only values that change between Current kernels and "
               "Optimized kernels. ★ = moves the FY2026 saving by \\$10B or more (What Matters tab). Cash basis; "
               "FY2026 is the base year (capex per company guidance or actual, noted on each company tab). AI spend = property & equipment additions (chips, servers, data centres) "
               "plus electricity — no labor. The servers and data centres sized by the GPUs follow the fleet by "
               "default; holding the data centres (contracted leases to FY33) is the sidebar sensitivity. Not "
               "investment advice.")


def value_bridge_tab(comps, g):
    lv, ts, fol = _bridge_levers(g), _train_shares(), _follow(g)
    held = _held(g)
    rows, t = value_bridge(g, comps, "fy26", lv, ts, follow=fol)
    st.caption("Read top to bottom. Teal rows are the Helarctos levers (edit them in the sidebar and on the "
               "Levers tab); blue cells are formulas. "
               + ("Data centres are HELD (sidebar switch): contracted leases to FY33." if held else
                  "Servers and data centres follow the fleet (headline basis; sidebar switch holds the data centres)."))
    section("Step 1 — What the six companies spend on AI this year (FY2026): P&E additions + electricity, no labor")
    show_table(["Item", "$B", "Note"], [
        [("AI data-centre capex", ""), (bn(t["ai_capex"]), "b"), ("company filings and guidance", "")],
        [("…of which AI chips (GPUs, TPUs)", "!"), (bn(t["accel"]), "b!"),
         ("the part the Helarctos levers act on directly", "")],
        [("…of which the servers around them (CPUs, chassis, networking)", ""), (bn(t["servers"]), "b"),
         ("the rest of the server bucket; sized by the accelerator count — follows the fleet", "")],
        [("…of which data centres (buildings, power & cooling, network)", ""), (bn(t["datacenter"]), "b"),
         ("sized by the GPUs they house — " + ("HELD here (leases contracted to FY33)" if held else "follows the fleet by default"), "")],
        [("Power & operations for those chips, per year", ""), (f"${t['ai_opex']:,.1f}B", "b"), ("electricity; always follows the fleet", "")],
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

    section("Step 3 — Training: a smaller model, fewer tokens, faster per token → smaller training clusters")
    show_table(["Item", "$B", "Lever", "Note"], [
        [("Training fleet capex today", ""), (bn(t["train_fleet"]), "b"), ("", ""), ("", "")],
        lever_row("H1", "same quality with a fraction of the parameters (projected)"),
        lever_row("H2", "a smaller model needs proportionally fewer training tokens"),
        lever_row("H3", f"faster per token at the same size over a modern 8k/64k/256k curriculum "
                        f"({'MEASURED step costs, illustrative curriculum shares' if st.session_state.get('scenario') != 'Optimized kernels' else 'fused-kernel TARGET'})"),
        [("GPU-hours per training run fall by", "!"), ("", ""), (f"×{lv['train_lever']:.1f}", "b!"), ("H1 × H2 × H3", "")],
        [("Training fleet needed with Helarctos", ""), (bn(t["train_fleet"] / lv["train_lever"]), "b"), ("", ""), ("", "")],
        [("Training capex avoided", ""), (bn(t["train_saved"]), "b"), ("", ""), ("", "")],
        [("+ power & operations those GPUs would have drawn", ""), (f"${t['train_power_saved']:,.1f}B", "b"),
         ("", ""), ("power scales with the fleet", "")],
        [("TRAINING SAVING (chips + power)", "!"), (bn(t["train_saved"] + t["train_power_saved"]), "b!"), ("", ""),
         (f"labs size training clusters to GPU-hours: ~{lv['train_lever']:.0f}× fewer GPU-hours means a cluster "
          f"~{lv['train_lever']:.0f}× smaller for the same programme", "")],
    ], wrap=True)
    section("Step 4 — Inference: small memory → more conversations per GPU, more conversations advancing per decode step, on a smaller model → fewer GPUs")
    eq = lv["serving_compute_lever"] / lv["size_factor"]
    show_table(["Item", "$B", "Lever", "Note"], [
        [("Inference fleet capex today", ""), (bn(t["serve_fleet"]), "b"), ("", ""), ("", "")],
        lever_row("H4", "fixed-size state instead of a memory that grows with every token"),
        lever_row("H5", f"{lv['conversations_per_gpu']:,.0f} resident 262k-token conversations per GPU where a transformer "
                        f"fits 1 (measured per layer, 2026-10-01 decode receipt)"),
        lever_row("H6", f"per-step decode speed vs the transformer at its feasible batch (measured per layer; {FFN_TXT}). "
                        + h6_meaning(lv["conversations_per_gpu"], lv["faster_decode"])),
        [("Tokens per GPU at equal model size, H5 × H6 with prompt processing added back", ""), ("", ""),
         (f"×{eq:,.0f}", "b"),
         (f"prompts are ~0.2% of a transformer's time at 262k (prompt-processing speed ×{lv['prefill_speed']:.1f}, "
          f"the only inference input that differs by scenario)", "")],
        [("× the smaller equal-quality model (H1) — the inference size factor", ""), ("", ""),
         (f"×{lv['size_factor']:.2f}", "b"),
         ("a model with ~24% of the parameters costs proportionally less per token; applied to inference as to "
          "training (2026-10-01)", "")],
        [("= Inference compute lever", "!"), ("", ""), (f"×{lv['serving_compute_lever']:,.0f}", "b!"), ("", "")],
        [("Inference GPUs needed fall by (the smaller of memory and the compute lever)", "!"), ("", ""),
         (f"×{lv['serving_gpu_lever']:,.0f}", "b!"),
         ("a GPU is bought whole — memory and compute together — so the fleet covers whichever runs out first; "
          + ("here the compute lever binds, so the per-layer decode step (H6) and the size factor count in the dollars"
             if lv["serving_compute_lever"] <= lv["memory"] else "here memory (H4) binds"), "")],
        [("Inference fleet needed with Helarctos", ""), (bn(t["serve_fleet"] / lv["serving_gpu_lever"]), "b"), ("", ""), ("", "")],
        [("Inference capex avoided", ""), (bn(t["serve_saved"]), "b"), ("", ""), ("", "")],
        [("+ power & operations those GPUs would have drawn", ""), (f"${t['infer_power_saved']:,.1f}B", "b"),
         ("", ""), ("power scales with the fleet", "")],
        [("INFERENCE SAVING (chips + power)", "!"), (bn(t["serve_saved"] + t["infer_power_saved"]), "b!"), ("", ""), ("", "")],
    ], wrap=True)
    section("Step 5 — The servers and data centres sized by those GPUs follow the fleet")
    show_table(["Item", "$B", "Factor", "Note"], [
        [("Chip fleet cut (training and inference blended)", "!"), ("", ""), (f"−{t['fleet_cut']:.1%}", "b!"),
         ("chip capex avoided ÷ chip capex today; the datacenters are a hybrid of training and serving, so the "
          "same blend applies to them", "")],
        [("Servers around the chips today", ""), (bn(t["servers"]), "b"), ("", ""), ("", "")],
        [("…follow the fleet", ""), ("", ""), (f"{fol['servers']:.0%}", "y"), ("fraction of the bucket that scales with the accelerator count", "")],
        [("Servers saved", ""), (bn(t["servers_saved"]), "b"), ("", ""), ("", "")],
        [("Data centres today (buildings, power & cooling, network)", ""), (bn(t["datacenter"]), "b"), ("", ""), ("", "")],
        [("…follow the fleet", ""), ("", ""), (f"{fol['datacenter']:.0%}", "y"),
         ("HELD: contracted leases to FY33 (sidebar switch)" if held else
          "default: they scale down when no longer necessary; tick the sidebar switch to hold them (leases to FY33)", "")],
        [("Data centres saved", ""), (bn(t["datacenter_saved"]), "b"), ("", ""), ("", "")],
        [("FOLLOWED SAVING (servers + data centres)", "!"), (bn(t["servers_saved"] + t["datacenter_saved"]), "b!"), ("", ""), ("", "")],
    ], wrap=True)
    section("Result — FY2026")
    show_table(["Item", "$B"], [
        [("Training (chips + power)", ""), (bn(t["train_saved"] + t["train_power_saved"]), "b")],
        [("Inference (chips + power)", ""), (bn(t["serve_saved"] + t["infer_power_saved"]), "b")],
        [("Servers and data centres that follow the fleet", ""), (bn(t["servers_saved"] + t["datacenter_saved"]), "b")],
        [("SPEND HELARCTOS MAKES UNNECESSARY, per year", "!"), (bn(t["spend_cut"]), "k!")],
        [("…as a share of all AI spend", "!"), (pct(t["pct_cut"]), "b!")],
        [("…as a share of the AI-chip bill", ""), (pct(t["accel_saved"] / t["accel"]), "b")],
        [("Net AI cash result with Helarctos", "!"), (bn(t["net_with"]), "b!")],
        [("Value of the yearly saving, capitalized (÷ discount rate)", ""),
         (f"${t['spend_cut'] / g['discount_rate'] / 1000:.1f}T", "b")],
    ], wrap=True)
    section("Switch the Helarctos levers on one at a time (FY2026)")
    lad = savings_ladder(g, comps, "fy26", lv, ts, follow=fol)
    labels = [s_["step"].replace(" (training)", " — training (H1–H3)").replace(" (inference)", " — inference (H4–H6)")
              .replace("the servers and datacenters sized by those GPUs follow",
                       "the servers sized by those GPUs follow (data centres held)" if held
                       else "the servers and data centres sized by those GPUs follow")
              for s_ in lad]

    def _fol(f):
        return ("servers + data centres" if f["datacenter"] else "servers") if f["servers"] else "—"

    show_table(["Step", "Training GPU-hours fall by", "Inference memory falls by", "Inference compute lever",
                "Buckets that follow", "Chip fleet cut", "Spend cut, $B/yr", "Added by this step"],
               [[(labels[i], "!" if i == len(lad) - 1 else ""), (f"×{s_['train_lever']:.1f}", "b"),
                 (f"÷{s_['memory']:.0f}", "b"), (f"×{s_['serving_compute_lever']:,.0f}", "b"),
                 (_fol(s_["follow"]), "y" if i == len(lad) - 1 else ""),
                 (f"{s_['fleet_cut']:.1%}", "b"), (bn(s_["spend_cut"]), "b!"),
                 (f"+${s_['increment']:,.0f}B" if i else "—", "b")]
                for i, s_ in enumerate(lad)], wrap=True)
    _colored_bars([f"{i}. {lab}" for i, lab in enumerate(labels)], [s_["spend_cut"] for s_ in lad],
                  LADDER_COLORS, horizontal=True)
    st.caption("Training savings come from the smaller model (fewer parameters AND fewer tokens compound) trained "
               "faster per token. Inference savings come from fixed-size memory, which lets each GPU hold many more "
               "long conversations, more conversations advancing per decode step, on a smaller model. Each fleet then shrinks by ~96–99.9%: "
               "bigger multiples can't add much, because a cost can only fall to zero once. The last step is not a "
               "Helarctos lever at all — it is the servers and data centres sized by the GPUs following the fleet. "
               "That is why this model reports dollars, not multiples.")
    _, tt = compute_year(g, comps, "fy26")
    chips_only = t["train_saved"] + t["serve_saved"] + t["opex_saved"]
    st.caption(f"Cross-check: the technical Totals tab prices memory and compute separately and counts chips plus the "
               f"legacy datacenter-scaling share — {md_usd(tt['spend_cut'])}B FY26 — against {md_usd(chips_only)}B for "
               f"chips and power here (within rounding: both remove ~99.9% of the chip bill). The remaining "
               f"{md_usd(t['servers_saved'] + t['datacenter_saved'])}B here is the servers and data centres following "
               f"the fleet, which the Totals engine does not count.")

    section("Detail by company — FY2026, $B (the engine behind every number above)")
    det_cols = ["Company", "AI-chip capex", "Training share", "Training fleet", "Inference fleet",
                "Training capex avoided", "Inference capex avoided", "Servers around chips today", "Servers saved",
                "Data centres today", "Data centres saved", "Power today", "Power saved", "TOTAL saving",
                "AI spend (capex + power)", "% of AI spend cut", "AI revenue", "Net AI today", "Net AI with Helarctos"]
    det = []
    for r in rows + [t]:
        tone = "s" if r is t else "b"
        det.append([(r["name"], tone + "!" if r is t else ""), (n1(r["accel"]), tone), (pct(r["train_share"]), tone),
                    (n1(r["train_fleet"]), tone), (n1(r["serve_fleet"]), tone), (n1(r["train_saved"]), tone),
                    (n1(r["serve_saved"]), tone), (n1(r["servers"]), tone), (n1(r["servers_saved"]), tone),
                    (n1(r["datacenter"]), tone), (n1(r["datacenter_saved"]), tone), (n1(r["power"]), tone),
                    (n1(r["opex_saved"]), tone), (n1(r["spend_cut"]), tone + "!"),
                    (n1(r["ai_capex"] + r["ai_opex"]), tone), (pct(r["pct_cut"]), tone), (n1(r["ai_rev"]), tone),
                    (n1(r["net_now"]), tone), (n1(r["net_with"]), tone + "!")])
    show_table(det_cols, det, widths={"Company": "small"})


def _scenario_levers(g):
    """Lever sets for both scenarios at the sidebar workload (◆ = differs)."""
    cur = value_bridge_levers(dict(g, flop_factor=g.get("flop_factor_current", g["flop_factor"])), kernels="current")
    opt = value_bridge_levers(dict(g, flop_factor=g.get("flop_factor_current", g["flop_factor"])), kernels="mature")
    tp_opt = float(g.get("flop_factor_optimized", opt["serving_compute_lever"]))
    opt["serving_throughput"] = opt["serving_compute_lever"] = tp_opt
    opt["serving_gpu_lever"] = min(opt["memory"], tp_opt)
    return cur, opt


def levers_tab(comps, g):
    lv, fol = _bridge_levers(g), _follow(g)
    held = _held(g)
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
        if code == "H3" and st.session_state.get("scenario") == "Optimized kernels":
            status = (f"TARGET ×{TRAIN_SPEED_BY_SCENARIO['mature']:.2f} — the fused-kernel 8k gate (not a receipt); "
                      f"×{TRAIN_SPEED_BY_SCENARIO['current']:.2f} is MEASURED on today's kernels")
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
    rows.append([("", ""), ("Supporting: the inference size factor (= H1)", ""),
                 (f"×{lv['size_factor']:.2f}", "b"), ("same", ""), ("same", ""),
                 ("PROJECTED — the same fit-derived equal-quality ratio as H1 (reinstated for inference 2026-10-01)", ""),
                 ("A smaller equal-quality model costs proportionally less per token, so the ratio multiplies the "
                  "inference compute lever as well as the training lever.", ""), ("Inference GPUs (tokens per GPU)", "")])
    show_table(["#", "Helarctos lever", "Active value", "◆ Current kernels", "◆ Optimized kernels",
                "How sure are we?", "What it means", "Saves money in"], rows, wrap=True)
    st.caption("Receipts: H5 and H6 from the 2026-10-01 per-layer decode receipt (one layer, bf16, graph-captured "
               f"one-token step; the transformer at its largest measured batch whose full-model cache fits a "
               f"{GEOM['hbm_gb']:.1f} GB card at that context, us at {H5_CONVERSATIONS:.0f} streams; {FFN_TXT}), "
               f"quoted at {GEOM_TXT} — anchored on {GEOM['anchor']}; the per-layer step "
               f"ratio is depth-invariant but the transformer's cache capacity and cache-read-bound step time are not. "
               "H3 from the 2026-08-24/29 single-layer training-step measurements at 2k and 8k, "
               "weighted over a modern 8k/64k/256k curriculum whose token shares are illustrative. H4 from the "
               "2026-08-14 full-model decode measurement on our test model. H1/H2 from the sealed 2026-08-14 quality "
               "refit (a projection of two fits).")
    section("How the levers combine — one number per fleet")
    eq = lv["serving_compute_lever"] / lv["size_factor"]
    show_table(["Fleet lever", "Value", "Built from"], [
        [("Training: GPU-hours per training run fall by", "!"), (f"×{lv['train_lever']:.1f}", "b!"),
         ("H1 × H2 × H3 — training clusters are sized to GPU-hours", "")],
        [("Inference: memory per conversation falls by", "!"), (f"÷{lv['memory']:,.0f}", "b!"), ("H4", "")],
        [("Inference: decode tokens per GPU rise by", "!"),
         (f"×{lv['conversations_per_gpu'] * lv['faster_decode']:,.0f}", "b!"),
         (f"H5 × H6: conversations per GPU × the per-step time ratio (measured per layer, 2026-10-01, at {GEOM_TXT})", "")],
        [("Inference: tokens per GPU at equal model size, including prompt processing", "!"),
         (f"×{eq:,.0f}", "b!"),
         ("H5 × H6 with prompts added back at the supporting prompt-processing speed", "")],
        [("Inference: compute lever = tokens per GPU × the equal-quality size factor", "!"),
         (f"×{lv['serving_compute_lever']:,.0f}", "b!"),
         (f"× H1 (×{lv['size_factor']:.2f}): a smaller equal-quality model costs proportionally less per token, in "
          f"inference as in training (2026-10-01)", "")],
        [("Inference: GPUs needed fall by", "!"), (f"×{lv['serving_gpu_lever']:,.0f}", "b!"),
         (f"the smaller of memory (H4) and the compute lever — a GPU is bought whole, so the fleet covers whichever "
          f"runs out first (here {'the compute lever, so the decode step and the size factor count' if lv['serving_compute_lever'] <= lv['memory'] else 'memory'}); "
          f"multiplying them ({lv['memory']:,.0f} × {lv['serving_compute_lever']:,.0f}) would be wrong", "")],
        [("Servers and data centres sized by those GPUs follow the fleet", "!"),
         (f"servers {fol['servers']:.0%} · data centres {fol['datacenter']:.0%}", "y"),
         ("not a Helarctos lever: each followed bucket falls by the blended chip-fleet cut × this fraction. "
          + ("Data centres HELD (sidebar): contracted leases to FY33." if held else
             "Default: both follow (2026-10-01 ruling). Sidebar switch holds the data centres (leases to FY33)."), "")],
        [("For reference: technical tabs' blended cut", ""), (f"×{reduction_factor(g):,.0f}", "b"),
         ("the appendix prices memory (~60% of a GPU's cost) and compute separately; same chip bill, both remove "
          "~99.9% of it", "")],
    ], wrap=True)
    section("Market & company data — the other inputs (not about Helarctos)")
    _, t = value_bridge(g, comps, "fy26", lv, _train_shares(), follow=fol)
    show_table(["Input", "Value", "Where to edit", "Note"], [
        [("AI data-centre capex, FY2026 (six companies)", ""), (bn(t["ai_capex"]), "b"),
         ("✏️ panel on each company tab ★", ""), ("disclosed / guided capex × data-centre share (10-K/10-Q notes); P&E additions, no labor", "")],
        [("…share that buys AI chips", ""), (pct(t["accel"] / t["ai_capex"]), "b"),
         ("✏️ panel on each company tab ★", ""), ("server share (CFO-disclosed) × accelerator share of servers", "")],
        [("…the servers around the chips", ""), (bn(t["servers"]), "b"), ("✏️ panel on each company tab", ""),
         ("server bucket − accelerators; follows the fleet", "")],
        [("…data centres: buildings, power & cooling, network", ""), (bn(t["datacenter"]), "b"),
         ("✏️ panel on each company tab ★", ""), ("AI data-centre capex − server bucket; " + ("HELD (sidebar)" if held else "follows the fleet (sidebar switch holds it)"), "")],
        [("Training share of the chip fleet (chip-weighted)", ""), (pct(t["train_share"]), "b"),
         ("per company, below", ""), ("analyst estimates; no company discloses this", "")],
        [("Fully-loaded cost per GPU", ""), (usd0(g["gpu_cost"]), "b"), ("sidebar", ""),
         ("sizes the fleet for the power calculation", "")],
        [("Wall power per GPU (kW)", ""), (f"{g['wall_power_kw']:.1f}", "b"), ("sidebar", ""),
         ("power is ~2% of the saving", "")],
        [("Electricity ($/kWh)", ""), (f"${g['elec_rate']:.2f}", "b"), ("sidebar", ""), ("power is ~2% of the saving", "")],
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
    st.markdown("- AI spend = property & equipment additions (AI chips, the servers around them, data-centre "
                "buildings, power & cooling, network) plus electricity. No labor is counted anywhere.\n"
                "- The servers and data centres sized by the GPUs follow the blended chip-fleet cut by default "
                "(2026-10-01: they scale down when no longer necessary). Holding the data centres — leases "
                "contracted to FY33 — is the sidebar sensitivity and delays, not removes, that saving.\n"
                "- Training is priced on GPU-hours only; no memory credit is taken on training clusters.\n"
                "- H4 (×2,032) is measured on our test model; for frontier-size models with grouped-query attention "
                "the estimate is ~×208, where memory would bind again (about −\\$2B on FY2026). H5 and H6 are "
                f"measured per layer on the 2026-10-01 decode receipt; {FFN_TXT}. H3 is measured step cost over an "
                "illustrative curriculum mix.\n"
                "- H1 is a projection: quality trends measured up to 663M parameters, extended to ~1T "
                "dense-equivalent — about what today's 5–6T-total mixture-of-experts flagships (Grok 5 at 6T, "
                "Kimi K3 at 2.8T) amount to. It multiplies inference as well as training.\n"
                "- The saving is spend no longer needed for the same AI output; firms will likely reinvest it. "
                "Cash basis; capitalized = yearly saving ÷ discount rate.")


def _wm_why():
    """Why-it-matters notes for the What Matters rows, with the live values."""
    def _tp(ctx):
        h5, h6 = decode_levers(ctx)
        return inference_throughput(h5, h6, prefill_advantage("current", ctx), prompt_time_ratio(ctx)) * INFERENCE_SIZE_FACTOR
    out = {
        "Server share of AI capex": "How much AI capex buys servers rather than buildings, power and networking. "
            "With both buckets following the fleet the split barely matters. CFO-disclosed for Microsoft and Alphabet.",
        "Accelerator share of servers": "The GPU/TPU share of server spend. With the servers around the chips "
            "following the fleet, the split barely matters. From teardowns (67–80%).",
        "Smaller model for the same quality": "The biggest Helarctos-specific assumption: it drives the training "
            "saving and multiplies the inference lever. A projection from models up to 663M parameters to ~1T "
            "dense-equivalent (today's 5–6T mixture-of-experts flagships). High case: a 5T dense model.",
        "More conversations per GPU": f"{H5_CONVERSATIONS:.0f} vs the transformer's 1 at 262k (measured per layer, "
            f"2026-10-01). Small effect: inference GPUs already shrink by ~99.9%, so even the 64-stream 2026-08-14 "
            f"cell costs under $1B.",
        "Faster decode per token": f"×{H6_DECODE:.2f} per step at 262k (measured per layer, 2026-10-01; {FFN_TXT}). "
            f"It counts in the dollars, but only ~$0.4B across the range.",
        "Memory per conversation": "Measured ×2,032 at 262k on our test model. Frontier-size models with grouped-query "
            "attention: ~÷208 (estimate), where memory would bind again (≈ −$2B).",
        "Average conversation length": f"Shorter conversations shrink both the memory advantage and tokens per GPU "
            f"(inference lever ×{_tp(32768):,.0f} at 32k); longer ones grow them (×{_tp(1048576):,.0f} at 1M, where one "
            f"transformer conversation no longer fits a GPU). Set it in the sidebar.",
        "Data-center share of capex": "Share of capex that goes into data centres (10-K/10-Q property notes, 93–98%; "
            "Amazon's AWS share 68–76%). Now the whole AI bill follows the fleet, so this moves the dollars one-for-one.",
        "Datacenters follow the fleet": "Default: the data-centre bucket (buildings, power & cooling, network) scales "
            "down with the GPUs it houses. Held: leases are contracted to FY33, so that saving arrives later — the "
            "single largest sensitivity. Sidebar switch.",
        "Training speed per token": f"×{TRAIN_SPEED_BY_SCENARIO['current']:.2f} measured over a modern curriculum; "
            f"×{TRAIN_SPEED_BY_SCENARIO['mature']:.2f} is the fused-kernel target. Small effect: the smaller model "
            f"already removes ~94% of training GPU-hours.",
        "Electricity rate": "Only changes the power part of the saving (~2% of the total).",
        "Wall power per GPU": "Only changes the power part of the saving (~2% of the total).",
        "Training share of the chip fleet": "Small effect: both fleets shrink by 96–99.9%.",
        "Memory share of GPU cost": "No effect on the front tabs (GPUs are bought whole).",
    }
    # the model may rename a lever's sensitivity row; key the notes by the H-code as well
    for code, key in (("H6", "Faster decode per token"), ("H5", "More conversations per GPU"),
                      ("H4", "Memory per conversation"), ("H3", "Training speed per token"),
                      ("H1+H2", "Smaller model for the same quality")):
        out[code] = out[key]
    return out


def _wm_why_lookup(why, name, code):
    return why.get(name) or why.get(code or "", "")


def wm_label(name, code):
    if not code:
        return name
    return "H1 + H2 · Smaller model, trained on fewer tokens ★" if code == "H1+H2" else lever_label(code)


def what_matters_tab(comps, g):
    import altair as alt
    from ai_capex_model import sensitivity_table
    fol = _follow(g)
    _, t = value_bridge(g, comps, "fy26", _bridge_levers(g), _train_shares(), follow=fol)
    _, t_follow = value_bridge(g, comps, "fy26", _bridge_levers(g), _train_shares(), follow=BRIDGE_FOLLOW_20261001)
    st.caption("Each row moves ONE input to a plausible low and high value, everything else as set, and shows "
               f"the change in the FY2026 saving. Deltas are on the headline basis — servers and data centres "
               f"following the fleet ({md_usd(t_follow['spend_cut'])}B/yr)"
               + (f"; the sidebar currently holds the data centres ({md_usd(t['spend_cut'])}B/yr), which is the "
                  f"'Datacenters follow the fleet' row below" if _held(g) else "")
               + ". ★ = moves it by \\$10B or more.")
    rows = sensitivity_table(g, comps)
    WM_WHY = _wm_why()
    section(f"What moves the FY2026 saving ({bn(t_follow['spend_cut'])}/yr) — Helarctos levers vs market data")
    body, chart = [], []
    for gname, items, gcode in (("Helarctos levers — what the architecture changes", [r for r in rows if r[2]], "h"),
                                ("Market & company data — not about Helarctos", [r for r in rows if not r[2]], "s")):
        body.append([("", gcode), (gname, gcode + "!"), ("", gcode), ("", gcode), ("", gcode), ("", gcode),
                     ("", gcode), ("", gcode)])
        for name, where, code, lo_l, lo, hi_l, hi in items:
            big = max(abs(lo), abs(hi)) >= 10.0
            lab = wm_label(name, code)
            if name == "Datacenters follow the fleet":
                where = "sidebar checkbox (workbook: Levers C6)"
            body.append([("★" if big else "", ""), (lab, "h" if code else ("!" if big else "")), (where, ""),
                         (lo_l, ""), (f"{lo:+,.1f}", "b*" if big else "b"), (hi_l, ""),
                         (f"{hi:+,.1f}", "b*" if big else "b"), (_wm_why_lookup(WM_WHY, name, code), "")])
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
               "these firms spend on AI (the data-centre bucket and whether it follows the fleet, then the chip "
               "shares) more than by how large the Helarctos multiples are; the Helarctos levers move it only if "
               "they fall far below today's values.")


# ---- main ----------------------------------------------------------------------
st.title("AI Capex Efficiency")
st.caption("How much AI spend (chips, the servers around them, data centres and their power) the Helarctos "
           "architecture makes unnecessary for the six biggest AI spenders — a tab-for-tab mirror of the workbook. "
           "Inference and training levers from the 2026-10-01 re-base (per-layer decode receipt; the equal-quality "
           "size factor applies to inference too; servers and data centres follow the fleet by default). Set the "
           "scenario, the levers and the datacenter switch in the sidebar.")
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
# the sidebar's ONE headline number: the value-bridge total for the active scenario (and switch),
# rendered after the company-tab edits are read so it matches the Summary exactly
_cur, _opt = _scenario_levers(g)
_, _tc = value_bridge(g, comps, "fy26", _cur, _train_shares(), follow=_follow(g))
_, _to = value_bridge(g, comps, "fy26", _opt, _train_shares(), follow=_follow(g))
_active_opt = st.session_state.get("scenario") == "Optimized kernels"
_now, _other = (_to, _tc) if _active_opt else (_tc, _to)
SIDEBAR_HEAD.metric(f"Spend made unnecessary, FY2026 ({st.session_state.get('scenario')}"
                    f"{'; data centres held' if _held(g) else ''})", f"{bn(_now['spend_cut'])}/yr",
                    delta=f"{'Current' if _active_opt else 'Optimized'} kernels: {bn(_other['spend_cut'])}/yr",
                    delta_color="off")
SIDEBAR_HEAD.caption(f"Scenario switch changes H3 ×{_cur['train_speed']:.2f} → ×{_opt['train_speed']:.2f}, "
                     f"prompt speed ×{_cur['prefill_speed']:.1f} → ×{_opt['prefill_speed']:.1f}, inference lever "
                     f"×{_cur['serving_compute_lever']:,.0f} → ×{_opt['serving_compute_lever']:,.0f}; dollars "
                     f"{bn(_tc['spend_cut'])} → {bn(_to['spend_cut'])} (table on the Summary tab).")
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
