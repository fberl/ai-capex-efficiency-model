"""AI Capex Efficiency — interactive mirror of the AI_Capex_Efficiency workbook.

One Streamlit tab per worksheet: the audience layer (Summary, Value Bridge, Levers, What Matters),
one tab per company, then the technical appendix (Totals, Inputs, Sensitivity, CostLadder,
Serving·Training, Evidence, Methodology). Same colors and markers as the Excel:
  teal H1-H6 = Helarctos lever · yellow = assumption · green = disclosed data · blue = formula
  ◆ purple = differs by scenario · ★ orange border = high-impact input
The sidebar separates the Helarctos levers from market & modeling data. All math comes from
ai_capex_model.py, so the app and the spreadsheet can't drift.

Copy conventions (2026-10-01 rewrite): plain declarative sentences, "we"/"our", no shouting
status words, no process jargon (receipt, banked, re-based, position of record). Dates live
in one "as of" line in the footer and in the Methodology history expander. Strings that come
from ai_capex_model (statuses, labels) are normalized at render time by scrub_external().

Run locally:  uv run --with streamlit --with pandas streamlit run app.py
Deploy free:  push to GitHub -> share.streamlit.io  (needs requirements.txt)
"""

from pathlib import Path

import importlib

import streamlit as st

# ---- external-content scrub (mirrors ai_capex_efficiency._EXTERNAL_SCRUB) ----------
# The app is public: no file paths / module or constant names / kernel-stack internals /
# mechanism terms may reach a viewer. Every Streamlit text surface is wrapped once here.
import re as _re
_EXTERNAL_SCRUB = [
    (_re.compile(r"(?:/home/)?[\w~.-]+(?:/[\w~.()\[\]-]+)+\.(?:json|md|png|py|csv)"), "our measurement archive"),
    (_re.compile(r"\b[\w-]+\.(?:json|md|png|py|csv)\b"), "our measurement archive"),
    (_re.compile(r"\b(?:paper|experiments|bdm|src)/[\w./{},-]+"), "our measurement archive"),
    (_re.compile(r"\bai_capex_model\.\w+"), "the model"),
    (_re.compile(r"\bSERVING\['\w+'\]"), "the model"),
    (_re.compile(r"\bTriton\s+"), ""),
    (_re.compile(r"\bmegakernel\b"), "fused kernel"),
    (_re.compile(r"\brecurrence\b", _re.I), "internal dynamics"),
    (_re.compile(r"\brecurrent\b", _re.I), "internal"),
    (_re.compile(r"\bRNN\b"), "our architecture"),
    (_re.compile(r"\s->\s"), ", so "),   # note-style arrows in the model's source notes
]

# Status words the model writes in capitals. The app shows them in plain case: the color of
# a status cell is decided before the text reaches the screen, so this is cosmetic only.
_PLAIN_CASE = {
    "MEASURED": "measured", "PROJECTED": "projected", "PROJECTION": "projection", "TARGET": "target",
    "ESTIMATE": "estimate", "ILLUSTRATIVE": "illustrative", "ASSUMPTION": "assumption", "HELD": "held",
    "FOLLOW": "follow", "FOLLOWS": "follows", "CONSERVATIVE": "conservative", "LEGACY": "legacy",
    "MODELLED": "modeled", "MODELED": "modeled", "RETIRED": "retired", "SUPERSEDED": "superseded",
    "EXCLUDED": "excluded", "DIRECTIONAL": "directional", "LABEL": "label", "SOURCE": "source",
    "AGAINST": "against", "DRAG": "drag", "CONTRIBUTOR": "contributor", "FLAT": "flat", "GROWS": "grows",
    "NOT": "not", "ONLY": "only", "AND": "and", "ONE": "one", "ALL": "all", "MORE": "more",
    "EQUAL": "equal", "FRONTIER": "frontier", "FIT-DERIVED": "fit-derived", "RE-BASED": "re-based",
    "GLOBAL": "Global", "TOTAL": "Total", "MEMORY": "memory", "COMPUTE": "compute", "PEAK": "peak",
    "CURRICULUM": "curriculum", "DECODE": "decode", "FULL": "full", "LOWER": "lower",
    "PRE-CAMPAIGN": "pre-campaign", "TODAY": "today", "CEILING": "ceiling", "SHARES": "shares",
    "KNOBS": "knobs", "MIX": "mix", "EXACTLY": "exactly", "OWN": "own", "AHEAD": "ahead",
    "CAPEX": "capex", "OPEX": "opex", "FILING": "Filing", "READ FIRST": "read first",
    "SERVING": "serving", "TRAINING": "training", "GRANTED": "granted", "FLAT-TO-FALLING": "flat-to-falling",
    # British spellings in the model's strings; the app is American English
    "modelled": "modeled", "modelling": "modeling", "programme": "program", "favours": "favors",
    "favour": "favor", "unoptimised": "unoptimized", "optimised": "optimized", "realised": "realized",
    "colour": "color", "colours": "colors",
}
_PLAIN_CASE_RE = _re.compile(r"\b(" + "|".join(sorted(map(_re.escape, _PLAIN_CASE), key=len, reverse=True)) + r")\b")


_MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September",
           "October", "November", "December"]
_ISO_DATE_RE = _re.compile(r"\b(20\d\d)-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])\b")


def scrub_external(text):
    if not isinstance(text, str):
        return text
    for pat, rep in _EXTERNAL_SCRUB:
        text = pat.sub(rep, text)
    out = _PLAIN_CASE_RE.sub(lambda m: _PLAIN_CASE[m.group(1)], text)
    out = _ISO_DATE_RE.sub(lambda m: f"{_MONTHS[int(m.group(2)) - 1]} {m.group(1)}", out)
    # a status word that opened a sentence keeps its capital ("Estimate: ...")
    if " " in out.strip() and text[:1].isupper() and out[:1].islower():
        out = out[0].upper() + out[1:]
    return out


def _wrap_text_api(name):
    orig = getattr(st, name)

    def wrapped(body=None, *a, **k):
        if "help" in k:
            k["help"] = scrub_external(k["help"])
        return orig(scrub_external(body), *a, **k) if body is not None else orig(*a, **k)
    setattr(st, name, wrapped)


for _n in ("markdown", "caption", "write", "info", "warning", "text", "subheader", "header", "title"):
    _wrap_text_api(_n)
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

GEOM = DECODE_GEOMETRY_OF_RECORD  # the ~90-layer grouped-query frontier model the decode levers are quoted at
GEOM_TXT = (f"a frontier-size model of about {GEOM['layers']} layers with grouped-query attention "
            f"({GEOM['kv_bytes_per_token_per_layer'] // 1024} KB of cache per token per layer)")
GEOM_ANCHOR = GEOM["anchor"].replace(" -> ~", ", so about ").replace(" -> ", ", so ")
AS_OF = "1 October 2026"
_FFN_EST = "bytes-bound" in DECK_LAYER_CARD_20261001["status"]
FFN_TXT = ("our feed-forward block is added as an estimate from memory traffic" if _FFN_EST else
           "both steps include their feed-forward blocks, and ours runs uncompiled, so the ratio understates us")
# How H3 (training speed per token) was measured, in one sentence. Used wherever H3 appears.
H3_HOW = ("we are slower per step than the transformer at short context (×1.72 at 2k and ×1.79 at 8k, "
          "measured on one layer), faster past a crossover the fit puts near 13k tokens (measured at 21–24k "
          "on another setup), and the ×{v:.2f} is the cost ratio over an assumed 8k/64k/256k training mix, "
          "fitted beyond 8k")


def h3_how(v):
    return H3_HOW.format(v=v)


def h6_meaning(h5, h6):
    """Plain-language reading of H6 (per-step decode ratio), whichever side of 1 it lands."""
    if h6 >= 1.0:
        return (f"One decode step for all {h5:,.0f} live conversations takes less time than the transformer's "
                f"step for its single conversation (×{h6:.2f}), because there is no growing cache to re-read.")
    return (f"The transformer's single long-context conversation still steps faster than our "
            f"{h5:,.0f}-conversation step (×{h6:.2f}, so our step is {1 / h6:.2f}× longer). The gain is H5: "
            f"{h5:,.0f} conversations advance per step instead of one.")

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
    text wraps instead of being cut off. (Docstring first: Streamlit renders a bare
    string expression that follows a statement.)"""
    rows = [[(scrub_external(c[0]),) + tuple(c[1:]) if isinstance(c, tuple) else scrub_external(c) for c in r] for r in rows]
    columns = [scrub_external(c) for c in columns]
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


# ---- sidebar: Helarctos levers vs market & modeling data ----------------------
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
    s.caption("Change any value and every tab recomputes. The six **H1–H6** levers are the only inputs about "
              "the Helarctos architecture. Everything else is market, company or modeling data.")
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
    s.subheader("Market & modeling data")
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
            help="Input tokens per generated token. About 10:1 for code and agent traces, 50–100:1 for retrieval-heavy use."),
        "context_tokens": int(wl_box.number_input(
            "Average conversation length (tokens)", min_value=1024, max_value=1048576, step=1024,
            key="context_tokens", format="%d",
            help="Average conversation length across the workload. H4–H6 are quoted at 262k. Changing this "
                 "resets H5 and H6 to their values at the new length.")),
        "train_share": wl_box.number_input(
            "Training share (workload blend, technical tabs)", min_value=0.0, max_value=1.0, step=0.05,
            key="train_share", format="%.2f",
            help="Only feeds the measured workload blend on the Serving·Training tab."),
    }
    adv = workload_compute_advantage(wl)
    wl_box.caption(
        f"Measured blend at this workload: prompt processing **×{adv['prefill_ratio']:.2f}**, decode "
        f"**×{adv['decode_ratio']:.1f}**, blended **×{adv['blended']:.2f}** (Serving·Training tab).")

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
                   help="Current kernels: the software we have built and measured today. Optimized kernels: "
                        "the funded kernel program lands. That one is a target, not a measurement.")
    pf = {k: prefill_advantage(v, ctx) for k, v in scenarios.items()}
    kern = scenarios[st.session_state["scenario"]]
    h3 = TRAIN_SPEED_BY_SCENARIO[kern]
    scen_box.caption(f"◆ Two inputs differ by scenario: prompt-processing speed (×{pf[TODAY_LABEL]:.1f} current, "
                     f"×{pf[LANDED_LABEL]:.1f} optimized) and H3 training speed per token "
                     f"(×{TRAIN_SPEED_BY_SCENARIO['current']:.2f} measured, "
                     f"×{TRAIN_SPEED_BY_SCENARIO['mature']:.2f} target).")
    # The servers around the chips and the datacenters follow the fleet by default
    # (they scale down when no longer necessary); holding the datacenters
    # (contracted leases to FY33) is the sensitivity.
    st.session_state.setdefault("hold_dc", False)
    scen_box.checkbox("Hold datacenters (contracted leases to FY33)", key="hold_dc",
                      help="Unchecked: the servers around the chips and the datacenters (buildings, power and "
                           "cooling, network) shrink with the chip fleet. Checked: the datacenters stay at "
                           "today's spend because their leases run to FY33, and only chips, servers and power "
                           "fall. Affects the front tabs only (Summary, Value Bridge, Levers).")
    g["follow"] = BRIDGE_FOLLOW_HELD if st.session_state["hold_dc"] else BRIDGE_FOLLOW_20261001

    # H5 / H6 defaults track the context; a changed context resets them.
    d5, d6 = decode_levers(ctx)
    if st.session_state.get("_h56_ctx") != ctx:
        st.session_state["conv_per_gpu"], st.session_state["decode_speedup"] = float(d5), float(d6)
        st.session_state["_h56_ctx"] = ctx
    lev_box.caption(f"**H1** smaller model ×{_gain:.2f} and **H2** fewer training tokens ×{_gain:.2f}. H1 also "
                    f"applies to inference: a {_gain:.2f}× smaller model of the same quality costs about "
                    f"{_gain:.2f}× less per token. **H3** training speed per token ×{h3:.2f} "
                    f"({'measured at 2k and 8k, fitted beyond, over an assumed 8k/64k/256k training mix' if kern == 'current' else 'target for the fused kernels'}; "
                    f"details on the Levers tab).")
    g["mem_factor"] = gnum("H4 · Memory per conversation, ÷", "mem_factor", 1, 20000, 10, "%.0f",
                           help="Measured ×2,032 at 262k on our test model. For a frontier-size model with "
                                f"grouped-query attention we estimate about ÷{fleet_memory_lever(262144):.0f}.",
                           box=lev_box)
    g["conv_per_gpu"] = lev_box.number_input(
        "H5 · More conversations per GPU, ×", min_value=1.0, max_value=1024.0, step=1.0, key="conv_per_gpu",
        format="%.0f", help=f"Measured on one layer and quoted for {GEOM_TXT}. We hold "
                            f"{H5_CONVERSATIONS:.0f} live conversations per GPU (1 MB of state per layer each). "
                            f"The transformer holds the largest batch whose whole-model cache fits the card: "
                            f"one conversation at 262k. The default follows the conversation length.")
    g["decode_speedup"] = lev_box.number_input(
        "H6 · Decode step vs the transformer, per layer, ×", min_value=0.1, max_value=100.0, step=0.1, key="decode_speedup",
        format="%.2f", help=f"Measured on one layer: the transformer's decode step at its feasible batch divided "
                            f"by ours at {H5_CONVERSATIONS:.0f} conversations ({FFN_TXT}). Reads ×{H6_DECODE:.2f} "
                            f"at 262k for {GEOM_TXT}, with the transformer's step scaled to the cache it re-reads "
                            f"(×{deck_layer_levers(262144)['h6']:.2f} on the 24-layer model we measured). "
                            + ("Below 1 means the transformer's single conversation steps faster. The lever is H5." if H6_DECODE < 1 else ""))
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
    lev_box.caption(f"Tokens per GPU at equal model size (H5 × H6, prompts included) "
                    f"**×{eq[st.session_state['scenario']]:,.0f}**, times the **×{INFERENCE_SIZE_FACTOR:.2f}** "
                    f"smaller model (H1), gives the inference lever **×{g['flop_factor']:,.0f}**. Inference GPUs fall "
                    f"by **×{min(g['mem_factor'], g['flop_factor']):,.0f}**, the smaller of that and the memory lever."
                    + (" H5 and H6 at this length are modeled from the lengths we measured "
                       "(4k, 32k, 128k, 262k)." if _lvc["modelled"] else "")
                    + (" At this length one transformer conversation no longer fits a GPU. We ignore its sharding "
                       "cost, which understates us." if _lvc["sharded"] else ""))

    m = mkt_box
    g["gpu_cost"] = gnum("Fully-loaded $ per GPU", "gpu_cost", 5000, 150000, 1000, "%.0f", box=m)
    g["wall_power_kw"] = gnum("Wall power per GPU (kW)", "wall_power_kw", 0.3, 5.0, 0.1, "%.1f", box=m)
    g["elec_rate"] = gnum("Electricity ($/kWh)", "elec_rate", 0.02, 0.40, 0.01, "%.2f", box=m)
    g["discount_rate"] = gnum("Discount rate ★", "discount_rate", 0.02, 0.30, 0.01, "%.2f",
                              help="Changes only the capitalized value (about 1 ÷ rate).", box=m)
    m.caption("Capex and the datacenter, server and accelerator shares are edited on each company tab. "
              "Training shares are on the Levers tab.")
    with m.expander("Technical tabs only"):
        g["mem_share"] = gnum("Memory share of GPU cost", "mem_share", 0.0, 1.0, 0.05, "%.2f", box=st)
        auto_energy = st.checkbox("Derive the energy reduction from the cost reduction", value=True,
                                  help="Energy splits between memory and compute the way cost does. Uncheck to set it by hand.")
        if auto_energy:
            g["opex_reduction_override"] = None
        else:
            st.session_state.setdefault("opex_override", round(reduction_factor(g), 1))
            g["opex_reduction_override"] = st.number_input("Energy reduction override (×)", min_value=1.0,
                                                           max_value=200.0, step=1.0, key="opex_override",
                                                           format="%.1f")
        st.caption(f"Energy reduction **{x1(energy_reduction(g))}** "
                   f"({'derived' if auto_energy else 'set by hand'}).")
        g["cooling_overhead"] = gnum("Cooling/ops overhead", "cooling_overhead", 0.0, 1.0, 0.05, "%.2f", box=st)
        g["fleet_life_yr"] = gnum("Fleet life (yr)", "fleet_life_yr", 1, 10, 1, "%.0f", box=st)
        g["dc_scale"] = gnum("Datacenter scaling factor (older Totals engine only)", "dc_scale", 0.0, 1.0, 0.05,
                             "%.2f", box=st,
                             help="Technical Totals and company tabs only. 0 means only the accelerator silicon "
                                  "shrinks, 1 means the whole datacenter scales. The front tabs let servers and "
                                  "datacenters follow the fleet by default (checkbox above).")
        g["named_share_of_global"] = gnum("Named share of global AI capex", "named_share_of_global",
                                          0.3, 1.0, 0.05, "%.2f", box=st)
        g["spacex_mktcap"] = gnum("SpaceX market cap ($B)", "spacex_mktcap", 200, 4000, 10, "%.0f", box=st)
        st.metric("Older engine: cost-weighted reduction", x1(reduction_factor(g)),
                  help="Accelerators and power only, cost-weighted. Used on the technical Totals and company "
                       "tabs. The headline is the value bridge on the Summary tab.")
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
    st.caption("🟡 assumption · 🟢 disclosed data. Edit a value and the tables below recompute.")
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

    section(f"{name}: company data (green is disclosed, yellow is an estimate)")
    show_table(CB, [
        [("Total capex ($B)", ""), (n1(t26), "y"), (n1(t25), "g"), ("FY2026 from company guidance or actuals (see Sources); FY2025 as disclosed", "")],
        [("Datacenter share of capex ★", ""), (pct(i26), "y*"), (pct(i25), "y*"), (INFRA_SHARE_BASIS.get(name, "strips non-datacenter capex"), "")],
        [("Server (short-lived) share ★", ""), (pct(s26), "y*"), (pct(s25), "y*"), ("disclosed by the CFO", "")],
        [("Accelerator share within servers ★", ""), (pct(a26), "y*"), (pct(a25), "y*"), ("hardware teardowns, 67–80%", "")],
        [("Market cap ($B)", ""), (n0(c["mcap"]), "g"), ("", ""), ("approximate market data", "")],
    ], widths=W, wrap=True)

    section("How much of the capex buys AI chips")
    show_table(CB, [
        [("AI datacenter capex ($B)", ""), (n1(d26["ai_capex"]), "b"), (n1(d25["ai_capex"]), "b"), ("total × datacenter share", "")],
        [("Server bucket ($B)", ""), (n1(d26["ai_capex"] * s26), "b"), (n1(d25["ai_capex"] * s25), "b"), ("× server share", "")],
        [("Accelerator capex ($B)", ""), (n1(d26["accel"]), "b"), (n1(d25["accel"]), "b"), ("× accelerator share", "")],
        [("Accelerators as a share of total capex", ""), (pct(d26["accel_pct"]), "b"), (pct(d25["accel_pct"]), "b"), ("varies by company", "")],
    ], widths=W, wrap=True)

    section("Fleet and power, from the AI-chip capex")
    show_table(CB, [
        [("Fleet size (GPU-equivalents)", ""), (n0(fb26["fleet"]), "b"), (n0(fb25["fleet"]), "b"), ("accelerator capex ÷ $ per GPU", "")],
        [("Total wall power (MW)", ""), (n0(fb26["mw"]), "b"), (n0(fb25["mw"]), "b"), ("GPUs × kW", "")],
        [("Yearly power cost ($M)", ""), (n0(fb26["ann_m"]), "b"), (n0(fb25["ann_m"]), "b"), ("MWh × rate × (1 + overhead) × 365", "")],
        [("Power cost over the fleet's life ($B)", ""), (n1(fb26["life_b"]), "b"), (n1(fb25["life_b"]), "b"), ("× fleet life", "")],
    ], widths=W, wrap=True)

    section("What Helarctos changes (older engine: memory and compute priced separately)")
    show_table(CB, [
        [("AI capex still needed ($B)", ""), (n1(d26["ai_capex"] - d26["capex_avoided"]), "b"), (n1(d25["ai_capex"] - d25["capex_avoided"]), "b"), ("AI capex − avoided", "")],
        [("H4·H5 → Capex avoided per year ($B)", "h"), (n1(d26["capex_avoided"]), "b"), (n1(d25["capex_avoided"]), "b"), ("Helarctos enters here: chips × (1 − 1 ÷ the cost-weighted cut from H4 memory and H5 compute)", "")],
        [("H4·H5 → Yearly power saved ($M)", "h"), (n0(d26["opex_saved"] * 1000), "b"), (n0(d25["opex_saved"] * 1000), "b"), ("power shrinks with the fleet", "")],
        [("Yearly benefit ($B/yr)", ""), (n1(d26["spend_cut"]), "b"), (n1(d25["spend_cut"]), "b"), ("avoided capex + power saved", "")],
        [("Capitalized value ($B)", ""), (n0(d26["capitalized"]), "b"), (n0(d25["capitalized"]), "b"), ("benefit ÷ discount rate", "")],
        [("Share of market cap", ""), (f"{d26['capitalized'] / c['mcap']:.1%}" if c["mcap"] else "—", "b"), (f"{d25['capitalized'] / c['mcap']:.1%}" if c["mcap"] else "—", "b"), ("", "")],
    ], widths=W, wrap=True)

    section("AI economics on a cash basis: AI revenue − AI capex − AI power")
    show_table(CB, [
        [("AI revenue ($B)", ""), (n1(c["ai_rev"][1]), "y"), (n1(c["ai_rev"][0]), "y"), ("estimate (see Methodology)", "")],
        [("AI capex ($B)", ""), (n1(d26["ai_capex"]), "b"), (n1(d25["ai_capex"]), "b"), ("all AI datacenter capex: chips, servers, buildings, power, network", "")],
        [("AI power cost ($B)", ""), (n1(d26["ai_opex"]), "b"), (n1(d25["ai_opex"]), "b"), ("yearly electricity and operations", "")],
        [("Net AI today ($B)", ""), (n1(d26["net_now"]), "b"), (n1(d25["net_now"]), "b"), ("revenue − capex − power (cash burn)", "")],
        [("Spend cut with Helarctos ($B)", ""), (n1(d26["spend_cut"]), "b"), (n1(d25["spend_cut"]), "b"), ("accelerator capex avoided + power saved", "")],
        [("Net AI with Helarctos ($B)", ""), (n1(d26["net_arch"]), "b"), (n1(d25["net_arch"]), "b"), ("net today + spend cut", "")],
        [("AI spend cut, share", ""), (pct(d26["pct_cut"]), "b"), (pct(d25["pct_cut"]), "b"), ("spend cut ÷ total AI spend", "")],
    ], widths=W, wrap=True)

    if c.get("sources"):
        section("Sources")
        for label, url in c["sources"]:
            st.markdown(f"- [{label}]({url})")


# ---- totals tab ----------------------------------------------------------------
def econ_show(rows, tot, glob):
    cols = ["Company", "AI revenue", "AI capex", "AI power", "Net AI today", "Spend cut", "Net AI with Helarctos", "Share cut"]
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

    st.caption("**This is the older engine: accelerators and power only, cost-weighted.** It prices memory and "
               "compute separately and counts only the chips, plus whatever share of the datacenter the scaling "
               "factor lets through (default 0). The headline lives on the Summary tab, where the servers and "
               "datacenters sized by the GPUs also follow the fleet.")
    c1, c2, c4 = st.columns(3)
    c1.metric("Cost-weighted reduction (older engine)", x1(reduction_factor(g)))
    c2.metric(f"Net AI today ({len(comps)} companies, FY2026)", f"{bn(tot26['net_now'])}/yr")
    c4.metric("AI spend cut, FY2026", pct(tot26["pct_cut"]))
    n2c, n3c = st.columns(2)
    n2c.metric("Net AI with Helarctos (FY2026, older engine)", f"{bn(tot26['net_arch'])}/yr", delta=f"{bn(tot26['spend_cut'])} cut")
    n3c.metric("Net AI with Helarctos (FY2027, street estimate)", f"{bn(tot27['net_arch'])}/yr", delta=f"{bn(tot27['spend_cut'])} cut")
    st.caption("FY2027 uses street-estimate capex (the Morgan Stanley +57% path; Oracle's ~$70B is the only real "
               "FY2027 guidance) and AI revenue at a ×1.5 placeholder, so treat it as a forecast. FY2026 is the "
               "base year: the capex is set, and each company tab notes whether it is guidance or actual. FY2025 "
               "is shown below as last year.")
    verdict = ("AI turns **profitable** at these settings." if tot26["net_arch"] > 0
               else f"The FY2026 AI cash burn shrinks from **{md_usd(-tot26['net_now'])}B** to **{md_usd(-tot26['net_arch'])}B** per year "
                    f"(spend cut **{md_usd(tot26['spend_cut'])}B**, about {md_usd(tot26['capitalized'])}B capitalized).")
    st.markdown(f"On the older engine: {verdict}")
    st.caption(f"The {len(comps)} named firms are a floor. If they are about {g['named_share_of_global']:.0%} of "
               f"world AI capex, the global FY2026 spend cut is about {md_usd(glob26['spend_cut'])}B, or "
               f"~\\${glob26['capitalized'] / 1000:.1f}T capitalized (FY2025: ~\\${glob25['capitalized'] / 1000:.1f}T). "
               f"The rest of the world is other clouds, China, neoclouds, xAI and sovereign AI.")

    section("Net AI economics, FY2026 (cash basis)")
    econ_show(rows26, tot26, glob26)
    section("Net AI economics, FY2025 (last year, cash basis)")
    econ_show(rows25, tot25, glob25)

    section("Savings breakdown and capitalized value")
    disc = g["discount_rate"]
    o25 = sum(r["opex_saved"] for r in rows25); k25 = sum(r["capex_avoided"] for r in rows25)
    o26 = sum(r["opex_saved"] for r in rows26); k26 = sum(r["capex_avoided"] for r in rows26)
    show_table(["Item", "Power saved", "Capex avoided", "Total"], [
        [("FY2026 per year ($B)", ""), (n1(o26), "b"), (n1(k26), "b"), (n1(o26 + k26), "b")],
        [("FY2026 capitalized ($B)", ""), (n0(o26 / disc), "b"), (n0(k26 / disc), "b"), (n0((o26 + k26) / disc), "s")],
        [("FY2025 per year, last year ($B)", ""), (n1(o25), "b"), (n1(k25), "b"), (n1(o25 + k25), "b")],
        [("FY2025 capitalized ($B)", ""), (n0(o25 / disc), "b"), (n0(k25 / disc), "b"), (n0((o25 + k25) / disc), "s")],
        [("Reduction", ""), (pct(1 - 1 / energy_reduction(g)), "b"), (pct(1 - 1 / reduction_factor(g)), "b"), ("", "")],
    ])
    st.caption("Power saved recurs every year. Capex avoided is AI capex that no longer needs to be spent. "
               "This older engine counts accelerators plus the **datacenter scaling factor** share of the rest "
               "(sidebar, technical inputs; 0 is accelerators only, 1 is the whole datacenter). The Summary and "
               "Value Bridge tabs let the servers and datacenters sized by the GPUs follow the fleet by default, "
               "which is why they read higher.")


# ---- inputs tab ----------------------------------------------------------------
def inputs_tab(g):
    section("Global inputs (edit them in the sidebar). Teal rows are Helarctos levers")
    items = [
        ("H4 · Memory reduction factor", f"{g['mem_factor']:,.0f}×", "h", "A fixed-size state against a KV cache that grows with context. Measured on our full test model: 64 concurrent 262k conversations on one GPU in 1.62 GB, where the transformer fits one at 51.5 GB and a second runs out of memory. 25.4 MB of state per conversation against 197 KB of KV per token of context gives ×2,032 at 262k. For a frontier-size model with grouped-query attention we estimate about ÷208"),
        ("H1·H5·H6 · Inference compute lever", f"{g['flop_factor']:,.0f}×", "b", f"Tokens per GPU at equal model size, (1 + rq) ÷ (rq ÷ PF + 1 ÷ (H5 × H6)) = ×{g['flop_factor'] / INFERENCE_SIZE_FACTOR:,.0f}, times the ×{INFERENCE_SIZE_FACTOR:.2f} smaller model of the same quality (H1; a smaller model costs proportionally less per token in inference as in training). H5 = {g['conv_per_gpu']:.0f} conversations per GPU and H6 = ×{g['decode_speedup']:.2f} per decode step, both measured on one layer ({FFN_TXT}). PF is the scenario's prompt-processing speed against the transformer (×{prefill_advantage('current'):.1f} current, ×{prefill_advantage('mature'):.1f} optimized); rq is the transformer's prompt time divided by its decode time. Edit H5 and H6 in the sidebar"),
        ("Memory share of GPU cost", pct(g["mem_share"]), "y", "HBM plus most of the packaging, about 60/40 memory to compute from teardowns"),
        ("Energy reduction", x1(energy_reduction(g)), "b" if g.get("opex_reduction_override") is None else "y", "Derived from the cost-weighted reduction (energy splits between memory and compute the way cost does). Override in the sidebar"),
        ("Discount rate ★", pct(g["discount_rate"]), "y", "capitalized value = yearly benefit ÷ rate"),
        ("Fully-loaded $ per GPU", usd0(g["gpu_cost"]), "y", "GPU plus its share of the server, NVLink and networking"),
        ("Wall power per GPU (kW)", f"{g['wall_power_kw']:.1f}", "y", "GB200 NVL72: about 1.7–1.8 kW of IT load per GPU, times a PUE of about 1.3"),
        ("Electricity rate ($/kWh)", f"{g['elec_rate']:.2f}", "y", "datacenter wholesale"),
        ("Cooling and operations overhead", pct(g["cooling_overhead"]), "y", "non-power running cost as a fraction of electricity"),
        ("Fleet useful life (years)", f"{g['fleet_life_yr']:.0f}", "y", "AI-GPU depreciation life"),
        ("Datacenter scaling factor (older Totals engine only)", pct(g["dc_scale"]), "y", "Technical Totals and company tabs: 0 means accelerators only, 1 means the whole datacenter scales. The front tabs let servers and datacenters follow the fleet by default (sidebar checkbox)"),
        ("Named share of global AI capex", pct(g["named_share_of_global"]), "y", "the named firms' share of worldwide AI capex, used for the global row"),
        ("SpaceX market cap ($B)", n0(g["spacex_mktcap"]), "g", "market data, about $1.84T in August 2026 (IPO June 2026 at about $1.77T)"),
    ]
    show_table(["Input", "Value", "Kind", "Note"],
               [[(a, "h" if a.startswith("H") else ""), (b, c + ("*" if "★" in a else "")),
                 (f"Helarctos lever {a.split(' · ')[0]}" if a.startswith("H") else
                  ("Derived" if a.startswith("Energy") else "Market / modeling data"), "h" if a.startswith("H") else ""),
                 (d, "")] for a, b, c, d in items], wrap=True)

    section("The cost-weighted reduction, derived")
    cs = 1 - g["mem_share"]; mf = g["mem_share"] / g["mem_factor"]; cf = cs / g["flop_factor"]; res = mf + cf
    show_table(["Metric", "Value"], [
        [("Compute share of GPU cost", ""), (pct(cs), "b")],
        [("Memory cost left after the reduction", ""), (f"{mf:.2%}", "b")],
        [("Compute cost left after the reduction", ""), (f"{cf:.2%}", "b")],
        [("Cost left in total", ""), (pct(res), "b")],
        [("Cost-weighted reduction factor", ""), (x1(1 / res), "b")],
    ])
    _bind = "compute (the inference lever" if g["flop_factor"] < g["mem_factor"] else "memory (H4"
    st.caption(f"The component that shrinks least sets the floor. At these levers that is {_bind} "
               f"×{min(g['flop_factor'], g['mem_factor']):,.0f}; memory ×{g['mem_factor']:,.0f} against an inference "
               f"lever of ×{g['flop_factor']:,.0f}).")


# ---- sensitivity tab -----------------------------------------------------------
def sensitivity_tab(comps, g):
    section("SpaceX, FY2026: current kernels, optimized kernels and the live cost-weighted figure")
    sx = next(c for c in comps if c["name"] == "SpaceX")
    d = compute_company(sx, g, "fy26")
    accel, opx, disc, mcap = d["accel"], d["opex_saved"] * 1000, g["discount_rate"], g["spacex_mktcap"]
    # Tiers mirror the sidebar picker: the model's two inference levers
    # (equal-size tokens per GPU x the size factor) plus the live sidebar value.
    r_today = reduction_factor(dict(g, flop_factor=INFERENCE_LEVER_CURRENT))
    r_ceil = reduction_factor(dict(g, flop_factor=INFERENCE_LEVER_OPTIMIZED))
    tiers = [(f"Current kernels {r_today:.0f}×", r_today), (f"Optimized kernels {r_ceil:.0f}×", r_ceil),
             (f"Cost-weighted (live) {reduction_factor(g):.0f}×", reduction_factor(g))]
    cols = ["Metric"] + [t[0] for t in tiers]

    def row(label, fn, fmt):
        return [(label, "")] + [(fmt(fn(e)), "b") for _, e in tiers]

    show_table(cols, [
        row("Accelerator capex still needed ($B)", lambda e: accel / e, n1),
        row("Capex avoided per year ($B)", lambda e: accel - accel / e, n1),
        row("Power saved per year ($M)", lambda e: opx, n1),
        row("Yearly benefit ($B)", lambda e: (accel - accel / e) + opx / 1000, n1),
        row("Capitalized value ($B)", lambda e: ((accel - accel / e) + opx / 1000) / disc, n0),
        row("Share of market cap", lambda e: ((accel - accel / e) + opx / 1000) / disc / mcap, pct),
    ])
    st.caption("The base is SpaceX's accelerator capex, not its total capex. It matches the SpaceX tab with the "
               "datacenter scaling factor at 0.")

    section("How the inference lever moves with conversation length (per-layer decode measurement)")
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
            ("modeled from the measured lengths" if _l["modelled"] else "measured", ""),
        ])
    show_table(["Conversation length", "Transformer conversations that fit a GPU", "H5 conversations per GPU",
                "H6 decode step", "H5 × H6", "Tokens per GPU at equal size (current kernels)",
                "Inference lever (× size factor)", "H5 × H6 on the 24-layer model we measured", "Basis"], rc_rows,
               widths={"Basis": "medium"})
    st.caption(f"Quoted for {GEOM_TXT}. The depth is anchored on disclosed frontier models "
               f"({GEOM_ANCHOR}). The transformer runs the largest batch whose whole-model cache fits a "
               f"{GEOM['hbm_gb']:.0f} GB card at that length; from 32k up, its step is scaled to the cache it "
               f"re-reads (grouped-query, where the model we measured used multi-head attention). We run "
               f"{H5_CONVERSATIONS:.0f} conversations at {DECK_LAYER_CARD_20261001['own_state_mb_per_stream']:.1f} MB "
               f"per layer each, whatever the length. The per-layer step ratio does not change with depth, so it is "
               f"also the whole-model ratio; {FFN_TXT}. Lengths we did not measure are modeled from the measured ones and marked. "
               f"The last column is the 24-layer, d2048 model we measured, for reference.")

    section("Conversation-length sensitivity on the older estimate (superseded)")
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
    show_table(["Conversation length", "Transformer prompt share", "Our prompt share (current)",
                "◆ Tokens per GPU at equal size, current kernels (older estimate)",
                "◆ Tokens per GPU at equal size, optimized kernels (older estimate)", "Gap"], ctx_rows)
    st.caption("This table still runs on the older estimate (a 2.5 ms per-token step at 64 conversations), which "
               "we no longer use. It is kept to show how prompt processing's share of serving cost moves with "
               "length. The table above is the measured lever the model uses. Cost here is box wall-clock, not "
               "FLOPs. The transformer's decode is bound by KV bandwidth (measured: cost grows one-for-one with "
               "context) while its prompt processing runs near peak, so prompts are under 1% of its serving cost "
               "at every length. So the ×3.94 prompt speed-up we have (current kernels) and the full ×7.03 "
               "(optimized kernels) land within a few percent of each other.")


# ---- cost ladder tab -----------------------------------------------------------
def costladder_tab(g):
    section("Cost ladder: $ per H100-equivalent GPU-hour, at scale")
    lad = [
        ("Own custom silicon (TPU/Trainium)", "0.90", "1.40", "Cost of goods plus a modest Broadcom/Marvell margin, power and datacenter. No NVIDIA margin."),
        ("Buy and operate NVIDIA (at scale)", "1.50", "2.00", "NVIDIA's ~84% gross margin sits in the capex, plus power and datacenter."),
        ("Rent NVIDIA, neocloud or committed", "2.00", "3.50", "Adds the cloud provider's capex recovery and margin."),
        ("Rent NVIDIA, hyperscaler on-demand", "3.00", "7.00", "Adds utilization risk and a flexibility premium."),
    ]
    show_table(["How the GPU is bought", "$/hr low", "$/hr high", "What is in the price"],
               [[(m, ""), (lo, "g"), (hi, "g"), (note, "")] for m, lo, hi, note in lad],
               widths={"How the GPU is bought": "large", "What is in the price": "large"})

    section("Cross-check: total cost of an owned NVIDIA GPU, from the inputs")
    util = 0.85
    capx = g["gpu_cost"] / (g["fleet_life_yr"] * 8760 * util)
    powr = g["wall_power_kw"] * g["elec_rate"]; dc = 0.30
    show_table(["Metric", "$/hr", "Basis"], [
        [("Utilization", ""), (pct(util), "y"), ("assumed", "")],
        [("Capex per hour", ""), (f"{capx:.2f}", "b"), ("$ per GPU ÷ (life × 8,760 h × utilization)", "")],
        [("Power per hour", ""), (f"{powr:.2f}", "b"), ("wall kW × $/kWh", "")],
        [("Datacenter and staff per hour", ""), ("0.30", "y"), ("assumed", "")],
        [("Owned total per hour", ""), (f"{capx + powr + dc:.2f}", "b"), ("compare with 'Buy and operate NVIDIA'", "")],
    ], widths={"Basis": "large"})
    st.caption("Own silicon to bought NVIDIA is about 1.4–2× (NVIDIA's margin). Bought to rented is about 2–3.5× "
               "(the cloud's margin). Own silicon to rented is about 3–5×.")


# ---- serving & training tab ------------------------------------------------------
# The model's SERVING_TRAINING_ASSUMPTIONS rows are written for an internal reader. The app
# shows these plain-English versions instead (same facts, keyed by the model's row name);
# any row without an entry here falls through unchanged. Status colors use the model's status.
_ASSUMPTION_PLAIN = {
    "Compute lever (2026-10-01 re-base)": (
        "Inference compute lever",
        "×768. Tokens per GPU at equal model size are ×182 (H5: 256 conversations per GPU, H6: ×0.74 per decode "
        "step; both measured on one layer including the feed-forward block on both sides, ours uncompiled so the "
        "ratio understates us; depth, grouped-query cache and feasible batch modeled for a ~90-layer frontier "
        "model, where the 24-layer model we measured reads H5 × H6 = ×364; prompts added at the ×3.94 prompt speed "
        "we have). Times the ×4.22 smaller model of the same quality at trillion scale, from our quality fits. "
        "Optimized kernels: ×782 (prompts at the full ×7.03). The older estimate of 2.5 ms per token at 64 "
        "conversations (×1,553 with the size factor) is retired.",
        "per-layer decode measurement, 1 October 2026"),
    "Equal-quality parameter matching": (
        "Equal-quality parameter matching",
        "From the fits to our four-model ladder per family (47M–663M parameters): the fits cross at 392M, and above "
        "that bAttention matches the transformer fit's quality on 84.2% of the parameters at 1B, 55.2% at 10B, "
        "36.2% at 100B, 23.7% at 1T and 15.5% at 10T. This is a projection of two fits, not a measurement.",
        "quality fits over our measured model ladder"),
    "Single-GPU training step (AGAINST us)": (
        "Single-GPU training step (favors the transformer)",
        "On one GH200, one layer, forward and backward, bf16, no checkpointing on either side, against a modern "
        "transformer block (24 query and 4 KV heads, head size 256, RoPE, gated attention): a bAttention step costs "
        "×1.72 more at 2,048 tokens (101.7 vs 59.3 ms) and ×1.79 at 8,192. Forward ×1.82, backward ×2.35. It was "
        "×6.12 before the fused-kernel work. Training stays out of the serving claim, and the remaining gap is a "
        "funded kernel target rather than a structural loss.",
        "GH200 fused-kernel measurement, August 2026"),
    "Training PEAK MEMORY (AGAINST us)": (
        "Training peak memory (favors the transformer)",
        "×1.58 at 2,048 tokens (17,088 vs 10,824 MiB) and ×1.57 at 8,192, down from ×2.83 before the fused-kernel "
        "work. This is training memory, not the H4 serving-memory lever, which is unaffected. It is not an input "
        "to the cost model; it limits how many sequences fit on a GPU.",
        "GH200 fused-kernel measurement, August 2026"),
    "Kernel program (TARGET, not a result)": (
        "Kernel program (a target, not a result)",
        "The funded goal is an 8,192-token step at or below 83.08 ms, which beats the transformer and needs ×1.79 "
        "more, or 44% of the step. The named levers: removing recompute (about 12.8 ms), redesigning the backward "
        "pass and occupancy (about 3× headroom in the dominant fused kernel), and eliminating copies (24.8% of the "
        "step). Memory target ×0.88, at or below the transformer's. The optimized-kernels prompt speed is the "
        "×3.94 we have measured times this ×1.79 target.",
        "kernel program plan; no measurement behind the target half"),
    "Context scaling of the training gap": (
        "How the training gap changes with context",
        "At matched tokens per step, going from 2k to 8k (4× context) our step grows ×1.00 and the transformer's "
        "×1.32, so the ratio decays ×0.76 per 4×. The flat per-token model puts the crossover near 13,000 tokens "
        "(projected). Holding the older ×0.80 decay constant instead would say ×1.43 at 32k and a crossover near "
        "310,000 tokens. That is the conservative extrapolation, because the transformer's quadratic attention "
        "term makes its growth per 4× rise with context while ours stays flat.",
        "two measured ratio points plus the fitted cost form"),
    "Training CONTEXT MIX (reads the other way)": (
        "Training context mix (favors us)",
        "Quoting one context for all of training is the pessimistic corner. Cost-weighted over a training mix the "
        "training term flips from a drag to a contributor: ×0.58 with everything at 2k, ×2.22 at a modern "
        "8k/64k/256k mix, ×7.74 long-context heavy, ×12.08 with long reinforcement-learning rollouts. Our per-token "
        "cost is flat in context and the transformer's grows, so long-context tokens dominate its bill while "
        "staying a minority of tokens: 10% of tokens at 262k is 47% of its training cost. The step ratio is "
        "measured at 2k and 8k and modeled beyond; the mix shares are assumptions exposed as knobs, since we have "
        "no citation for any lab's recipe.",
        "fitted cost form and the training-mix table above"),
    "Long-context training memory (EXCLUDED, runs our way)": (
        "Long-context training memory (left out, favors us)",
        "At 262k and beyond the transformer also pays for sequence parallelism and activation memory that a "
        "fixed-size state avoids. Its KV cache grows at 156 KB per token of context per conversation, which forces "
        "sharding, ring or Ulysses attention, and their communication overhead. Our own measurement shows the "
        "asymmetry at small scale: our peak training memory grew only ×0.993 relative to the transformer's from 2k "
        "to 8k, so the ratio is flat to falling in context while the absolute gap the transformer must shard "
        "grows. Not in the cost model, which counts step time only, so leaving it out understates us. Pricing it "
        "needs a multi-GPU long-context training measurement we do not have.",
        "peak-memory ratios at 2k and 8k; KV bytes per token"),
    "$/GPU-hour": ("$ per GPU-hour", "$2.50 (H100 rent, neocloud or committed, mid-band; the ladder runs $2.00–3.50)",
                   "CostLadder tab, unchanged"),
    "Utilization": ("Utilization", "85%", "same value as the owned-GPU cross-check on the CostLadder tab"),
    "Context-length mix": (
        "Context-length mix",
        "The headline is quoted at 262k, a measured decode cell; the table sweeps 4k to 1M. The decode lever "
        "crosses 1 near 29,800 tokens, and below that the transformer serves more tokens per GPU-second than we "
        "do. We say so wherever it appears.",
        "serving cost curve and decode throughput ratio"),
    "Decode lever is MEMORY-CEILING, not latency (READ FIRST)": (
        "The decode lever is a memory ceiling, not latency (read this first)",
        "Per generated token with one conversation, the transformer is ahead of us at 64k context: 4.92 vs 5.74 "
        "GPU-busy per token. The lever is that it cannot hold many long-context conversations. It re-reads its "
        "whole KV cache for every token it emits, so its aggregate throughput falls one-for-one with context while "
        "ours is flat. Measured: one conversation at 262k (a second runs out of memory, an exact ceiling) and 8 at "
        "32k (16 run out, so that ceiling sits somewhere in 8–15 and the 32k cell is ×0.97–1.14, near parity), "
        "against 64 of ours in 1.62 GB.",
        "full-model decode measurement, August 2026"),
    "Transformer-stack maturity": (
        "Transformer stack maturity",
        "Not granted by default: the headline runs on the measured cells, in which neither family's decode is "
        "optimized (ours ran an unoptimized per-step path at 9.5% GPU-busy while the transformer sat at 95.6%). "
        "Granting the transformer paged attention and KV quantization (8× compression) divides our decode lever "
        "by 8 and puts the transformer ahead at 64k. We show that row.",
        "model setting for KV compression"),
    "bAttention serving throughput": (
        "bAttention serving throughput",
        "562 tokens/s per GPU (4,497 per box): 64 concurrent 262,144-token conversations on one GPU, flat from 32k "
        "to 262k (−4.0%). The 64-conversation cell is a grid cap, not a ceiling: 1.62 GB of state on a 96 GB card "
        "at 9.5% GPU-busy, so it is a lower bound. Replaces a 349,880 tokens/s per box figure measured on a "
        "proxy that counted only part of the state.",
        "full-model decode measurement, August 2026"),
    "bAttention per-stream state": (
        "bAttention state per conversation",
        "25.4 MB per conversation for the full cell (24.2 MiB measured), flat in context. The 0.15 MB figure "
        "quoted earlier came from a proxy that counted only part of the state and is retired.",
        "full-model decode measurement, August 2026"),
    "Transformer KV wall": (
        "Transformer KV-cache wall",
        "197 KB per token of context per conversation as reserved end to end (51.5 GB for one 262,144-token "
        "conversation; the linear KV fit alone is 156 KB per token, the rest is workspace). A second conversation "
        "runs a 96 GB card out of memory at 262k, and 16 do at 32,768 tokens. Its retired 88.5 per-token decode "
        "cost was inflated by a harness artifact; the traced figure is 14.92 GPU-busy at 262k.",
        "full-model decode measurement, August 2026"),
    "Training throughput at 8 GPUs": (
        "Training throughput at 8 GPUs",
        "×5.15 (bAttention, forward and backward) vs ×3.70 (the transformer's best forward-only Ulysses run) at a "
        "matched 16 sequences in flight, so ×1.39; ×6.27 at 256 in flight, so ×1.70.",
        "8-GPU scaling measurement, August 2026"),
    "Memory walls (component scope)": (
        "Memory walls (component scope)",
        "A 64k-token sequence: 30.9 GB fits one GPU, where the transformer ran out at 78.4 GB attempted. Pipeline "
        "stage 9.05 GB flat vs 43.3 GB for GPipe.",
        "8-GPU scaling measurement, August 2026"),
    "Compute lever (flop_factor)": (
        "Prompt-processing lever",
        "Cross-family prompt processing, bf16, one H100, parameter counts exactly matched, one layer at batch 16: "
        "transformer ÷ bAttention 4.02 at d512 and 1M context, 3.83 at d1024 and 1M, 2.01 at d2048 and 262k. "
        "Short context favors the transformer, and we say so. Measured on our older kernels and not re-run since "
        "the fused-kernel work, so the current-kernels lever is conservative; the ×3.94 we have since measured sits "
        "in the optimized-kernels prompt speed, not here.",
        "parameter-matched prompt sweep, July 2026"),
    "Compute, fp32 lane (scope reference)": (
        "Prompt processing, fp32 (scope reference only)",
        "The d2048, 24-layer fp32 sweep gives 7.91 at 262k at whole-model scope. No fp32 fused-attention kernel "
        "exists, so that lane runs memory-efficient attention on the transformer and our side is fp16 internally, "
        "with parameters 16.7% apart. The full model in bf16 has not been run.",
        "fp32 speed sweep, August 2026"),
    "16-bit IO (training)": (
        "16-bit input and output (training)",
        "On our own architecture at d1536 on a GH200: fp16 is ×1.185 the step time with −17.5% peak memory, bf16 "
        "×1.171 and −20.6%; quality moved +0.001 to +0.004 on the smaller test vehicle. Reported on its own; not an "
        "input to the compute lever.",
        "16-bit gate measurement, August 2026"),
    "Quality parity at 70B": (
        "Quality parity at 70B",
        "The transformer needs ×4.1 the parameters [2.2–7.7] or ×1.7 the tokens [1.33–2.10, assuming β = 0.28]; "
        "the fits cross near 321M parameters, the largest model we measured.",
        "quality fits over our measured model ladder"),
    "Scope": (
        "Scope",
        "The systems numbers compare components (our d1536 forward-and-backward block against a transformer d2048 "
        "forward-only layer). Every speed-up is each family against its own single-GPU baseline.",
        "scope notes in our measurement archive"),
}


def serving_training_tab(g):
    st.caption("Our multi-GPU measurements (2, 4 and 8 H100s), turned into dollars on the $ per GPU-hour ladder "
               "from the CostLadder tab. Every number below traces to our measurement archive or to a row in the "
               "assumptions table at the bottom. **Scope**: the systems numbers compare components (our d1536 "
               "forward-and-backward block against a transformer d2048 forward-only layer). Every speed-up is "
               "each family against its **own** single-GPU baseline, so the scope cancels inside each ratio.")

    m1, m2, m3, m4 = st.columns(4)
    r262 = serving_economics(262144)
    m1.metric("Serving cost ratio at 262k context", f"{r262['cost_ratio']:.1f}×",
              help="Measured aggregate decode throughput per GPU, each family at its own concurrency limit "
                   "(full-model test). If the transformer is granted an idealized stack with its KV cache "
                   f"compressed 8×: {serving_economics(262144, s={'tf_kv_compression': 8.0})['cost_ratio']:.1f}×.")
    m2.metric("Concurrent 262k conversations per GPU", f"{MEASURED['serve_streams_per_gpu']} vs 1",
              help="Measured on the full test model: 64 bAttention conversations in 1.62 GB of state on one GPU. "
                   "The transformer fits one 262k conversation at 51.5 GB and a second runs out of memory. "
                   f"Aggregate throughput ×8.8. The newer per-layer measurement replaces this as the source of H5 "
                   f"({H5_CONVERSATIONS:.0f} conversations vs 1) and H6 (×{H6_DECODE:.2f} per step).")
    m3.metric("Training throughput on 8 GPUs", f"×{training_throughput_ratio():.2f}",
              delta=f"×{training_throughput_ratio(deep=True):.2f} with 256 sequences in flight",
              help="Measured at matched load (16 sequences in flight for both families): ×5.15 forward-and-backward "
                   "against ×3.70 for the transformer's best forward-only Ulysses run.")
    m4.metric("Quality parity at 70B (projection)", f"×{MEASURED['parity_70B_param_multiple']:.1f} params",
              delta=f"or ×{MEASURED['parity_70B_token_multiple']:.2f} tokens (β = 0.28)", delta_color="off",
              help="Projected from the fits to our measured model ladder; 95% interval ×2.2–×7.7 on parameters, "
                   "×1.33–×2.10 on tokens. Not a measurement.")

    section("Serving at long context: $ per million generated tokens, one 8-GPU H100 box")
    rows = []
    for r in serving_cost_curve():
        note = ("bAttention rate extrapolated past 262k (measured flat from 4k to 262k)" if r["battn_extrapolated"]
                else ("the measured decode point" if r["ctx"] == 262144 else ""))
        rows.append([
            (f"{r['ctx']:,}", ""),
            (f"${r['battn_usd_per_mtok']:.4f}", "g" if not r["battn_extrapolated"] else "y"),
            (f"${r['tf_usd_per_mtok']:.2f}", "b"),
            (f"{r['tf_streams_per_gpu']}", "b"),
            (f"×{r['cost_ratio']:.1f}", "s"),
            (note, ""),
        ])
    show_table(["Context (tokens)", "bAttention $ per million tokens", "Transformer $ per million tokens (mature stack)",
                "Transformer conversations per GPU", "Cost ratio", "Note"], rows,
               widths={"Note": "large"})
    st.caption("This is a memory-ceiling result, not a per-token one. Our decode rate is flat in context "
               "(562 tokens/s per GPU with 64 concurrent 262k conversations in 1.62 GB). The transformer's "
               "aggregate rate falls one-for-one with context once the card is full of KV cache (197 KB per "
               "token of context per conversation). **The transformer is ahead below about 30k context.** "
               "We are ×2.2 ahead at 64k and ×8.8 at 262k. The full derivation is on the Methodology tab.")

    section("Training at scale: same cluster, more steps per second")
    show_table(["Metric", "bAttention", "Transformer", "Ratio", "Status"], [
        [("8-GPU speed-up at matched load (16 sequences in flight)", ""), ("×5.15 (forward and backward)", "g"), ("×3.70 (forward only, best Ulysses run)", "g"), (f"×{training_throughput_ratio():.2f}", "s"), ("measured", "g")],
        [("8-GPU speed-up at deep load (256 in flight)", ""), ("×6.27, the pipeline keeps filling", "g"), ("saturated at 16 in flight", "g"), (f"×{training_throughput_ratio(deep=True):.2f}", "s"), ("measured", "g")],
        [("GPU-hours for the same training work", ""), ("−28% (matched) to −41% (deep)", "b"), ("baseline", ""), ("", ""), ("derived", "b")],
        [("64k-token sequence on one 80 GB GPU", ""), ("30.9 GB, fits", "g"), ("out of memory (78.4 GB attempted)", "g"), ("", ""), ("measured", "g")],
        [("Pipeline peak per GPU (flat in load)", ""), ("9.05 GB", "g"), ("43.3 GB (GPipe stage)", "g"), ("×4.8", "b"), ("measured", "g")],
        [("Parameters for equal quality at 70B", ""), ("1×", ""), ("×4.1 [2.2–7.7]", "y"), ("", ""), ("projection", "y")],
        [("Tokens for equal quality at 70B (β = 0.28)", ""), ("1×", ""), ("×1.7 [1.33–2.10]", "y"), ("", ""), ("projection", "y")],
    ], widths={"Metric": "large"})
    st.caption("Speed-ups are each family against its own single-GPU baseline, so the scope cancels. The memory "
               "rows compare components at different widths and are not whole-model claims. The parity rows are "
               "projections from the fits to our measured quality ladder (the fits cross near 321M parameters, "
               "about the largest model we measured; the later fit behind the compute lever crosses at 392M). "
               "β = 0.28 is the Chinchilla assumption.")

    section("Estimated training cost at equal quality")
    st.caption("Training compute is about 6 × parameters × tokens. At equal quality we need parameters ÷ s, "
               "where **s** is the equal-quality parameter ratio from our fits (the same fit as the compute "
               "lever). Two token regimes follow, one power of s apart. **Compute-optimal** (Chinchilla, tokens "
               "proportional to parameters): the smaller model also trains on fewer tokens, so the cost ratio is "
               "s² ÷ r. **Fixed token budget** (data-constrained, same tokens on both sides): the cost ratio is "
               "s ÷ r. **r** is the step-time ratio at matched parameters: measured at 2,048 tokens (×1.72) and "
               "8,192 (×1.79, at batch 4, with a possible batch artifact), and modeled beyond from the per-token "
               "cost form below. Above 1× we are cheaper to train to the same quality.")
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
        "\\* 256k is modeled. r comes from a per-token cost form, not a constant decay: the transformer pays "
        "a fixed cost plus an attention cost that grows with context per token (its fused attention kernel is "
        "linear per token, quadratic over the sequence), while our state is fixed-size, so our per-token cost "
        "is flat. We measured that flatness: our backward pass costs 3.27–3.31 µs per token "
        "across an 8× range of context (spread 1.4%, 25 runs at d4096) while the transformer's rises 66%, and "
        "our decode moves 0.83% over a 128× range. The two cost terms are fitted to the 2k and 8k cells. The "
        f"fit gives r = {training_step_ratio(32768):.2f} at 32,768 tokens and puts the crossover near "
        f"{training_context_crossover():,.0f} tokens (projected; measured separately at 21–24k with batch 4 and "
        f"about 16k with batch 16 on an H100 at d4096). One caveat: the transformer's cost is fitted to both of "
        f"its measured cells, but our cost is held at its 2k value, with our measured 8k step treated as a "
        f"batch-4 artifact. The fit therefore reads ×{training_step_ratio(8192):.2f} at 8k where the measured "
        "8k cell read ×1.79. "
        "The older constant-decay curve is kept as the conservative bound and is far too pessimistic past 32k "
        f"(it says ×{training_step_ratio(262144, mode='constant_decay'):.2f} at 256k where the cost form says "
        f"×{training_step_ratio(262144):.2f}), because holding the growth ratio constant assumes the "
        "transformer's attention term stops growing. The quality fits cross at 392M parameters, so below that "
        "we need more parameters, not fewer, and every cell past about 1B extrapolates beyond the 47M–663M "
        "models we measured. Peak training memory (×1.58 measured, ×0.88 targeted) is deliberately left out: "
        "it limits how many sequences fit on a GPU, not the FLOPs.")
    _tc_png = Path(__file__).resolve().parent / "paper/figures/vc_memo_training_cost_scaling.png"
    if _tc_png.exists():
        st.image(str(_tc_png), width="stretch")

    section("Training is a mix of context lengths, not a single one")
    st.caption(
        "Quoting r at one context assumes all training happens there. Real training is a curriculum: a "
        "short-context bulk phase, then long-context extension phases, then long-rollout reinforcement "
        "learning. The right training term is the ratio of costs added up over the mix, not the average of "
        "the per-context ratios. The difference matters because our per-token cost is flat in context and the "
        "transformer's grows: a small share of long-context tokens carries a large share of the transformer's "
        "bill. The token shares below are assumptions we expose as knobs. We have no citation for any lab's "
        "recipe and have not invented one. The r curve underneath them is measured and modeled as described "
        "above.")
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
    show_table(["Training mix", "Training advantage (transformer ÷ ours)", "Effective r", "Mix: token share → transformer cost share", "Status"],
               _cur_rows, widths={"Mix: token share → transformer cost share": "large", "Training mix": "medium"})
    st.caption(
        f"Read the middle columns together. Effective r drops below 1 at every modern mix, so training stops "
        f"being a loss and starts contributing. Raising the dollar headline is a higher bar: the blended lever "
        f"is a harmonic, cost-weighted average, so training only pushes the headline up once it beats the serving "
        f"advantage of ×{training_helps_headline_threshold():.2f}, not just ×1. Between the two, training is "
        f"profitable but still dilutes that one number. The right-hand column shows why: in the modern mix, "
        f"10% of tokens sit at 256k and carry "
        f"{training_advantage_mix('modern_standard')['rungs'][-1]['tf_cost_share'] * 100:.0f}% of the "
        f"transformer's training cost.")

    section("What the older engine's FY2026 headline does when training enters the blend")
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
    show_table(["Training mix", "training 0% of spend", "training 20%", "training 40%", "Direction"], _hl_rows,
               widths={"Training mix": "medium"})
    st.caption(
        "Every cell is the FY2026 spend cut across the six named firms on the older engine, recomputed end to "
        "end with the blended lever. Assuming all training at 2k was the only mix that pulled the headline "
        "down materially. At a modern mix the headline is flat, and at long-context-heavy mixes it rises. The "
        "whole range is narrow: once the chip bill is mostly gone, the dollar headline is not very sensitive "
        "to this lever in either direction.")
    st.caption(
        "**Left out, and it favors us.** At 262k and beyond the transformer also pays for sequence "
        f"parallelism and activation memory that a fixed-size state avoids. Its KV cache grows at "
        f"{KV_MB_PER_TOKEN_PER_STREAM * 1024:.0f} KB per token of context per conversation, which is what "
        f"forces sharding, ring or Ulysses attention, and the communication that comes with them. Our own "
        f"measurement shows the asymmetry at small scale: from 2k to 8k the peak-memory ratio went from "
        f"×{KERNEL_CAMPAIGN_20260824['mem_ratio_2k']:.3f} to ×{KERNEL_CAMPAIGN_20260824['mem_ratio_8k']:.3f}, "
        "flat to falling in context, while the absolute footprint the transformer must shard keeps growing. "
        "None of this is in the cost model above, which counts step time only, so leaving it out understates "
        "us. Pricing it needs a multi-GPU long-context training measurement we do not have yet.")

    section("How the workload mix sets the compute lever")
    st.caption("Training, prompt processing and decode point in different directions at today's kernel "
               "maturity, so the lever is blended over the workload rather than taken from one phase. Cost is "
               "added up per generated token: the input tokens through prompt processing plus one token "
               "through decode, using measured throughput on both sides. The sidebar controls move this table.")
    _wl_rows = []
    for _r in (1.0, 10.0, 100.0):
        for _c in (8192, 65536, 262144):
            _a = workload_compute_advantage({"in_out_ratio": _r, "context_tokens": _c,
                                             "train_share": 0.0})
            _tone = "s" if _a["blended"] >= 1.0 else "b"
            _note = "" if _a["blended"] >= 1.0 else "transformer ahead"
            if (_r, _c) == (WORKLOAD["in_out_ratio"], WORKLOAD["context_tokens"]):
                _note = "model default (code and agent use)"
            _wl_rows.append([
                (f"{_r:.0f}:1", ""), (f"{_c:,}", ""),
                (f"×{_a['prefill_ratio']:.2f}", "b"),
                (f"×{_a['decode_ratio']:.1f}", "g"),
                (f"×{_a['blended']:.2f}", _tone),
                (f"{_a['tf_prefill_share'] * 100:.0f}%", ""),
                (_note, ""),
            ])
    show_table(["Input:output", "Average context", "Prompt processing", "Decode", "Blended",
                "Prompt share of transformer cost", "Note"], _wl_rows,
               widths={"Note": "medium"})
    st.caption("Decode dominates the transformer's serving cost in every row, which is why the blend tracks it. "
               "Our decode cost is flat in context and the transformer's grows one-for-one, so the decode "
               "advantage is linear in the average context and mixing contexts costs nothing in the average. "
               "**Training is excluded**, but the gap is much smaller than it was. On one GH200, one layer, "
               "forward and backward, bf16, no checkpointing on either side, against a modern transformer block: "
               f"the transformer is ×{KERNEL_CAMPAIGN_20260824['step_ratio_2k']:.2f} faster per step at 2,048 "
               f"tokens and ×{KERNEL_CAMPAIGN_20260824['step_ratio_8k']:.2f} at 8,192. The flat per-token model "
               "puts the crossover near 13k tokens (projected; measured at 21–24k with batch 4 on d4096). A "
               "training share above about 0.30 now takes the blend below 1; the threshold was about 0.06 "
               "before the kernel work.")

    section("Prompt processing, 16-bit, parameter-matched")
    st.caption("Measured where both families run their production 16-bit attention kernels, the lane the "
               "competition serves in, with parameter counts exactly matched: one layer, batch 16, one H100. "
               "**These are our older kernels.** This lane has not been re-run since the fused-kernel work "
               "sped our side up ×3.94, so the lever below understates us by construction. The remaining "
               "target is carried in the Optimized-kernels scenario, not here.")
    prow = []
    for T, tf_ms, ba_ms in PREFILL_BF16_D2048_SWEEP:
        ratio = tf_ms / ba_ms
        note, tone = ("transformer ahead", "b") if ratio < 1.0 else ("", "g")
        if T == 262144:
            note, tone = "width and context match the fp32 reference on the Evidence tab", "s"
        prow.append([(f"{T:,}", ""), (f"{tf_ms:,.1f}", "b"), (f"{ba_ms:,.1f}", "g"),
                     (f"×{ratio:.2f}", tone), (note, "")])
    show_table(["Context (tokens)", "Transformer ms", "bAttention ms", "Transformer ÷ bAttention", "Note"],
               prow, widths={"Note": "medium"})
    st.caption(f"Shown at d2048. The lever itself is taken at **d512 and 1M context, ×"
               f"{MEASURED['prefill_bf16_tf_battn_d512_1m']:.2f}** "
               f"(d1024 at 1M: ×{MEASURED['prefill_bf16_tf_battn_d1024_1m']:.2f}; "
               f"d2048 at 262k: ×{MEASURED['prefill_bf16_tf_battn_d2048_262k']:.2f}). The ratio rises with "
               f"context and falls with width. Below the crossover the transformer is faster, and the model "
               f"says so.")

    section("Assumptions, one row each")
    show_table(["Assumption", "What it is", "Status", "Where it comes from"],
               [[(pa, ""), (pb, ""),
                 (c, "g" if c == "MEASURED" else ("y" if c in ("PROJECTION", "ASSUMPTION") else "b")),
                 (pd_, "")]
                for a, b, c, d in SERVING_TRAINING_ASSUMPTIONS
                for pa, pb, pd_ in [_ASSUMPTION_PLAIN.get(a, (a, b, d))]],
               widths={"Assumption": "medium", "What it is": "large", "Where it comes from": "large"})
    st.caption("Rates are unchanged from the CostLadder tab ($2.00–3.50 per hour to rent, $2.50 in the middle). "
               "Production transformer stacks (paged attention, KV quantization) push their memory wall out. "
               "The mature column already grants that, which is why the headline uses it as the floor.")


# ---- evidence tab --------------------------------------------------------------
def evidence_tab(g):
    section("The measurements behind the two levers")
    show_table(["Lever", "Value", "Measurement", "Device and date", "Scope"], [
        [("Prompt-processing speed (the prompt term of the compute lever)", ""), ("×4.02 at d512, 1M context", "s"),
         ("parameter-matched prompt sweep, d512", ""),
         ("one H100 80 GB, July 2026", ""),
         ("both families, bf16 with the transformer's fused attention kernel, parameters exactly matched, one layer at batch 16", "g")],
        [("Prompt processing, other widths", ""), ("×3.83 at d1024 and 1M; ×2.01 at d2048 and 262k", "b"),
         ("parameter-matched prompt sweep, d1024 and d2048", ""),
         ("one H100 80 GB, July 2026", ""),
         ("the ratio rises with context and falls with width", "g")],
        [("Prompt processing, fp32", ""), ("×7.91 at d2048, 262k", "y"),
         ("fp32 speed sweep", ""),
         ("one H100 80 GB, August 2026", ""),
         ("full 24-layer model; no fp32 fused attention kernel exists, our side is fp16 internally, parameters 16.7% apart", "y")],
        [("Memory per conversation (H4)", ""), ("÷2,032 (measured on our test model)", "s"),
         ("multi-GPU decode measurement at 262k", ""),
         ("8-GPU H100 box, August 2026", ""),
         ("serving state: 64 concurrent 262k conversations on one GPU in 1.62 GB, where the transformer fits one and a second runs out of memory; 25.4 MB of full-model state per conversation against 51.5 GB of KV cache. Peak memory during prompt processing runs the other way, about 2.5× against us", "g")],
        [("16-bit input and output (training)", ""), ("×1.185 step time, −17.5% memory", "y"),
         ("16-bit gate measurement", ""),
         ("one GH200 480 GB, August 2026", ""),
         ("our own architecture at d1536; training step time, reported on its own", "y")],
    ], widths={"Measurement": "large", "Scope": "large"})
    st.caption("The compute lever compares the two families in bf16 at block scope. The memory lever compares "
               "them at serving scope. The full model in bf16 is the one cell not yet run.")

    section("What a GPU costs to make (teardown = true resource cost)")
    show_table(["Component", "$ cost", "Share of cost", "Note"], [
        [("H100: HBM3 memory (80 GB)", ""), ("1,350", ""), ("41%", ""), ("memory", "")],
        [("H100: CoWoS packaging", ""), ("750", ""), ("23%", ""), ("mostly memory (the interposer carries the HBM)", "")],
        [("H100: test and assembly", ""), ("920", ""), ("28%", ""), ("shared", "")],
        [("H100: logic die (compute)", ""), ("300", ""), ("9%", ""), ("compute, the cheapest part", "")],
        [("H100 total cost of goods", ""), ("3,320", ""), ("100%", ""), ("sells for about $28k, about 88% margin", "")],
        [("B200 total cost of goods", ""), ("6,400", ""), ("HBM 45%", ""), ("memory outweighs logic; sells for about $40k, about 84% margin", "")],
    ], widths={"Note": "large"})

    section("Accelerator share of a server's cost")
    show_table(["Server", "Accelerator share", "Note"], [
        [("8× H100 server (J.P. Morgan)", ""), ("83%", ""), ("accelerators are $200k of $240k", "")],
        [("8× A100 server (J.P. Morgan)", ""), ("71%", ""), ("a GB200 rack is about 76–80% (SemiAnalysis)", "")],
    ], widths={"Note": "large"})

    section("Filing data (FY2025 actuals)")
    show_table(["Company", "Total capex", "Server life", "Note"], [
        [("Microsoft (including leases)", ""), ("~$88B", "g"), ("2–6 years", ""), ("'half' to 'two-thirds' short-lived, per the CFO", "")],
        [("Alphabet", ""), ("$91.4B", "g"), ("6 years", ""), ("60% servers, 40% datacenters, per the CFO", "")],
        [("Amazon (cash capex)", ""), ("$128.3B", "g"), ("5 years (shortened)", ""), ("AWS is 67.8% of net property and equipment additions", "")],
        [("Meta (including finance leases)", ""), ("$72.2B", "g"), ("5.5 years", ""), ("servers are the 'largest portion', per the CFO", "")],
        [("Oracle", ""), ("$21.2B", "g"), ("—", ""), ("nearly all OCI and GPU datacenters; FY2026 guided at about $50B", "")],
        [("SpaceX (S-1 AI capex)", ""), ("$12.7B", "g"), ("—", ""), ("nearly all accelerators (the greenfield Colossus build)", "")],
    ], widths={"Note": "large"})

    section("Cost-weighted reduction against the memory share (live)")
    show_table(["Case", "Memory share", "Reduction"],
               [[(f"memory share {int(w * 100)}%", ""), (pct(w), "y"),
                 (x1(1 / (w / g["mem_factor"] + (1 - w) / g["flop_factor"])), "b")]
                for w in (0.45, 0.50, 0.60, 0.70, 0.82)])
    st.caption("The compute term dominates, so the reduction stays far below the memory factor.")


# ---- methodology tab -----------------------------------------------------------
def methodology_tab(g):
    section("Every input, and where it comes from")
    ek = ("derived", "b") if g.get("opex_reduction_override") is None else ("set by hand", "y")
    _kc = KERNEL_CAMPAIGN_20260824
    rows = [
        [("Memory reduction (×)", ""), (f"{g['mem_factor']:.0f}×", "y"), ("assumption", "y"), ("H4. A fixed-size state against a KV cache that grows with context. Measured on the full test model: 64 concurrent 262k conversations on one GPU in 1.62 GB, where the transformer fits one at 51.5 GB and a second runs out of memory. 25.4 MB of state per conversation against 197 KB of KV per token gives ×2,032 at 262k. A frontier-size model with grouped-query attention would be about ×208 (estimate)", "")],
        [("Inference compute lever (×)", ""), (f"{g['flop_factor']:,.0f}×", "y"), ("measured × projection", "y"), (f"◆ Set by the scenario (Current kernels by default). Tokens per GPU at equal model size (×{g['flop_factor'] / INFERENCE_SIZE_FACTOR:,.0f}: H5 = {g['conv_per_gpu']:.0f} conversations per GPU times H6 = ×{g['decode_speedup']:.2f} per decode step, both measured on one layer, the transformer at the largest batch whose whole-model cache fits the card at that length and us at {H5_CONVERSATIONS:.0f} conversations; {FFN_TXT}; prompt processing added at the ×{KERNEL_SPEEDUP_REALIZED_20260824:.2f} prompt speed-up we have) times the ×{INFERENCE_SIZE_FACTOR:.2f} smaller model of the same quality at 1T from our quality fits. A smaller model of the same quality costs proportionally less per token in inference as in training. The fits (four models per family) cross at 392M and give the transformer's quality on 23.7% of the parameters at 1T. That is a projection of two fits, not a measurement. Optimized kernels: prompts at the full ×{CEILING_PREFILL_SPEEDUP:.2f} (×{KERNEL_SPEEDUP_REALIZED_20260824:.2f} measured times a ×{KERNEL_SPEEDUP_REMAINING_TARGET:.2f} target), giving ×{INFERENCE_LEVER_OPTIMIZED:,.0f}", "")],
        [("Training step cost, favors the transformer", ""), (f"×{_kc['step_ratio_2k']:.2f}", "y"), ("measured", "b"), (f"One GH200, one layer, forward and backward, bf16, no checkpointing on either side, against a modern transformer block (24 query and 4 KV heads, head size 256, RoPE, gated attention): a bAttention training step costs ×{_kc['step_ratio_2k']:.2f} more at 2,048 tokens ({_kc['battn_ms_2k']:.1f} vs {_kc['tf_ms_2k']:.1f} ms per step) and ×{_kc['step_ratio_8k']:.2f} at 8,192 ({_kc['battn_ms_8k']:.1f} vs {_kc['tf_ms_8k']:.1f}); forward ×{_kc['fwd_ratio']:.2f}, backward ×{_kc['bwd_ratio']:.2f}. At equal quality the 2,048-token figure falls to ×{_kc['step_ratio_2k'] * param_matching_fraction(DECK_DEPLOYMENT_SCALE):.2f}. Training stays out of the serving claim", "")],
        [("Training peak memory, favors the transformer", ""), (f"×{_kc['mem_ratio_2k']:.2f}", "y"), ("measured", "b"), (f"Same setup: ×{_kc['mem_ratio_2k']:.3f} at 2,048 tokens ({_kc['battn_peak_mib_2k']:,.1f} vs {_kc['tf_peak_mib_2k']:,.1f} MiB) and ×{_kc['mem_ratio_8k']:.3f} at 8,192. This is training memory, not the serving-memory lever above, which is unaffected. It is not an input to the cost model; it limits how many sequences fit on a GPU", "")],
        [("Kernel program (a target, not a result)", ""), (f"≤{_kc['target_8k_win_gate_ms']:.1f} ms", "y"), ("target", "y"), (f"The funded goal is an 8,192-token step at or below {_kc['target_8k_win_gate_ms']:.2f} ms, which beats the transformer and needs ×{_kc['target_8k_gap_remaining']:.2f} more, or 44% of the step. Named levers: removing recompute (about 12.8 ms), redesigning the backward pass and occupancy (about 3× headroom in the dominant fused kernel), and eliminating copies (24.8% of the step). Memory target ×{_kc['target_mem_ratio']:.2f}, at or below the transformer's. There is no measurement behind any of this. It is the program plan, and it is the factor inside the Optimized-kernels scenario", "")],
        [("Memory share of GPU cost", ""), (pct(g["mem_share"]), "y"), ("assumption", "y"), ("Teardown: HBM about 41% plus packaging about 23% (mostly memory) against a logic die of about 9%, so roughly 60/40 (Evidence tab)", "")],
        [("Cost-weighted reduction (×)", ""), (x1(reduction_factor(g)), "b"), ("derived", "b"), ("1 ÷ (memory share ÷ memory factor + compute share ÷ compute factor). The component that shrinks least sets the floor", "")],
        [("Energy reduction (×)", ""), (x1(energy_reduction(g)), ek[1]), (ek[0], ek[1]), ("Equals the cost-weighted reduction by default (energy splits between memory and compute the way cost does). Override in the sidebar", "")],
        [("Discount rate", ""), (pct(g["discount_rate"]), "y"), ("assumption", "y"), ("Capitalization rate for a perpetuity; set it to your cost of capital (6% gives ×16.7)", "")],
        [("Fully-loaded $ per GPU", ""), (usd0(g["gpu_cost"]), "y"), ("assumption", "y"), ("A B200-class GPU (about $40k) plus its share of the server, NVLink and networking", "")],
        [("Wall power per GPU", ""), (f"{g['wall_power_kw']:.1f} kW", "y"), ("assumption", "y"), ("About 1 kW of chip power times a PUE of about 1.3, plus node overhead", "")],
        [("Electricity rate", ""), (f"${g['elec_rate']:.2f}/kWh", "y"), ("assumption", "y"), ("Datacenter wholesale, about $0.06–0.10 per kWh", "")],
        [("Cooling and operations overhead", ""), (pct(g["cooling_overhead"]), "y"), ("assumption", "y"), ("Non-power running cost as a fraction of electricity", "")],
        [("Fleet life", ""), (f"{g['fleet_life_yr']:.0f} yr", "y"), ("assumption", "y"), ("AI-GPU depreciation life. Filings say 5–6 years; we use 4, which is conservative", "")],
        [("Datacenter scaling factor (older Totals engine only)", ""), (pct(g["dc_scale"]), "y"), ("toggle", "y"), ("Technical tabs only: the share of the non-accelerator datacenter that also shrinks (0 means accelerators only). The front tabs let the servers around the chips and the datacenters follow the fleet by default; holding the datacenters (leases contracted to FY33) is their sensitivity", "")],
        [("Named share of global AI capex", ""), (pct(g["named_share_of_global"]), "y"), ("assumption", "y"), ("The named firms' share of worldwide AI capex; the remainder is grossed up in proportion", "")],
        [("SpaceX market cap", ""), (n0(g["spacex_mktcap"]), "g"), ("data", "g"), ("About $1.84T in August 2026 (IPO June 2026 at about $1.77T)", "")],
        [("Per-company total capex (FY2025)", ""), ("disclosed", "g"), ("data", "g"), ("10-K filings and earnings calls; see each company tab and its sources", "")],
        [("Per-company FY2026 capex", ""), ("estimate", "y"), ("assumption", "y"), ("Midpoint of management guidance; see each company tab", "")],
        [("Per-company datacenter, server and accelerator shares", ""), ("estimate", "y"), ("assumption", "y"), ("CFO commentary for the datacenter and server shares; teardowns for accelerators (67–80%)", "")],
        [("Per-company AI revenue", ""), ("mixed", "y"), ("assumption", "y"), ("Disclosed run-rates where available (Microsoft $37B, Amazon $15B); otherwise estimated", "")],
    ]
    show_table(["Input", "Current value", "Kind", "How it is derived, and the source"], rows,
               widths={"Input": "medium", "How it is derived, and the source": "large"})

    _hf = headline_family()
    _r4k = deck_layer_levers(4096, geometry=GEOM)['ratio']
    _r32k = deck_layer_levers(32768, geometry=GEOM)['ratio']
    _r128k = deck_layer_levers(131072, geometry=GEOM)['ratio']
    _r262k = deck_layer_levers(262144, geometry=GEOM)['ratio']
    _r262k_24 = deck_layer_levers(262144)['ratio']
    st.markdown(rf"""
### How the number is built

**The engine.** A GPU is about 60% memory and 40% compute by cost. The cost-weighted reduction is set by
whichever component shrinks least. Both scenarios price serving at a 262k-token average conversation.
The inference lever is tokens per GPU at equal model size (H5, {H5_CONVERSATIONS:.0f} conversations per GPU,
times H6, ×{H6_DECODE:.2f} per decode step, both measured on one layer, with prompt processing added:
×{INFERENCE_LEVER_EQUAL_SIZE:,.0f}) times the ×{INFERENCE_SIZE_FACTOR:.2f} smaller model of the same quality. A
{INFERENCE_SIZE_FACTOR:.2f}× smaller model costs about {INFERENCE_SIZE_FACTOR:.2f}× less per token, in inference as in
training. **Current kernels** carries prompt processing at the ×3.94 we have measured (inference lever about
×{INFERENCE_LEVER_CURRENT:,.0f}). **Optimized kernels** adds the remaining ×1.79 target (about ×{INFERENCE_LEVER_OPTIMIZED:,.0f})
and the fused-kernel training speed (H3 ×{TRAIN_SPEED_BY_SCENARIO['mature']:.2f} target against
×{TRAIN_SPEED_BY_SCENARIO['current']:.2f} measured). Only those ◆ inputs differ between the scenarios. At these
levers inference is limited by the compute lever (×{INFERENCE_LEVER_CURRENT:,.0f}), not by memory (H4, ×2,032).
Either way over 99.9% of the inference chip bill is gone, so the two scenarios land within rounding in
dollars. The levers do not multiply together, because costs add.

**The decode measurement.** One layer, bf16, a one-token step on a {GEOM['hbm_gb']:.1f} GB card. The transformer runs
the largest batch whose whole-model cache fits at that context; we run 256 conversations at 1 MB of state per
layer each, whatever the context (256 is where our grid stopped, not a ceiling). {FFN_TXT[0].upper() + FFN_TXT[1:]}.
The per-layer step ratio does not change with depth, so it is also the whole-model ratio. The transformer's
cache capacity and cache-bound step time do change with depth, so the levers are quoted for {GEOM_TXT},
anchored on {GEOM_ANCHOR}, rather than for the 24-layer d2048 model we measured. H5 × H6 for that
frontier model: ×{_r4k:.1f} at 4k, ×{_r32k:.0f} at 32k, ×{_r128k:.0f} at 128k, ×{_r262k:.0f} at 262k (the 24-layer model reads
×{_r262k_24:.0f} at 262k). At 262k the transformer fits one conversation (90 GiB of cache) and its single-conversation
step is {'faster' if H6_DECODE < 1 else 'slower'} than our 256-conversation step (H6 ×{H6_DECODE:.2f}). The lever is the 256
conversations advancing per step (H5).

**Training (H3): slower per step at short context, faster at long context on the fitted curve.** On one GH200, one layer,
forward and backward, bf16, no checkpointing on either side, against a modern transformer block (24 query
and 4 KV heads, head size 256, RoPE, gated attention), a bAttention training step costs ×1.72 more at 2,048
tokens (101.7 vs 59.3 ms) and ×1.79 at 8,192 (148.4 vs 83.1 ms, batch 4, with a possible batch artifact):
forward ×1.82, backward ×2.35. At equal quality the 2,048-token figure becomes ×0.41, a win, but at matched
size and short context it is still a loss, so the serving claim excludes training. Our per-token cost is
flat in context (measured: backward 3.27–3.31 µs per token across an 8× range) while the transformer's grows,
so the fitted cost curve crosses near 13k tokens (projected; measured separately at 21–24k with batch 4). One
caveat: the transformer's cost is fitted to both of its measured cells, but our cost is held at its 2k value, with
our measured 8k step treated as a batch-4 artifact, so the fit reads ×{training_step_ratio(8192):.2f} at 8k where
the measured cell read ×1.79. The ×{TRAIN_SPEED_BY_SCENARIO['current']:.2f} H3 figure is the cost ratio over an
assumed 8k/64k/256k training mix, where the 10% of tokens at 256k carry 47% of the transformer's bill. Training peak
memory runs against us the same way: ×1.58 at 2,048 tokens and ×1.57 at 8,192. That is training memory, a
different quantity from the H4 serving-memory lever.

**The remaining gap is a funded program, and its numbers are targets.** The goal is an 8,192-token step at
or below 83.08 ms, which beats the transformer and needs ×1.79 more, or 44% of the step. Named levers:
removing recompute (about 12.8 ms), redesigning the backward pass and occupancy (about 3× headroom in the
dominant fused kernel), and eliminating copies (24.8% of the step). Memory target ×0.88, at or below the
transformer's. None of that has a measurement yet, and it is exactly the factor inside the
Optimized-kernels scenario.

**The serving blend, for background.** A single compute number cannot represent training, prompt processing
and decode: at today's kernel maturity they point in different directions. Prompt processing crosses over
near 65k context. Decode crosses over near 30k, and only on aggregate throughput per GPU, never per token.
So the lever is computed from the workload rather than asserted. At 10:1 input to output, 64k average context
and training excluded, prompt processing ×1.17 and decode ×2.20 blend to ×2.19. Decode is about 99.7% of the
transformer's serving cost at that point, which is why it dominates. That blend compares the two families at
matched size. Our quality fits add the missing axis: how big each family has to be for the same quality.
Four models per family (47M–663M parameters) fitted as a power law in parameters cross at 392M, and above the
crossing bAttention reaches the transformer fit's quality on 84.2% of the parameters at 1B, 55.2% at 10B,
36.2% at 100B, 23.7% at 1T and 15.5% at 10T. Per-token cost is about linear in parameters, so both scenarios
multiply the serving lever by that ratio (×4.22 at 1T). It is a projection of two fits, not a measurement;
every point past about 1B extrapolates beyond the models we measured.

**Where the transformer wins.** At 100:1 input to output and 8k context the blend is ×0.28. A training share
above about 0.30 takes the blend below 1 (0.2 gives ×1.23). Both are reachable in the sidebar. Both phases use
measured throughput, with the transformer granted an idealized mature stack (paged attention, 8× KV compression,
bandwidth-floor serving), which is more generous than our measured lane. The memory lever (H4) is the measured
×2,032 at a 262k conversation, measured directly as a concurrency result below.

**One lane: 16-bit.** The prompt lever is measured where both families run their production 16-bit attention
kernels with parameters exactly matched (×4.02 at d512 and 1M, ×2.01 at d2048 and 262k), the lane the
competition serves in. 16-bit is also the production training lane (on our own baseline: ×1.185 step time,
−17.5% peak memory). The fp32 full-model sweep (×7.91 at 262k; no fp32 fused-attention kernel exists) is on
the Evidence tab as a scope reference only.

**Decode is a memory ceiling, not a latency win.** Per generated token with one conversation, the transformer
is faster than we are at 64k context (about 4.9 ms of GPU-busy against our flat 5.7 ms). What we win is the
ceiling: the transformer re-reads its entire KV cache for every token it emits, so once a card is full of
cache its aggregate throughput falls one-for-one with context while ours stays flat. Measured on one GPU at
262,144 tokens, the transformer fits one conversation (a second runs out of memory) and serves 64 tokens/s,
while bAttention holds 64 conversations in 1.62 GB and serves 562 tokens/s, an ×8.8 aggregate advantage. At
32,768 tokens the comparison is near parity, ×0.97–1.14: 8 conversations ran and 16 ran out of memory, so the
ceiling sits somewhere in 8–15, and whatever fits, a 32k step re-reads 6.44 GB of cache per conversation,
capping the transformer near 601 tokens/s against our 585. The lever crosses 1 near 30,000 tokens; below that,
the transformer serves more tokens per GPU-second than we do. Full-model state is 25.4 MB per conversation
against 197 KB of KV per token of context: ×508 at 64k, ×2,032 at 262k. Both measured cells understate us: the
transformer sits at 95.6% GPU-busy with no headroom, while our 64-conversation cell is 9.5% GPU-busy on a 96 GB
card on an unoptimized decode path. Granting the transformer 8× KV compression divides our decode lever by 8
and puts it ahead at 64k. We show that row. On 8 GPUs we do ×1.39 the training steps per
second at matched load (×5.15 forward-and-backward against ×3.70 forward-only) and ×1.70 with 256 sequences
in flight. Quality parity at 70B (×4.1 parameters or ×1.7 tokens) is a projection from the fits. The systems
numbers compare components, and every speed-up is against the family's own single-GPU baseline.

**Per company.** Total capex (disclosed) × datacenter share × server share × accelerator share gives the
accelerator capex, then the fleet, the power, the version with Helarctos, and the value (FY2026 is the base
year; FY2025 is last year). The datacenter share comes from the 10-K and 10-Q property and equipment and
segment notes (Amazon is the AWS share, 68–76%). The server share is disclosed by the CFO. Accelerators are
67–80% of a server from teardowns.

**Totals and global.** The named firms roll up live with no double-counting. The global row grosses the named
total up to a worldwide estimate using the named share of global AI capex. The rest is other clouds, China,
neoclouds, xAI, sovereign and enterprise.

**Net AI economics** are on a cash basis: AI revenue minus AI capex minus AI power. With Helarctos, add the
spend cut. All six firms lose money on AI today. AI spend is property and equipment additions (chips, the
servers around them, datacenter buildings, power and cooling, network) plus electricity. No labor is counted.
On the technical tabs the datacenter scaling factor sets how much of the non-accelerator datacenter shrinks
too. On the front tabs the servers and datacenters sized by the GPUs follow the blended fleet cut by default,
scaling down with the GPUs they house; holding the datacenters to their FY33 leases is the sidebar
sensitivity.

**Key results (FY2026 base year).** The headline is the value bridge (training priced on GPU-hours, inference
on whole GPUs, the servers and datacenters sized by those GPUs following the fleet):
**about \${_hf['fy26_bridge_spend_cut']:,.0f}B for FY2026** (net AI turns to about +\${_hf['fy26_bridge_net_with']:,.0f}B;
about \${_hf['fy26_bridge_spend_cut_held']:,.0f}B with the datacenters held to their FY33 leases, about
\${_hf['fy26_bridge_spend_cut_mature']:,.0f}B with the fused-kernel targets; FY2025, last year, about
\${_hf['fy25_bridge_spend_cut']:,.0f}B). The older engine (accelerators and power only, memory and compute priced
separately, cost-weighted reduction about ×{_hf['today_reduction']:,.0f}) reads about \${_hf['fy26_spend_cut']:,.0f}B for
FY2026 (about \${_hf['fy26_capitalized'] / 1000:.1f}T capitalized at 6%; global estimate about
\${_hf['global_fy26_capitalized'] / 1000:.1f}T) and about \${_hf['fy25_spend_cut']:,.0f}B for FY2025 (about
\${_hf['fy25_capitalized'] / 1000:.1f}T; global about \${_hf['global_fy25_capitalized'] / 1000:.1f}T; the FY2025 burn was about
−\$284B a year on about \$357B of AI capex against about \$79B of AI revenue). Current and Optimized kernels
land within rounding of each other in dollars: the cut is fleet × (1 − 1 ÷ lever) and it saturates, while the
capex, shares and revenue underneath never move.

**Sensitivity.** The avoided capex goes as (1 − 1 ÷ R), which is 0.999 at R of about 1,000, so the dollar
headline barely moves with the compute lever at these levels. The discount rate is a first-order lever on the
capitalized figures, which scale as 1 ÷ rate.

**Caveats.** AI revenue is the softest input (Microsoft \$37B and Amazon \$15B run-rates are disclosed; the
rest is estimated; Meta's real payoff is indirect, through ad uplift). Totals are disclosed; the server and
accelerator splits are estimated to within 15–20%. Capitalization is a simple perpetuity (benefit ÷ discount
rate). This is an analytical estimate, not investment advice.

**Sources.** SEC filings and earnings calls (Microsoft, Alphabet, Amazon, Meta and Oracle 10-Ks and
transcripts; the SpaceX S-1); hardware teardowns (Silicon Analysts); TPU and Trainium cost of ownership
(SemiAnalysis); GPU rental pricing (Spheron). Per-company source links are on each company tab.

Measurements as of {AS_OF}.
""")
    with st.expander("What changed and when"):
        st.markdown(rf"""
- **1 October 2026.** The inference lever moved onto the per-layer decode measurement (H5 256 conversations
  per GPU, H6 ×{H6_DECODE:.2f} per step) times the equal-quality size factor, which now applies to inference as well
  as training. H3 moved to the measured ×{TRAIN_SPEED_BY_SCENARIO['current']:.2f}. The servers and datacenters sized
  by the GPUs now follow the fleet by default. This replaced the estimate of 2.5 ms per token at 64
  conversations that the model carried from 31 August to 30 September.
- **1 September 2026.** The composite ×9.24 inference lever, measured on kernels that no longer exist,
  is retired.
- **29 September 2026.** The H1–H6 lever structure, the value bridge and the 10-K/10-Q datacenter shares.
- **24–29 August 2026.** The fused-kernel work cut our training step from 400.4 ms (×6.12 against the
  transformer) to 101.7 ms (×1.72) at 2,048 tokens, a ×3.94 speed-up. The older ×6.46 figure from a d1536,
  27-layer fp16 run is retired. The old ×5.5 maturity assumption became ×3.94 measured times a ×1.79 target.
  The 32,768-token near-parity cell at batch 4 is retired as a batch artifact.
- **14 August 2026.** A full-model bf16 decode measurement retired two numbers: a transformer decode cost of
  87.9 per token inflated by a harness artifact, and a 0.15 MB per-conversation state from a proxy that
  counted only part of the state. The retired headlines are ×53 decode, ×15.9 decode, 512 conversations at 0.15 MB, and the
  50–500× band. The equal-quality fits were also sealed on this date.
- **7–8 August 2026.** The 2, 4 and 8 GPU training and serving measurements on the Serving·Training tab.
- **21 July 2026.** The parameter-matched 16-bit prompt-processing sweep (×4.02 at d512 and 1M).
""")


# ---- audience layer: Summary / Value Bridge / Levers / What Matters (mirrors the workbook) ----
LEV = {x["code"]: x for x in HELARCTOS_LEVERS}
_LEVER_TEXT = {  # code: (how sure are we?, what it means, saves money in)
    "H1": ("Projected. Quality trends measured on models we trained (47M–663M parameters), extended to "
           "about 1T dense-equivalent",
           f"A Helarctos model matches a transformer of about 1T dense-equivalent parameters, roughly what "
           f"today's 5–6T mixture-of-experts flagships amount to, with "
           f"~{param_matching_fraction(DECK_DEPLOYMENT_SCALE):.0%} of the parameters: "
           f"{param_matching_gain(DECK_DEPLOYMENT_SCALE):.1f}× fewer numbers to store, update and run.",
           "Training and inference. A smaller model of the same quality costs proportionally less per token"),
    "H2": ("Projected. Standard compute-optimal scaling",
           f"Frontier labs train on data in proportion to model size, so a {param_matching_gain(DECK_DEPLOYMENT_SCALE):.1f}× "
           f"smaller model reaches its best quality on about {param_matching_gain(DECK_DEPLOYMENT_SCALE):.1f}× fewer tokens.",
           "Training"),
    "H3": (f"Measured at 2k and 8k, fitted beyond. The ×{TRAIN_SPEED_BY_SCENARIO['current']:.2f} is the cost ratio "
           f"over an assumed 8k/64k/256k training mix; ×{TRAIN_SPEED_BY_SCENARIO['mature']:.2f} is the fused-kernel target",
           f"At the same model size {h3_how(TRAIN_SPEED_BY_SCENARIO['current'])}. Our cost per token is flat in "
           "context while the transformer's grows, so the long-context phases dominate its bill.",
           "Training"),
    "H4": ("Measured ×2,032 at 262k on our test model. For a frontier-size model with grouped-query attention we "
           f"estimate about ÷{fleet_memory_lever(262144):.0f}, where memory would become the limit again (about −$0.5B)",
           "A transformer's memory (the KV cache) grows with every token of every live conversation: 51.6 GB "
           "for one 262k-token conversation. Helarctos keeps a fixed-size state (25.4 MB).",
           "Inference GPUs (the memory limit; not the binding one at 262k)"),
    "H5": (f"Measured on one layer, quoted for {GEOM_TXT}: {H5_CONVERSATIONS:.0f} live conversations at 1 MB of "
           f"state per layer each (where our grid stopped, not a ceiling) against the transformer's largest batch "
           f"whose whole-model cache fits the card (one at 262k, with 90 GiB of cache)",
           f"The small memory (H4) lets one GPU hold {H5_CONVERSATIONS:.0f} live 262k-token conversations at "
           f"once, where a transformer fits one.",
           "Inference GPUs (tokens per GPU)"),
    "H6": (f"Measured on one layer: ×{H6_DECODE:.2f} per decode step at 262k for {GEOM_TXT} "
           f"(×{deck_layer_levers(262144)['h6']:.2f} on the 24-layer model we measured); {FFN_TXT}",
           h6_meaning(H5_CONVERSATIONS, H6_DECODE),
           "Inference GPUs (tokens per GPU)"),
}


def lever_fmt(code, v):
    return f"÷{v:,.0f}" if code == "H4" else (f"×{v:,.0f}" if code == "H5" else f"×{v:.2f}")


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
        f"with AI revenue of {md_usd(t26['ai_rev'])}B. Helarctos models do the same work, training and "
        f"inference at the same quality, on far fewer GPUs. The servers and datacenters sized for those GPUs "
        f"shrink with them. This page shows how much of that spend becomes unnecessary and which levers drive "
        f"it. AI spend here means property and equipment additions (chips, the servers around them, datacenter "
        f"buildings, power and cooling, network) plus electricity. No labor is counted.")
    st.caption(f"Scenario: **{st.session_state.get('scenario')}** (◆ switch it in the sidebar) · "
               f"{ctx_k(st.session_state.get('context_tokens', 262144))}-token average conversation · "
               + ("datacenters **held** at today's spend because their leases run to FY33 (the sensitivity)"
                  if held else "servers and datacenters **follow** the fleet (the headline basis)")
               + " · $B per year unless stated")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Spend Helarctos makes unnecessary (FY2026)", f"{bn(t26['spend_cut'])}/yr",
              delta=f"{pct(t26['pct_cut'])} of AI spend")
    k2.metric("AI spend (FY2026)", f"{bn(spend26)}")
    k3.metric("Net AI cash result with Helarctos", bn(t26["net_with"]),
              delta=f"today: {bn(t26['net_now'])}", delta_color="off")
    k4.metric(f"Capitalized value (at {g['discount_rate']:.0%})", f"${t26['spend_cut'] / g['discount_rate'] / 1000:.1f}T")

    # what the scenario switch changes: both scenarios side by side, at the active datacenter switch
    cur, opt = _scenario_levers(g)
    _, tc = value_bridge(g, comps, "fy26", cur, ts, follow=fol)
    _, to = value_bridge(g, comps, "fy26", opt, ts, follow=fol)
    gpus = lambda d: d * 1e9 / g["gpu_cost"]  # noqa: E731
    section("What the Optimized-kernels scenario changes, and what it does to the bill")
    show_table(["", "Current kernels (measured)", "Optimized kernels (funded program, a target)"], [
        [("H3 · training speed per token, same model size", "h"), (f"×{cur['train_speed']:.2f}", "p"), (f"×{opt['train_speed']:.2f}", "p")],
        [("Training GPU-hour lever (H1 × H2 × H3)", ""), (f"×{cur['train_lever']:.1f}", "b"), (f"×{opt['train_lever']:.1f}", "b")],
        [("Prompt-processing speed against the transformer (supporting)", ""), (f"×{cur['prefill_speed']:.1f}", "p"), (f"×{opt['prefill_speed']:.1f}", "p")],
        [("Inference compute lever (tokens per GPU at equal size × the size factor)", ""),
         (f"×{cur['serving_compute_lever']:,.0f}", "b"), (f"×{opt['serving_compute_lever']:,.0f}", "b")],
        [("Training-fleet GPUs still needed (FY2026, GPU-equivalents)", ""),
         (n0(gpus(tc["train_fleet"] - tc["train_saved"])), "b"), (n0(gpus(to["train_fleet"] - to["train_saved"])), "b")],
        [("Inference-fleet GPUs still needed (FY2026, GPU-equivalents)", ""),
         (n0(gpus(tc["serve_fleet"] - tc["serve_saved"])), "b"), (n0(gpus(to["serve_fleet"] - to["serve_saved"])), "b")],
        [("Spend made unnecessary, FY2026", "!"), (bn(tc["spend_cut"]), "b!"), (bn(to["spend_cut"]), "b!")],
    ], wrap=True)
    _act = to if st.session_state.get("scenario") == "Optimized kernels" else tc
    shrink = 1.0 / max(1e-9, 1.0 - _act["fleet_cut"])
    st.caption(f"The bill barely moves between scenarios because the chip fleet already shrinks about {shrink:,.0f}× "
               f"(1 ÷ (1 − the {_act['fleet_cut']:.1%} fleet cut)). Once that is a few hundred times, better kernels "
               f"change the multiple but not the dollars. What is left is the servers and buildings that cannot "
               f"shrink below one per site, plus power. Active scenario: **{st.session_state.get('scenario')}**.")

    section("The headline")
    dc_lab = "…datacenters: buildings, power and cooling, network (held: leases run to FY33)" if held else \
        "…datacenters: buildings, power and cooling, network (follow the fleet)"
    show_table(["", "FY2026", "FY2025 (last year)"], [
        [("AI spend: chips, servers, datacenters, power (property and equipment additions plus electricity; no labor)", ""),
         (bn(t26["ai_capex"] + t26["ai_opex"]), "b"), (bn(t25["ai_capex"] + t25["ai_opex"]), "b")],
        [("…of which AI chips (GPUs, TPUs)", ""), (bn(t26["accel"]), "b"), (bn(t25["accel"]), "b")],
        [("…of which the servers around them", ""), (bn(t26["servers"]), "b"), (bn(t25["servers"]), "b")],
        [("…of which datacenters (buildings, power and cooling, network)", ""), (bn(t26["datacenter"]), "b"),
         (bn(t25["datacenter"]), "b")],
        [("…of which electricity for the chips, per year", ""), (f"${t26['power']:,.1f}B", "b"), (f"${t25['power']:,.1f}B", "b")],
        [("AI revenue", ""), (bn(t26["ai_rev"]), "b"), (bn(t25["ai_rev"]), "b")],
        [("Net AI cash result today", ""), (bn(t26["net_now"]), "b"), (bn(t25["net_now"]), "b")],
        [("Spend Helarctos makes unnecessary, per year", "!"), (bn(t26["spend_cut"]), "k!"),
         (bn(t25["spend_cut"]), "k!")],
        [("…AI chips and their power", ""), (bn(t26["accel_saved"] + t26["opex_saved"]), "b"),
         (bn(t25["accel_saved"] + t25["opex_saved"]), "b")],
        [("…the servers around those chips (follow the fleet)", ""), (bn(t26["servers_saved"]), "b"),
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
    section(f"The FY2026 saving by fleet, adding to {bn(tot)}")
    st.caption("A breakdown of the one headline above. Each fleet's figure includes its chips, their power, and its "
               "share of the servers and datacenters that follow the fleet, split by how much of the chip saving "
               "each fleet produced.")
    lever_rows = {"Training": ["H1", "H2", "H3"], "Inference": ["H4", "H5", "H6"]}
    lever_text = {"H1": f"Same quality with about {param_matching_fraction(DECK_DEPLOYMENT_SCALE):.0%} of the parameters "
                        f"(projected from models we trained). Also applies to inference: a "
                        f"{lv['smaller_model']:.2f}× smaller model costs about {lv['smaller_model']:.2f}× less per token.",
                  "H2": "Compute-optimal training needs data in proportion to model size.",
                  "H3": (f"At the same model size {h3_how(lv['train_speed'])}."
                         if st.session_state.get('scenario') != 'Optimized kernels' else
                         f"×{lv['train_speed']:.2f} per token at the same size over an assumed 8k/64k/256k training mix (fused-kernel target)."),
                  "H4": "A fixed-size state instead of a memory that grows with every token (measured ×2,032 at "
                        "262k on our test model).",
                  "H5": f"{lv['conversations_per_gpu']:,.0f} live conversations per GPU where a transformer fits one at 262k "
                        f"(measured on one layer).",
                  "H6": f"×{lv['faster_decode']:.2f} per decode step against the transformer, measured on one layer "
                        f"({FFN_TXT}). "
                        + h6_meaning(lv['conversations_per_gpu'], lv['faster_decode'])}
    train_amt = t26["training_total"]
    infer_amt = t26["inference_total"]
    followed = t26["servers_saved"] + t26["datacenter_saved"]
    body = []
    for fleet, val, lever, how in (
        ("Training", train_amt, f"×{lv['train_lever']:.1f}",
         f"H1 × H2 × H3 gives about {lv['train_lever']:.0f}× fewer GPU-hours per training run, so training clusters "
         f"can be that much smaller. The GPUs never bought are capex avoided, and their power is saved every year."),
        ("Inference", infer_amt, f"×{lv['serving_gpu_lever']:,.0f}",
         f"Tokens per GPU at equal size rise about {lv['serving_compute_lever'] / lv['size_factor']:,.0f}× (H5 × H6, with "
         f"prompts included), times the ×{lv['size_factor']:.2f} smaller model of the same quality (H1), gives "
         f"×{lv['serving_compute_lever']:,.0f}. A GPU is bought whole, so the inference fleet shrinks by the smaller of "
         f"memory (×{lv['memory']:,.0f}) and that: about {lv['serving_gpu_lever']:,.0f}× fewer GPUs, plus the power they "
         f"would draw."),
    ):
        body.append([(f"{fleet} fleet (chips, power, and its share of the servers and datacenters that follow)", ""),
                     (bn(val), "b"), (pct(val / tot), "b"), (lever, "b"), (how, "")])
        for code in lever_rows[fleet]:
            body.append([("↳ " + lever_label(code), "h"), ("", ""), ("", ""),
                         (lever_fmt(code, lv[LEV[code]["key"]]), lever_value_code(code)),
                         (lever_text[code], "")])
    body.append([("↳ of which: the servers and datacenters that follow the fleet (inside both fleets above)", ""),
                 (bn(followed), "b"), (pct(followed / tot), "b"), (f"−{t26['fleet_cut']:.1%}", "b"),
                 (f"Servers {bn(t26['servers_saved'])} plus datacenters {bn(t26['datacenter_saved'])}"
                  + (" (held: leases run to FY33; untick the sidebar switch to let them follow)" if held else
                     f". Both are sized by the GPUs, so they fall with the blended chip-fleet cut ({t26['fleet_cut']:.1%}). "
                     f"Tick the sidebar switch to hold the datacenters at their FY33 leases")
                  + ". Split between training and inference by each fleet's share of the chip saving.", "")])
    body.append([("Total: the headline", "s!"), (bn(tot), "s!"), ("100%", "s!"), ("", "s"),
                 ("The step-by-step build and the one-lever-at-a-time view are on the Value Bridge tab.", "s")])
    show_table(["Fleet / Helarctos lever", "$B per year", "Share", "Lever", "How it works"], body, wrap=True)
    _stacked_bar(f"FY2026 saving {bn(tot)}/yr, training against inference (each includes power and its share of "
                 f"the servers and datacenters that follow)",
                 ["Training", "Inference"], [train_amt, infer_amt], SOURCE_COLORS[:2])

    section("By company, FY2026, $B per year")
    show_table(["Company", "AI spend", "Training saving", "Inference saving", "Servers and datacenters",
                "Total saving", "Share of AI spend", "Net AI today", "Net AI with Helarctos"],
               [[(r["name"], "s!" if r is t26 else ""), (bn(r["ai_capex"] + r["ai_opex"]), "s" if r is t26 else "b"),
                 (bn(r["train_saved"] + r["train_power_saved"]), "s" if r is t26 else "b"),
                 (bn(r["serve_saved"] + r["infer_power_saved"]), "s" if r is t26 else "b"),
                 (bn(r["servers_saved"] + r["datacenter_saved"]), "s" if r is t26 else "b"),
                 (bn(r["spend_cut"]), "s!" if r is t26 else "b!"), (pct(r["pct_cut"]), "s" if r is t26 else "b"),
                 (bn(r["net_now"]), "s" if r is t26 else "b"), (bn(r["net_with"]), "s!" if r is t26 else "b")]
                for r in rows26 + [t26]], wrap=True)

    section("Why we report dollars rather than a multiple")
    st.markdown(f"A GPU is bought whole, so the inference fleet shrinks by whichever need falls least "
                f"(×{lv['serving_gpu_lever']:,.0f}), not by memory times tokens per GPU "
                f"({lv['memory']:,.0f} × {lv['serving_compute_lever']:,.0f}). Cutting a fleet "
                f"{lv['serving_gpu_lever']:,.0f}× already removes over 99% of it, and bigger multiples only move the "
                f"last fraction of a percent. The dollars are therefore set by how much these firms spend on chips, "
                f"and on the servers and datacenters sized for those chips. The compute side is itself a product: "
                f"tokens per GPU at equal size (×{lv['serving_compute_lever'] / lv['size_factor']:,.0f}, H5 × H6 with "
                f"prompts) times the ×{lv['size_factor']:.2f} smaller model of the same quality (H1). That product "
                f"does not multiply the dollars any further.")

    section("How to read this app")
    st.markdown(LEGEND_MD, unsafe_allow_html=True)
    st.caption("Teal marks one of the six things the Helarctos architecture changes (H1–H6). Every other input is "
               "market, company or modeling data. ◆ marks the only values that change between Current and "
               "Optimized kernels. ★ marks an input that moves the FY2026 saving by \\$10B or more (What Matters "
               "tab). Cash basis. FY2026 is the base year, with capex from company guidance or actuals as noted on "
               "each company tab. AI spend is property and equipment additions (chips, servers, datacenters) plus "
               "electricity, with no labor. The servers and datacenters sized by the GPUs follow the fleet by "
               "default; holding the datacenters to their FY33 leases is the sidebar sensitivity. Not investment "
               "advice.")


def value_bridge_tab(comps, g):
    lv, ts, fol = _bridge_levers(g), _train_shares(), _follow(g)
    held = _held(g)
    rows, t = value_bridge(g, comps, "fy26", lv, ts, follow=fol)
    st.caption("Read it top to bottom. Teal rows are the Helarctos levers (edit them in the sidebar and on the "
               "Levers tab). Blue cells are formulas. "
               + ("Datacenters are held at today's spend (sidebar switch) because their leases run to FY33." if held else
                  "Servers and datacenters follow the fleet, the headline basis. The sidebar switch holds the datacenters."))
    section("Step 1: what the six companies spend on AI in FY2026 (property and equipment additions plus electricity, no labor)")
    show_table(["Item", "$B", "Note"], [
        [("AI datacenter capex", ""), (bn(t["ai_capex"]), "b"), ("company filings and guidance", "")],
        [("…of which AI chips (GPUs, TPUs)", "!"), (bn(t["accel"]), "b!"),
         ("the part the Helarctos levers act on directly", "")],
        [("…of which the servers around them (CPUs, chassis, networking)", ""), (bn(t["servers"]), "b"),
         ("the rest of the server bucket, sized by the accelerator count, so it follows the fleet", "")],
        [("…of which datacenters (buildings, power and cooling, network)", ""), (bn(t["datacenter"]), "b"),
         ("sized by the GPUs they house; " + ("held here, since the leases run to FY33" if held else "follows the fleet by default"), "")],
        [("Power and operations for those chips, per year", ""), (f"${t['ai_opex']:,.1f}B", "b"), ("electricity; always follows the fleet", "")],
        [("AI revenue", ""), (bn(t["ai_rev"]), "b"),
         ("disclosed run-rates (Microsoft, Amazon) or estimates (the others)", "")],
        [("Net AI cash result today", "!"), (bn(t["net_now"]), "b!"), ("revenue − capex − power", "")],
    ], wrap=True)
    section("Step 2: split the chip fleet by what it does")
    show_table(["Fleet", "$B", "Share", "Note"], [
        [("Training fleet, which builds new models", ""), (bn(t["train_fleet"]), "b"), (pct(t["train_share"]), "b"),
         ("per-company estimates of 25–55% (Levers tab); chip-weighted average", "")],
        [("Inference fleet, which answers users", ""), (bn(t["serve_fleet"]), "b"),
         (pct(1 - t["train_share"]), "b"), ("", "")],
    ], wrap=True)

    def lever_row(code, note):
        return [(lever_label(code), "h"), ("", ""), (lever_fmt(code, lv[LEV[code]["key"]]), lever_value_code(code)),
                (note, "")]

    section("Step 3: training. A smaller model, fewer tokens and a lower cost per token over the mix mean smaller clusters")
    show_table(["Item", "$B", "Lever", "Note"], [
        [("Training fleet capex today", ""), (bn(t["train_fleet"]), "b"), ("", ""), ("", "")],
        lever_row("H1", "same quality with a fraction of the parameters (projected)"),
        lever_row("H2", "a smaller model needs proportionally fewer training tokens"),
        lever_row("H3", (f"at the same model size {h3_how(lv['train_speed'])}"
                         if st.session_state.get('scenario') != 'Optimized kernels' else
                         "faster per token at the same size over an assumed 8k/64k/256k training mix (fused-kernel target)")),
        [("GPU-hours per training run fall by", "!"), ("", ""), (f"×{lv['train_lever']:.1f}", "b!"), ("H1 × H2 × H3", "")],
        [("Training fleet needed with Helarctos", ""), (bn(t["train_fleet"] / lv["train_lever"]), "b"), ("", ""), ("", "")],
        [("Training capex avoided", ""), (bn(t["train_saved"]), "b"), ("", ""), ("", "")],
        [("Plus the power and operations those GPUs would have drawn", ""), (f"${t['train_power_saved']:,.1f}B", "b"),
         ("", ""), ("power scales with the fleet", "")],
        [("Training saving (chips and power)", "!"), (bn(t["train_saved"] + t["train_power_saved"]), "b!"), ("", ""),
         (f"labs size training clusters to GPU-hours, so about {lv['train_lever']:.0f}× fewer GPU-hours means a cluster "
          f"about {lv['train_lever']:.0f}× smaller for the same program", "")],
    ], wrap=True)
    section("Step 4: inference. Small memory means more conversations per GPU, on a smaller model, so fewer GPUs")
    eq = lv["serving_compute_lever"] / lv["size_factor"]
    show_table(["Item", "$B", "Lever", "Note"], [
        [("Inference fleet capex today", ""), (bn(t["serve_fleet"]), "b"), ("", ""), ("", "")],
        lever_row("H4", "a fixed-size state instead of a memory that grows with every token"),
        lever_row("H5", f"{lv['conversations_per_gpu']:,.0f} live 262k-token conversations per GPU where a transformer "
                        f"fits one (measured on one layer)"),
        lever_row("H6", f"decode step against the transformer at its feasible batch (measured on one layer; {FFN_TXT}). "
                        + h6_meaning(lv["conversations_per_gpu"], lv["faster_decode"])),
        [("Tokens per GPU at equal model size: H5 × H6 with prompt processing included", ""), ("", ""),
         (f"×{eq:,.0f}", "b"),
         (f"prompts are about 0.2% of a transformer's time at 262k (prompt-processing speed ×{lv['prefill_speed']:.1f}, "
          f"the only inference input that differs by scenario)", "")],
        [("Times the smaller model of the same quality (H1), the inference size factor", ""), ("", ""),
         (f"×{lv['size_factor']:.2f}", "b"),
         ("a model with about 24% of the parameters costs proportionally less per token, in inference as in "
          "training", "")],
        [("Equals the inference compute lever", "!"), ("", ""), (f"×{lv['serving_compute_lever']:,.0f}", "b!"), ("", "")],
        [("Inference GPUs needed fall by the smaller of memory and the compute lever", "!"), ("", ""),
         (f"×{lv['serving_gpu_lever']:,.0f}", "b!"),
         ("a GPU is bought whole, memory and compute together, so the fleet covers whichever runs out first. "
          + ("Here the compute lever binds, so the decode step (H6) and the size factor count in the dollars"
             if lv["serving_compute_lever"] <= lv["memory"] else "Here memory (H4) binds"), "")],
        [("Inference fleet needed with Helarctos", ""), (bn(t["serve_fleet"] / lv["serving_gpu_lever"]), "b"), ("", ""), ("", "")],
        [("Inference capex avoided", ""), (bn(t["serve_saved"]), "b"), ("", ""), ("", "")],
        [("Plus the power and operations those GPUs would have drawn", ""), (f"${t['infer_power_saved']:,.1f}B", "b"),
         ("", ""), ("power scales with the fleet", "")],
        [("Inference saving (chips and power)", "!"), (bn(t["serve_saved"] + t["infer_power_saved"]), "b!"), ("", ""), ("", "")],
    ], wrap=True)
    section("Step 5: the servers and datacenters sized for those GPUs follow the fleet")
    show_table(["Item", "$B", "Factor", "Note"], [
        [("Chip fleet cut (training and inference blended)", "!"), ("", ""), (f"−{t['fleet_cut']:.1%}", "b!"),
         ("chip capex avoided ÷ chip capex today. The datacenters serve both training and inference, so the "
          "same blend applies to them", "")],
        [("Servers around the chips today", ""), (bn(t["servers"]), "b"), ("", ""), ("", "")],
        [("…follow the fleet", ""), ("", ""), (f"{fol['servers']:.0%}", "y"), ("the fraction of the bucket that scales with the accelerator count", "")],
        [("Servers saved", ""), (bn(t["servers_saved"]), "b"), ("", ""), ("", "")],
        [("Datacenters today (buildings, power and cooling, network)", ""), (bn(t["datacenter"]), "b"), ("", ""), ("", "")],
        [("…follow the fleet", ""), ("", ""), (f"{fol['datacenter']:.0%}", "y"),
         ("held: the leases run to FY33 (sidebar switch)" if held else
          "by default they scale down when no longer needed; tick the sidebar switch to hold them at their FY33 leases", "")],
        [("Datacenters saved", ""), (bn(t["datacenter_saved"]), "b"), ("", ""), ("", "")],
        [("Saving from the servers and datacenters that follow", "!"), (bn(t["servers_saved"] + t["datacenter_saved"]), "b!"), ("", ""), ("", "")],
    ], wrap=True)
    section("Result, FY2026")
    show_table(["Item", "$B"], [
        [("Training (chips and power)", ""), (bn(t["train_saved"] + t["train_power_saved"]), "b")],
        [("Inference (chips and power)", ""), (bn(t["serve_saved"] + t["infer_power_saved"]), "b")],
        [("Servers and datacenters that follow the fleet", ""), (bn(t["servers_saved"] + t["datacenter_saved"]), "b")],
        [("Spend Helarctos makes unnecessary, per year", "!"), (bn(t["spend_cut"]), "k!")],
        [("…as a share of all AI spend", "!"), (pct(t["pct_cut"]), "b!")],
        [("…as a share of the AI-chip bill", ""), (pct(t["accel_saved"] / t["accel"]), "b")],
        [("Net AI cash result with Helarctos", "!"), (bn(t["net_with"]), "b!")],
        [("Value of the yearly saving, capitalized (÷ discount rate)", ""),
         (f"${t['spend_cut'] / g['discount_rate'] / 1000:.1f}T", "b")],
    ], wrap=True)
    section("Switch the Helarctos levers on one at a time (FY2026)")
    lad = savings_ladder(g, comps, "fy26", lv, ts, follow=fol)
    labels = [s_["step"].replace(" (training)", ": training (H1–H3)").replace(" (inference)", ": inference (H4–H6)")
              .replace("the servers and datacenters sized by those GPUs follow",
                       "the servers sized by those GPUs follow (datacenters held)" if held
                       else "the servers and datacenters sized by those GPUs follow")
              for s_ in lad]

    def _fol(f):
        return ("servers and datacenters" if f["datacenter"] else "servers") if f["servers"] else "—"

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
    st.caption("Training savings come from a smaller model, trained on fewer tokens and faster per token. The "
               "fewer parameters and fewer tokens compound. Inference savings come from fixed-size memory, which "
               "lets each GPU hold many more long conversations and advance them all on each decode step, on a "
               "smaller model. Each fleet then shrinks by 96–99.9%, and bigger multiples add little because a cost "
               "can only fall to zero once. The last step is not a Helarctos lever at all. It is the servers and "
               "datacenters sized for the GPUs following the fleet. So this model reports dollars rather than "
               "multiples.")
    _, tt = compute_year(g, comps, "fy26")
    chips_only = t["train_saved"] + t["serve_saved"] + t["opex_saved"]
    st.caption(f"Cross-check: the technical Totals tab prices memory and compute separately and counts chips plus the "
               f"older datacenter-scaling share, {md_usd(tt['spend_cut'])}B for FY2026, against {md_usd(chips_only)}B for "
               f"chips and power here. The two agree within rounding, since both remove about 99.9% of the chip bill. "
               f"The remaining {md_usd(t['servers_saved'] + t['datacenter_saved'])}B here is the servers and datacenters "
               f"following the fleet, which the Totals engine does not count.")

    section("Detail by company, FY2026, $B (the engine behind every number above)")
    det_cols = ["Company", "AI-chip capex", "Training share", "Training fleet", "Inference fleet",
                "Training capex avoided", "Inference capex avoided", "Servers around chips today", "Servers saved",
                "Datacenters today", "Datacenters saved", "Power today", "Power saved", "Total saving",
                "AI spend (capex and power)", "Share of AI spend cut", "AI revenue", "Net AI today", "Net AI with Helarctos"]
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
    st.caption("The six teal levers (H1–H6) are the only inputs about the Helarctos architecture. Everything else "
               "is market or company data, listed further down. ◆ purple means the value differs between the two "
               "scenarios. ★ orange border means a high-impact input (What Matters tab).")
    section(f"The six things the architecture changes. Active scenario: "
            f"{st.session_state.get('scenario')} (sidebar)")
    rows = []
    for x in HELARCTOS_LEVERS:
        code, k = x["code"], x["key"]
        status, what, where = _LEVER_TEXT[code]
        differs = x["by_scenario"]
        if code == "H3" and st.session_state.get("scenario") == "Optimized kernels":
            status = (f"Target ×{TRAIN_SPEED_BY_SCENARIO['mature']:.2f}, the fused-kernel goal at 8k (not a measurement). "
                      f"×{TRAIN_SPEED_BY_SCENARIO['current']:.2f} is measured on today's kernels")
        rows.append([(code, "h"), (lever_label(code).split(" · ", 1)[1], "h"),
                     (lever_fmt(code, lv[k]), lever_value_code(code)),
                     (lever_fmt(code, cur[k]) if differs else "same", "p" if differs else ""),
                     (lever_fmt(code, opt[k]) if differs else "same", "p" if differs else ""),
                     (status, ""), (what, ""), (where, "")])
    rows.append([("", ""), ("Supporting: prompt-processing speed against the transformer ◆", ""),
                 (f"×{lv['prefill_speed']:.1f}", "p"), (f"×{cur['prefill_speed']:.1f}", "p"),
                 (f"×{opt['prefill_speed']:.1f}", "p"),
                 ("A ×3.94 kernel speed-up is measured; the full ×7.03 is a target", ""),
                 ("A property of our kernels, not a headline lever. Prompts are about 0.2% of a transformer's time at "
                  "262k, so this barely moves the dollars.", ""), ("Inference GPUs (tokens per GPU)", "")])
    rows.append([("", ""), ("Supporting: the inference size factor (equals H1)", ""),
                 (f"×{lv['size_factor']:.2f}", "b"), ("same", ""), ("same", ""),
                 ("Projected. The same equal-quality ratio as H1, from our fits", ""),
                 ("A smaller model of the same quality costs proportionally less per token, so the ratio multiplies "
                  "the inference compute lever as well as the training lever.", ""), ("Inference GPUs (tokens per GPU)", "")])
    show_table(["#", "Helarctos lever", "Active value", "◆ Current kernels", "◆ Optimized kernels",
                "How sure are we?", "What it means", "Saves money in"], rows, wrap=True)
    st.caption("Where the numbers come from. H5 and H6: a one-token decode step measured on one layer in bf16, "
               f"the transformer at the largest batch whose whole-model cache fits a {GEOM['hbm_gb']:.1f} GB card at "
               f"that context and us at {H5_CONVERSATIONS:.0f} conversations ({FFN_TXT}), quoted for {GEOM_TXT}, "
               f"anchored on {GEOM_ANCHOR}. The per-layer step ratio does not change with depth, but the "
               "transformer's cache capacity and cache-bound step time do. H3: single-layer training steps measured "
               "at 2k and 8k, fitted beyond, weighted over an assumed 8k/64k/256k training mix. H4: a full-model "
               "decode measurement on our test model. H1 and H2: our quality fits, a projection of two fits. "
               f"Measurements as of {AS_OF}.")
    section("How the levers combine: one number per fleet")
    eq = lv["serving_compute_lever"] / lv["size_factor"]
    show_table(["Fleet lever", "Value", "Built from"], [
        [("Training: GPU-hours per training run fall by", "!"), (f"×{lv['train_lever']:.1f}", "b!"),
         ("H1 × H2 × H3. Training clusters are sized to GPU-hours", "")],
        [("Inference: memory per conversation falls by", "!"), (f"÷{lv['memory']:,.0f}", "b!"), ("H4", "")],
        [("Inference: decode tokens per GPU rise by", "!"),
         (f"×{lv['conversations_per_gpu'] * lv['faster_decode']:,.0f}", "b!"),
         (f"H5 × H6: conversations per GPU times the per-step time ratio (measured on one layer, quoted for {GEOM_TXT})", "")],
        [("Inference: tokens per GPU at equal model size, including prompt processing", "!"),
         (f"×{eq:,.0f}", "b!"),
         ("H5 × H6 with prompts included at the supporting prompt-processing speed", "")],
        [("Inference: compute lever = tokens per GPU × the equal-quality size factor", "!"),
         (f"×{lv['serving_compute_lever']:,.0f}", "b!"),
         (f"× H1 (×{lv['size_factor']:.2f}). A smaller model of the same quality costs proportionally less per token, in "
          f"inference as in training", "")],
        [("Inference: GPUs needed fall by", "!"), (f"×{lv['serving_gpu_lever']:,.0f}", "b!"),
         (f"the smaller of memory (H4) and the compute lever. A GPU is bought whole, so the fleet covers whichever "
          f"runs out first (here {'the compute lever, so the decode step and the size factor count' if lv['serving_compute_lever'] <= lv['memory'] else 'memory'}). "
          f"Multiplying them ({lv['memory']:,.0f} × {lv['serving_compute_lever']:,.0f}) would be wrong", "")],
        [("Servers and datacenters sized for those GPUs follow the fleet", "!"),
         (f"servers {fol['servers']:.0%} · datacenters {fol['datacenter']:.0%}", "y"),
         ("Not a Helarctos lever. Each bucket that follows falls by the blended chip-fleet cut times this fraction. "
          + ("Datacenters are held (sidebar) because their leases run to FY33." if held else
             "By default both follow. The sidebar switch holds the datacenters at their FY33 leases."), "")],
        [("For reference: the technical tabs' blended cut", ""), (f"×{reduction_factor(g):,.0f}", "b"),
         ("the appendix prices memory (about 60% of a GPU's cost) and compute separately. Same chip bill; both "
          "remove about 99.9% of it", "")],
    ], wrap=True)
    section("Market and company data: the other inputs, none of them about Helarctos")
    _, t = value_bridge(g, comps, "fy26", lv, _train_shares(), follow=fol)
    show_table(["Input", "Value", "Where to edit", "Note"], [
        [("AI datacenter capex, FY2026 (six companies)", ""), (bn(t["ai_capex"]), "b"),
         ("✏️ panel on each company tab ★", ""), ("disclosed or guided capex × datacenter share (10-K and 10-Q notes); property and equipment additions, no labor", "")],
        [("…share that buys AI chips", ""), (pct(t["accel"] / t["ai_capex"]), "b"),
         ("✏️ panel on each company tab ★", ""), ("server share (disclosed by the CFO) × accelerator share of servers", "")],
        [("…the servers around the chips", ""), (bn(t["servers"]), "b"), ("✏️ panel on each company tab", ""),
         ("server bucket minus accelerators; follows the fleet", "")],
        [("…datacenters: buildings, power and cooling, network", ""), (bn(t["datacenter"]), "b"),
         ("✏️ panel on each company tab ★", ""), ("AI datacenter capex minus the server bucket; " + ("held (sidebar)" if held else "follows the fleet (the sidebar switch holds it)"), "")],
        [("Training share of the chip fleet (chip-weighted)", ""), (pct(t["train_share"]), "b"),
         ("per company, below", ""), ("analyst estimates; no company discloses this", "")],
        [("Fully-loaded cost per GPU", ""), (usd0(g["gpu_cost"]), "b"), ("sidebar", ""),
         ("sizes the fleet for the power calculation", "")],
        [("Wall power per GPU (kW)", ""), (f"{g['wall_power_kw']:.1f}", "b"), ("sidebar", ""),
         ("power is about 2% of the saving", "")],
        [("Electricity ($/kWh)", ""), (f"${g['elec_rate']:.2f}", "b"), ("sidebar", ""), ("power is about 2% of the saving", "")],
        [("Discount rate ★ (capitalized value only)", ""), (pct(g["discount_rate"]), "b*"), ("sidebar", ""),
         ("changes only the capitalized value, never the yearly saving", "")],
    ], wrap=True)
    section("Training share of each company's AI-chip fleet (market estimate, editable)")
    st.caption("No company discloses this. Analysts put training at 30–45% of AI compute in 2026 (Gartner, "
               "Deloitte). The rest is inference, which means serving users.")
    _train_shares()
    cols = st.columns(len(COMPANIES))
    for col, c in zip(cols, COMPANIES):
        col.number_input(c["name"], min_value=0.0, max_value=1.0, step=0.05, format="%.2f",
                         key=f"ts_{c['name']}")
    section("Fine print")
    st.markdown("- AI spend is property and equipment additions (AI chips, the servers around them, datacenter "
                "buildings, power and cooling, network) plus electricity. No labor is counted anywhere.\n"
                "- The servers and datacenters sized for the GPUs follow the blended chip-fleet cut by default, "
                "scaling down with the GPUs they house. Holding the datacenters, whose leases run to FY33, is the "
                "sidebar sensitivity. It delays that saving rather than removing it.\n"
                "- Training is priced on GPU-hours only. No memory credit is taken on training clusters.\n"
                "- H4 (×2,032) is measured on our test model. For a frontier-size model with grouped-query attention "
                "the estimate is about ×208, where memory would become the limit again (about −\\$2B on FY2026). H5 "
                f"and H6 are measured on one layer; {FFN_TXT}. H3 is measured step cost over an assumed training mix.\n"
                "- H1 is a projection: quality trends measured up to 663M parameters, extended to about 1T "
                "dense-equivalent, roughly what today's 5–6T mixture-of-experts flagships (Grok 5 at 6T, Kimi K3 "
                "at 2.8T) amount to. It applies to inference as well as training.\n"
                "- The saving is spend no longer needed for the same AI output. Firms will likely reinvest it. "
                "Cash basis; capitalized value is the yearly saving ÷ the discount rate.")


def _wm_why():
    """Why-it-matters notes for the What Matters rows, with the live values."""
    def _tp(ctx):
        h5, h6 = decode_levers(ctx)
        return inference_throughput(h5, h6, prefill_advantage("current", ctx), prompt_time_ratio(ctx)) * INFERENCE_SIZE_FACTOR
    out = {
        "Server share of AI capex": "How much AI capex buys servers rather than buildings, power and networking. "
            "With both buckets following the fleet the split barely matters. Disclosed by the CFO for Microsoft and Alphabet.",
        "Accelerator share of servers": "The GPU and TPU share of server spend. With the servers around the chips "
            "following the fleet, the split barely matters. From teardowns (67–80%).",
        "Smaller model for the same quality": "The biggest Helarctos-specific assumption. It drives the training "
            "saving and multiplies the inference lever. A projection from models up to 663M parameters to about 1T "
            "dense-equivalent (today's 5–6T mixture-of-experts flagships). High case: a 5T dense model.",
        "More conversations per GPU": f"{H5_CONVERSATIONS:.0f} against the transformer's one at 262k (measured on one "
            f"layer). Small effect: inference GPUs already shrink by about 99.9%, so even the 64-conversation "
            f"full-model cell costs under $1B.",
        "Faster decode per token": f"×{H6_DECODE:.2f} per step at 262k (measured on one layer; {FFN_TXT}). "
            f"It counts in the dollars, but only about $0.4B across the range.",
        "Memory per conversation": "Measured ×2,032 at 262k on our test model. For a frontier-size model with "
            "grouped-query attention we estimate about ÷208, where memory would become the limit again (about −$2B).",
        "Average conversation length": f"Shorter conversations shrink both the memory advantage and tokens per GPU "
            f"(inference lever ×{_tp(32768):,.0f} at 32k). Longer ones grow them (×{_tp(1048576):,.0f} at 1M, where one "
            f"transformer conversation no longer fits a GPU). Set it in the sidebar.",
        "Data-center share of capex": "The share of capex that goes into datacenters (10-K and 10-Q property notes, "
            "93–98%; Amazon's AWS share 68–76%). The whole AI bill follows the fleet, so this moves the dollars one-for-one.",
        "Datacenters follow the fleet": "By default the datacenter bucket (buildings, power and cooling, network) "
            "scales down with the GPUs it houses. Held: the leases run to FY33, so that saving arrives later. This is "
            "the single largest sensitivity. Sidebar switch.",
        "Training speed per token": f"×{TRAIN_SPEED_BY_SCENARIO['current']:.2f} over an assumed training mix, measured "
            f"at 2k and 8k and fitted beyond; ×{TRAIN_SPEED_BY_SCENARIO['mature']:.2f} is the fused-kernel target. Small "
            f"effect: the smaller model already removes about 94% of training GPU-hours.",
        "Electricity rate": "Only changes the power part of the saving, about 2% of the total.",
        "Wall power per GPU": "Only changes the power part of the saving, about 2% of the total.",
        "Training share of the chip fleet": "Small effect: both fleets shrink by 96–99.9%.",
        "Memory share of GPU cost": "No effect on the front tabs, since GPUs are bought whole.",
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
    _kern = "mature" if st.session_state.get("scenario") == "Optimized kernels" else "current"
    st.caption("Each row moves one input to a plausible low and high value, with everything else as set, and shows "
               f"the change in the FY2026 saving. The changes are on the active basis: {st.session_state.get('scenario')}, "
               f"datacenters {'held' if _held(g) else 'following the fleet'} ({md_usd(t['spend_cut'])}B/yr)"
               + (f". The headline basis with datacenters following is {md_usd(t_follow['spend_cut'])}B/yr, the "
                  f"'Datacenters follow the fleet' row below" if _held(g) else "")
               + ". ★ marks a move of \\$10B or more.")
    rows = sensitivity_table(g, comps, kernels=_kern, follow=fol)
    WM_WHY = _wm_why()
    section(f"What moves the FY2026 saving ({bn(t['spend_cut'])}/yr): Helarctos levers against market data")
    body, chart = [], []
    for gname, items, gcode in (("Helarctos levers: what the architecture changes", [r for r in rows if r[2]], "h"),
                                ("Market and company data: not about Helarctos", [r for r in rows if not r[2]], "s")):
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
               f"about \\${t['spend_cut'] / 0.04 / 1000:.1f}T at 4%, \\${t['spend_cut'] / 0.06 / 1000:.1f}T at 6% and "
               f"\\${t['spend_cut'] / 0.10 / 1000:.1f}T at 10%. **The takeaway:** the answer depends on how much "
               "these firms spend on AI (the datacenter bucket and whether it follows the fleet, then the chip "
               "shares) more than on how large the Helarctos multiples are. The Helarctos levers only move it if "
               "they fall far below today's values.")


# ---- main ----------------------------------------------------------------------
st.title("AI Capex Efficiency")
st.caption("How much of the AI spend of the six biggest spenders (chips, the servers around them, datacenters and "
           "their power) the Helarctos architecture makes unnecessary. A tab-for-tab mirror of the workbook. The "
           "assumptions are adjustable: set the scenario, the levers and the datacenter switch in the sidebar. "
           f"Measurements as of {AS_OF}.")
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
                    f"{'; datacenters held' if _held(g) else ''})", f"{bn(_now['spend_cut'])}/yr",
                    delta=f"{'Current' if _active_opt else 'Optimized'} kernels: {bn(_other['spend_cut'])}/yr",
                    delta_color="off")
SIDEBAR_HEAD.caption(f"The scenario switch moves H3 from ×{_cur['train_speed']:.2f} to ×{_opt['train_speed']:.2f}, "
                     f"prompt speed from ×{_cur['prefill_speed']:.1f} to ×{_opt['prefill_speed']:.1f}, and the inference "
                     f"lever from ×{_cur['serving_compute_lever']:,.0f} to ×{_opt['serving_compute_lever']:,.0f}. In "
                     f"dollars that is {bn(_tc['spend_cut'])} to {bn(_to['spend_cut'])} (table on the Summary tab).")
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
