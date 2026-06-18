"""AI Capex Efficiency — interactive investor view.

Every assumption is a slider, but the slider stays HIDDEN until you click its cell:
each assumption shows as a chip with its current value; click to reveal the slider
(a popover). Color convention mirrors the Excel sheet:
  🟡 assumption (a lever)    🟢 disclosed filing / market data    (derived = the table)

Shares all math with the Excel mirror via ai_capex_model.py (single source of truth).

Run locally:  uv run --with streamlit --with pandas streamlit run app.py
Deploy free:  push to GitHub -> share.streamlit.io  (needs requirements.txt)
"""

import streamlit as st
import pandas as pd

from ai_capex_model import (
    GLOBALS,
    COMPANIES,
    reduction_factor,
    compute_year,
    global_estimate,
)

st.set_page_config(page_title="AI Capex Efficiency", layout="wide")

# ---- display formatters --------------------------------------------------------
f_x = lambda v: f"{v:g}×"
f_pct = lambda v: f"{v:.0%}"
f_usd0 = lambda v: f"${v:,.0f}"
f_kw = lambda v: f"{v:g} kW"
f_kwh = lambda v: f"${v:.2f}/kWh"
f_yr = lambda v: f"{v:g} yr"


def cell(container, emoji, label, key, default, lo, hi, step, fmt, help=None):
    """A 'cell' that shows its current value; clicking it reveals the slider (popover).
    The value persists in st.session_state[key]. emoji: 🟡 assumption / 🟢 disclosed data.
    setdefault-then-keyed-slider avoids the value+key Streamlit warning."""
    st.session_state.setdefault(key, default)
    with container.popover(
        f"{emoji} {label}: {fmt(st.session_state[key])}", width="stretch"
    ):
        st.slider(label, lo, hi, step=step, key=key, help=help)
    return st.session_state[key]


st.title("AI Capex Efficiency")
st.caption(
    "Value of cutting AI memory 100× and FLOPs 10×. "
    "Click any 🟡 assumption or 🟢 data cell to reveal its slider. "
    "Same math as the Excel mirror (ai_capex_model.py)."
)

# year toggle, then a deferred slot for the headline (filled after the sliders compute it)
year = st.radio(
    "Year",
    ["fy25", "fy26"],
    horizontal=True,
    format_func=lambda y: "FY2025 (actual)" if y == "fy25" else "FY2026 (estimate)",
)
yi = 0 if year == "fy25" else 1
head = st.container()

# ---- GLOBAL ASSUMPTIONS (click a cell to reveal its slider) --------------------
g = dict(GLOBALS)
st.subheader("Assumptions — click a cell to reveal its slider")

st.markdown("**Architecture**")
a = st.columns(4)
g["mem_factor"] = cell(
    a[0],
    "🟡",
    "Memory reduction",
    "mem_factor",
    int(GLOBALS["mem_factor"]),
    1,
    200,
    1,
    f_x,
    help="1e-2 memory = 100× less (RNN O(1) state vs transformer O(T) KV cache).",
)
g["flop_factor"] = cell(
    a[1],
    "🟡",
    "FLOPs reduction",
    "flop_factor",
    int(GLOBALS["flop_factor"]),
    1,
    50,
    1,
    f_x,
    help="1e-1 FLOPs = 10× fewer.",
)
g["mem_share"] = cell(
    a[2],
    "🟡",
    "Memory share of GPU cost",
    "mem_share",
    float(GLOBALS["mem_share"]),
    0.0,
    1.0,
    0.05,
    f_pct,
    help="HBM + most CoWoS packaging → ~60% memory / ~40% compute (BOM).",
)
g["opex_reduction"] = cell(
    a[3],
    "🟡",
    "Opex / energy reduction",
    "opex_reduction",
    int(GLOBALS["opex_reduction"]),
    1,
    50,
    1,
    f_x,
    help="Energy ~ total FLOPs executed → 10× floor.",
)

st.markdown("**Cost & energy**")
b = st.columns(5)
g["gpu_cost"] = cell(
    b[0],
    "🟡",
    "Fully-loaded $/GPU",
    "gpu_cost",
    int(GLOBALS["gpu_cost"]),
    10000,
    100000,
    1000,
    f_usd0,
    help="B200-class GPU + share of server, NVLink, networking.",
)
g["wall_power_kw"] = cell(
    b[1],
    "🟡",
    "Wall power / GPU",
    "wall_power_kw",
    float(GLOBALS["wall_power_kw"]),
    0.5,
    4.0,
    0.1,
    f_kw,
    help="~1 kW TDP + node overhead × PUE 1.3.",
)
g["elec_rate"] = cell(
    b[2],
    "🟡",
    "Electricity rate",
    "elec_rate",
    float(GLOBALS["elec_rate"]),
    0.02,
    0.30,
    0.01,
    f_kwh,
    help="Datacenter wholesale; raise to 0.10–0.12 for grid colo.",
)
g["cooling_overhead"] = cell(
    b[3],
    "🟡",
    "Cooling / ops overhead",
    "cooling_overhead",
    float(GLOBALS["cooling_overhead"]),
    0.0,
    1.0,
    0.05,
    f_pct,
    help="Non-power running cost as a fraction of electricity.",
)
g["fleet_life_yr"] = cell(
    b[4],
    "🟡",
    "Fleet life",
    "fleet_life_yr",
    int(GLOBALS["fleet_life_yr"]),
    1,
    8,
    1,
    f_yr,
    help="AI-GPU depreciation life.",
)

st.markdown("**Finance & scope**")
d = st.columns(4)
g["discount_rate"] = cell(
    d[0],
    "🟡",
    "Discount rate",
    "discount_rate",
    float(GLOBALS["discount_rate"]),
    0.02,
    0.25,
    0.01,
    f_pct,
    help="Perpetuity capitalization: value = annual benefit / rate.",
)
g["dc_scale"] = cell(
    d[1],
    "🟡",
    "Datacenter scaling factor",
    "dc_scale",
    float(GLOBALS["dc_scale"]),
    0.0,
    1.0,
    0.05,
    f_pct,
    help="0 = only accelerator silicon shrinks (conservative). 1 = whole datacenter scales. ~0.7 ≈ net-AI breakeven.",
)
g["spacex_mktcap"] = cell(
    d[2],
    "🟢",
    "SpaceX market cap ($B)",
    "spacex_mktcap",
    int(GLOBALS["spacex_mktcap"]),
    500,
    3000,
    10,
    f_usd0,
    help="Market data (IPO 2026-06-12 ~$1.77T).",
)
g["named_share_of_global"] = cell(
    d[3],
    "🟡",
    "Named share of global AI capex",
    "named_share_of_global",
    float(GLOBALS["named_share_of_global"]),
    0.3,
    1.0,
    0.05,
    f_pct,
    help="GLOBAL estimate: the named firms' share of worldwide AI capex; the rest (other clouds, China, neoclouds, xAI, sovereign, enterprise) is grossed up pro-rata. Clearly an estimate.",
)

# ---- PER-COMPANY ASSUMPTIONS (selected year; click a company to edit) ----------
st.markdown(
    f"**Per company — {'FY2025 (actual)' if yi == 0 else 'FY2026 (estimate)'}**  ·  "
    "click a company to edit its capex, shares & AI revenue"
)
cap_emoji = (
    "🟢" if yi == 0 else "🟡"
)  # FY25 total capex is disclosed; FY26 is an estimate
comps = [dict(c) for c in COMPANIES]
cols = st.columns(len(comps))
for col, c in zip(cols, comps):
    name = c["name"]
    t, infra, server, accel = c[year]
    p = f"{name}_{year}_"
    st.session_state.setdefault(p + "total", float(t))
    st.session_state.setdefault(p + "infra", float(infra))
    st.session_state.setdefault(p + "server", float(server))
    st.session_state.setdefault(p + "accel", float(accel))
    st.session_state.setdefault(p + "rev", float(c["ai_rev"][yi]))
    with col.popover(name, width="stretch"):
        st.slider(
            f"{cap_emoji} Total capex ($B)", 0.0, 300.0, step=1.0, key=p + "total"
        )
        st.slider("🟡 Infra / DC share", 0.0, 1.0, step=0.01, key=p + "infra")
        st.slider("🟡 Server share", 0.0, 1.0, step=0.01, key=p + "server")
        st.slider("🟡 Accelerator share", 0.0, 1.0, step=0.01, key=p + "accel")
        st.slider("🟡 AI revenue ($B)", 0.0, 200.0, step=0.5, key=p + "rev")
    c[year] = (
        st.session_state[p + "total"],
        st.session_state[p + "infra"],
        st.session_state[p + "server"],
        st.session_state[p + "accel"],
    )
    air = list(c["ai_rev"])
    air[yi] = st.session_state[p + "rev"]
    c["ai_rev"] = tuple(air)

# ---- compute -------------------------------------------------------------------
R = reduction_factor(g)
rows, tot = compute_year(g, comps, year)
glob = global_estimate(tot, g)

# ---- headline metrics (rendered into the deferred slot above the assumptions) ---
with head:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Cost-weighted reduction", f"{R:.1f}×")
    c2.metric(f"Net AI now ({len(comps)} cos)", f"${tot['net_now']:,.0f}B/yr")
    c3.metric(
        "Net AI with our arch",
        f"${tot['net_arch']:,.0f}B/yr",
        delta=f"{tot['spend_cut']:,.0f} spend cut",
    )
    c4.metric("% of AI spend cut", f"{tot['pct_cut']:.0%}")
    verdict = (
        "AI flips **profitable** at these settings."
        if tot["net_arch"] > 0
        else f"AI burn shrinks from **${-tot['net_now']:,.0f}B** to **${-tot['net_arch']:,.0f}B** per year."
    )
    st.markdown(f"**{verdict}**")
    st.caption(
        f"These {len(comps)} are a floor. GLOBAL estimate (named ≈ {glob['named_share']:.0%} of world AI capex): "
        f"~${glob['spend_cut']:,.0f}B/yr spend cut → ~${glob['capitalized'] / 1000:.1f}T capitalized — "
        "clearly an estimate; the rest is other clouds, China, neoclouds, xAI & sovereign AI."
    )

# ---- net AI table --------------------------------------------------------------
st.subheader(
    f"Net AI economics — {'FY2025' if yi == 0 else 'FY2026'} "
    "(cash basis: AI revenue − AI capex − AI opex)"
)
disp = pd.DataFrame(rows + [tot])[
    [
        "name",
        "ai_rev",
        "ai_capex",
        "ai_opex",
        "net_now",
        "spend_cut",
        "net_arch",
        "pct_cut",
    ]
].rename(
    columns={
        "name": "Company",
        "ai_rev": "AI revenue",
        "ai_capex": "AI capex",
        "ai_opex": "AI opex",
        "net_now": "Net AI NOW",
        "spend_cut": "Spend cut",
        "net_arch": "Net AI W/ ARCH",
        "pct_cut": "% spend cut",
    }
)
st.dataframe(
    disp.style.format(
        {
            "AI revenue": "${:,.1f}B",
            "AI capex": "${:,.1f}B",
            "AI opex": "${:,.1f}B",
            "Net AI NOW": "${:,.1f}B",
            "Spend cut": "${:,.1f}B",
            "Net AI W/ ARCH": "${:,.1f}B",
            "% spend cut": "{:.0%}",
        }
    ),
    width="stretch",
    hide_index=True,
)

# ---- charts --------------------------------------------------------------------
left, right = st.columns(2)
with left:
    st.subheader("Net AI: now vs with our architecture")
    bar = pd.DataFrame(
        {
            "Net AI now": [r["net_now"] for r in rows],
            "Net AI w/ arch": [r["net_arch"] for r in rows],
        },
        index=[r["name"] for r in rows],
    )
    st.bar_chart(bar)

with right:
    st.subheader(f"{len(comps)}-co net AI vs datacenter-scaling factor")
    curve = []
    for i in range(0, 21):
        gg = dict(g)
        gg["dc_scale"] = i / 20
        _, t = compute_year(gg, comps, year)
        curve.append(
            {"dc_scale": gg["dc_scale"], "Net AI w/ arch ($B/yr)": t["net_arch"]}
        )
    st.line_chart(pd.DataFrame(curve).set_index("dc_scale"))

st.caption(
    "🟡 assumption · 🟢 disclosed data · table = derived. "
    "Spend cut = capex avoided + opex saved. Net AI is cash basis (capex not depreciated). "
    "AI revenue: Microsoft $37B & Amazon $15B run-rates disclosed; Google/Meta/Oracle/SpaceX estimated "
    "(Meta's real payoff is indirect ad-uplift). Capitalized value & per-company detail are on the Excel mirror."
)
