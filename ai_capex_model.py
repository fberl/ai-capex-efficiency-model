"""Single source of truth for the AI-Capex-Efficiency model.

Both the Excel generator (ai_capex_efficiency.py) and the Streamlit app (app.py)
import their DEFAULTS and LOGIC from here, so the spreadsheet and the web app
never drift.

Engine: a GPU is ~60% memory / ~40% compute by cost. Cutting memory 100x and
FLOPs 10x gives a residual cost of ~0.6%+4% -> ~22x cost-weighted reduction,
floored by the less-reduced component (compute).
"""

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


if __name__ == "__main__":
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
