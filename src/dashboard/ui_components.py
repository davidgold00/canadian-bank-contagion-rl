"""
Premium UI components for the Canadian Bank Contagion Command Center.

Design language: professional financial terminal — dark accent palette,
monospace data, color-coded risk indicators, clear signal typography.
"""

import html

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Shared colour tokens ────────────────────────────────────────────────────
PALETTE = {
    "bg": "#0f1923",
    "surface": "#16202d",
    "card": "#1c2a38",
    "border": "#2a3a4a",
    "border_accent": "#334455",
    "ink": "#e8edf2",
    "muted": "#7a91a6",
    "blue": "#1e88e5",
    "blue_light": "#42a5f5",
    "green": "#00c853",
    "green_muted": "#1b5e20",
    "amber": "#ffb300",
    "amber_muted": "#ff6f00",
    "red": "#f44336",
    "red_muted": "#b71c1c",
    "teal": "#00bcd4",
}

PLOTLY_TEMPLATE = dict(
    layout=go.Layout(
        paper_bgcolor="#16202d",
        plot_bgcolor="#0f1923",
        font=dict(family="'JetBrains Mono', 'SF Mono', 'Fira Code', monospace", color="#e8edf2", size=12),
        # NOTE: no title= here — to_plotly_json() would put "title" in the dict, conflicting
        # with explicit title= kwargs in update_layout() calls across pages.
        # Title font inherits from the global font above (color="#e8edf2").
        xaxis=dict(
            gridcolor="#2a3a4a",
            linecolor="#2a3a4a",
            tickfont=dict(color="#7a91a6"),
            title_font=dict(color="#7a91a6"),
        ),
        yaxis=dict(
            gridcolor="#2a3a4a",
            linecolor="#2a3a4a",
            tickfont=dict(color="#7a91a6"),
            title_font=dict(color="#7a91a6"),
        ),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#e8edf2")),
        colorway=["#1e88e5", "#00c853", "#ffb300", "#f44336", "#00bcd4", "#e040fb"],
        margin=dict(l=20, r=20, t=50, b=20),
    )
)


def _apply_plotly_template(fig: go.Figure, height: int = 430) -> go.Figure:
    fig.update_layout(
        **{k: v for k, v in PLOTLY_TEMPLATE["layout"].to_plotly_json().items()},
        height=height,
    )
    return fig


def apply_dashboard_style() -> None:
    st.markdown(
        """
        <style>
        /* ── Fonts ─────────────────────────────── */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

        /* ── Root tokens ──────────────────────── */
        :root {
            --bg:            #0f1923;
            --surface:       #16202d;
            --card:          #1c2a38;
            --border:        #2a3a4a;
            --border-accent: #334455;
            --ink:           #e8edf2;
            --muted:         #7a91a6;
            --blue:          #1e88e5;
            --blue-light:    #42a5f5;
            --green:         #00c853;
            --amber:         #ffb300;
            --red:           #f44336;
            --teal:          #00bcd4;
            --radius:        8px;
        }

        /* ── Page & body ────────────────────────── */
        html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
            background-color: var(--bg) !important;
            color: var(--ink) !important;
            font-family: 'Inter', ui-sans-serif, system-ui, sans-serif !important;
        }

        /* Hide Streamlit hamburger/branding but keep toolbar for sidebar toggle */
        #MainMenu, footer { display: none !important; }
        [data-testid="stToolbar"] { display: none !important; }

        /* ── Sidebar ─────────────────────────── */
        [data-testid="stSidebar"] {
            background: var(--surface) !important;
            border-right: 1px solid var(--border) !important;
        }
        /* Color text inside sidebar but NOT the SVG icons used for collapse button */
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] span,
        [data-testid="stSidebar"] div:not([data-testid="stSidebarCollapseButton"]) {
            color: var(--ink) !important;
        }
        [data-testid="stSidebar"] [data-testid="stMarkdown"] p {
            color: var(--muted) !important;
        }

        /* ── Sidebar navigation links ──────────── */
        [data-testid="stSidebarNavLink"] {
            border-radius: 6px !important;
            padding: 6px 10px !important;
            margin: 2px 4px !important;
            color: var(--muted) !important;
            font-size: 0.88rem !important;
            font-weight: 500 !important;
            transition: background 0.15s, color 0.15s !important;
        }
        [data-testid="stSidebarNavLink"]:hover {
            background: var(--card) !important;
            color: var(--ink) !important;
        }
        [data-testid="stSidebarNavLink"][aria-current="page"],
        [data-testid="stSidebarNavLink"][aria-selected="true"] {
            background: var(--card) !important;
            color: var(--blue-light) !important;
            border-left: 3px solid var(--blue-light) !important;
        }

        /* ── Sidebar collapse / expand button ─── */
        /* The button INSIDE the sidebar to collapse it */
        [data-testid="stSidebarCollapseButton"] {
            display: flex !important;
            opacity: 1 !important;
            visibility: visible !important;
        }
        [data-testid="stSidebarCollapseButton"] button {
            background: transparent !important;
            border: 1px solid var(--border) !important;
            color: var(--muted) !important;
            border-radius: 6px !important;
        }
        [data-testid="stSidebarCollapseButton"] button:hover {
            background: var(--card) !important;
            color: var(--ink) !important;
        }
        [data-testid="stSidebarCollapseButton"] svg {
            fill: var(--muted) !important;
            stroke: var(--muted) !important;
        }

        /* The floating button OUTSIDE the sidebar to expand it when collapsed */
        [data-testid="stSidebarCollapsedControl"] {
            display: block !important;
            opacity: 1 !important;
            visibility: visible !important;
            z-index: 999999 !important;
        }
        [data-testid="stSidebarCollapsedControl"] button {
            background: var(--surface) !important;
            border: 1px solid var(--border-accent) !important;
            color: var(--ink) !important;
            border-radius: 0 6px 6px 0 !important;
            box-shadow: 3px 0 12px rgba(0,0,0,0.4) !important;
        }
        [data-testid="stSidebarCollapsedControl"] button:hover {
            background: var(--card) !important;
            border-color: var(--blue) !important;
        }
        [data-testid="stSidebarCollapsedControl"] svg {
            fill: var(--ink) !important;
            stroke: var(--ink) !important;
            color: var(--ink) !important;
        }

        /* ── Main block container ─────────────── */
        .block-container {
            padding-top: 2rem !important;
            padding-bottom: 3rem !important;
            max-width: 1440px !important;
            background: var(--bg) !important;
        }

        /* ── Typography ────────────────────────── */
        h1 {
            font-size: 2.25rem !important;
            font-weight: 700 !important;
            letter-spacing: 0 !important;
            line-height: 1.12 !important;
            color: var(--ink) !important;
            margin-bottom: 0.2rem !important;
        }
        h2 { font-weight: 600 !important; color: var(--ink) !important; }
        h3 { font-weight: 600 !important; color: var(--ink) !important; }
        h4 { font-weight: 500 !important; color: var(--ink) !important; margin-bottom: 0.3rem !important; }
        p, li { color: #c5d1db !important; line-height: 1.55 !important; }
        .stMarkdown p { color: #c5d1db !important; }

        /* ── Metric cards ──────────────────────── */
        div[data-testid="stMetric"] {
            background: var(--card) !important;
            border: 1px solid var(--border) !important;
            border-radius: var(--radius) !important;
            padding: 1rem 1.1rem !important;
            box-shadow: 0 2px 12px rgba(0,0,0,0.3) !important;
            transition: border-color 0.2s !important;
        }
        div[data-testid="stMetric"]:hover {
            border-color: var(--border-accent) !important;
        }
        div[data-testid="stMetricLabel"] p {
            color: var(--muted) !important;
            font-size: 0.78rem !important;
            text-transform: uppercase !important;
            letter-spacing: 0.06em !important;
            font-weight: 500 !important;
        }
        div[data-testid="stMetricValue"] {
            color: var(--ink) !important;
            font-size: 1.55rem !important;
            font-family: 'JetBrains Mono', monospace !important;
            font-weight: 500 !important;
        }
        div[data-testid="stMetricDelta"] {
            font-family: 'JetBrains Mono', monospace !important;
            font-size: 0.85rem !important;
        }

        /* ── Tabs ──────────────────────────────── */
        button[data-baseweb="tab"] {
            font-size: 0.88rem !important;
            color: var(--muted) !important;
            font-weight: 500 !important;
            border-radius: 6px 6px 0 0 !important;
            background: transparent !important;
            border: none !important;
        }
        button[data-baseweb="tab"][aria-selected="true"] {
            color: var(--blue-light) !important;
            border-bottom: 2px solid var(--blue-light) !important;
        }
        div[data-baseweb="tab-panel"] { padding-top: 1rem !important; }

        /* ── Selectbox / slider / checkbox ──────── */
        div[data-testid="stSelectbox"] label,
        div[data-testid="stSlider"] label,
        div[data-testid="stCheckbox"] label {
            color: var(--muted) !important;
            font-size: 0.82rem !important;
            text-transform: uppercase !important;
            letter-spacing: 0.04em !important;
        }
        [data-baseweb="select"] div {
            background: var(--card) !important;
            border-color: var(--border) !important;
            color: var(--ink) !important;
        }

        /* ── Dataframes ────────────────────────── */
        [data-testid="stDataFrame"] {
            border-radius: var(--radius) !important;
            overflow: hidden !important;
        }
        [data-testid="stDataFrame"] * {
            max-width: 100%;
        }
        [data-testid="stDataFrame"] th {
            background: var(--surface) !important;
            color: var(--muted) !important;
            font-size: 0.78rem !important;
            text-transform: uppercase !important;
            letter-spacing: 0.05em !important;
        }
        [data-testid="stDataFrame"] td {
            color: var(--ink) !important;
            font-family: 'JetBrains Mono', monospace !important;
            font-size: 0.86rem !important;
        }

        /* ── Expanders ─────────────────────────── */
        details { border-color: var(--border) !important; }
        details summary {
            color: var(--blue-light) !important;
            font-weight: 500 !important;
        }

        /* ── Dividers ──────────────────────────── */
        hr { border-color: var(--border) !important; opacity: 0.6 !important; }

        /* ── Custom components (defined below) ─── */
        .cc-card {
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            padding: 1.1rem 1.2rem;
            margin: 0.4rem 0 0.9rem;
            box-shadow: 0 2px 10px rgba(0,0,0,0.25);
        }
        .cc-card h4 {
            margin: 0 0 0.5rem;
            font-size: 0.95rem;
            font-weight: 600;
            color: var(--ink);
        }
        .cc-card p { color: #c5d1db; font-size: 0.9rem; margin: 0; }

        .cc-card.success { border-left: 4px solid var(--green); background: #0d2318; }
        .cc-card.warning { border-left: 4px solid var(--amber); background: #1f1700; }
        .cc-card.danger  { border-left: 4px solid var(--red);   background: #1f0a08; }
        .cc-card.info    { border-left: 4px solid var(--blue);  background: #0a1929; }
        .cc-card.teal    { border-left: 4px solid var(--teal);  background: #001f26; }

        .cc-business-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.75rem;
            margin: 0.6rem 0 1rem;
        }
        .cc-business-card {
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            padding: 0.9rem 1rem;
            min-height: 132px;
        }
        .cc-business-card .cc-kicker {
            color: var(--teal);
            font-size: 0.68rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 0.4rem;
        }
        .cc-business-card h4 {
            font-size: 0.92rem !important;
            font-weight: 700 !important;
            margin: 0 0 0.35rem !important;
            color: var(--ink) !important;
        }
        .cc-business-card p {
            color: #c5d1db !important;
            font-size: 0.82rem !important;
            line-height: 1.45 !important;
            margin: 0 !important;
        }

        /* Signal badges */
        .sig-buy    { display:inline-block; padding:3px 10px; border-radius:999px; background:#0d2318; color:var(--green);  border:1px solid var(--green);  font-size:0.78rem; font-weight:700; letter-spacing:0.05em; }
        .sig-hold   { display:inline-block; padding:3px 10px; border-radius:999px; background:#1f1700; color:var(--amber);  border:1px solid var(--amber);  font-size:0.78rem; font-weight:700; letter-spacing:0.05em; }
        .sig-reduce { display:inline-block; padding:3px 10px; border-radius:999px; background:#1f0a08; color:var(--red);    border:1px solid var(--red);    font-size:0.78rem; font-weight:700; letter-spacing:0.05em; }

        /* Source pills */
        .cc-pill {
            display: inline-block;
            padding: 4px 10px;
            border: 1px solid var(--border);
            border-radius: 999px;
            background: var(--surface);
            color: var(--muted);
            font-size: 0.78rem;
            margin-right: 6px;
            margin-bottom: 6px;
        }

        /* Action list */
        .cc-action-item {
            display: flex;
            align-items: flex-start;
            gap: 0.6rem;
            margin-bottom: 0.55rem;
            font-size: 0.88rem;
            color: #c5d1db;
        }
        .cc-action-bullet {
            flex-shrink: 0;
            width: 6px;
            height: 6px;
            border-radius: 50%;
            background: var(--blue-light);
            margin-top: 6px;
        }

        /* Regime banner */
        .regime-banner {
            border-radius: var(--radius);
            padding: 1rem 1.4rem;
            margin: 1rem 0;
            display: flex;
            align-items: center;
            gap: 1rem;
            font-size: 0.95rem;
        }
        .regime-banner .regime-label {
            font-size: 1.4rem;
            font-weight: 700;
            font-family: 'JetBrains Mono', monospace;
            white-space: nowrap;
        }

        /* Small text / caption */
        .cc-caption { color: var(--muted); font-size: 0.8rem; margin-top: 0.6rem; }

        /* Conviction stars */
        .cv-star-filled  { color: var(--amber); }
        .cv-star-empty   { color: var(--border-accent); }

        .cc-decision-grid,
        .cc-intro-grid {
            display: grid;
            grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
            gap: 0;
        }
        .cc-decision-grid {
            margin: 0.5rem 0 1rem;
            overflow: hidden;
        }
        .cc-intro-grid {
            margin: 0 0 1.2rem;
            overflow: hidden;
        }
        .cc-grid-cell {
            min-width: 0;
            overflow-wrap: anywhere;
        }

        @media (max-width: 900px) {
            .block-container {
                padding-left: 1rem !important;
                padding-right: 1rem !important;
            }
            h1 {
                font-size: 1.65rem !important;
            }
            .cc-decision-grid,
            .cc-intro-grid {
                grid-template-columns: 1fr !important;
            }
            .cc-business-grid {
                grid-template-columns: 1fr !important;
            }
            .cc-decision-grid .cc-grid-cell + .cc-grid-cell,
            .cc-intro-grid .cc-grid-cell + .cc-grid-cell {
                border-left: 0 !important;
                border-top: 1px solid var(--border) !important;
            }
            .regime-banner {
                align-items: flex-start;
                flex-direction: column;
            }
        }

        </style>
        """,
        unsafe_allow_html=True,
    )


# ── Header components ───────────────────────────────────────────────────────

def analyst_header(title: str, subtitle: str, date_text: str | None = None, source_text: str | None = None) -> None:
    st.title(title)
    st.markdown(f"<p style='font-size:1.05rem;color:#9bb4c8;margin-top:-0.4rem'>{subtitle}</p>", unsafe_allow_html=True)
    pills = []
    if date_text:
        pills.append(f"<span class='cc-pill'>Data through {date_text}</span>")
    if source_text:
        pills.append(f"<span class='cc-pill'>{source_text}</span>")
    if pills:
        st.markdown(" ".join(pills), unsafe_allow_html=True)


def page_header(title: str, subtitle: str, why_it_matters: str, how_to_read: str) -> None:
    st.title(title)
    st.markdown(f"<p style='font-size:1.05rem;color:#9bb4c8;margin-top:-0.4rem'>{subtitle}</p>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        insight_card("Why This Matters", why_it_matters, status="teal")
    with c2:
        insight_card("How to Read This Page", how_to_read, status="info")
    st.divider()


# ── Card components ─────────────────────────────────────────────────────────

def insight_card(title: str, body: str, status: str = "info") -> None:
    status_class = {
        "success": "success",
        "warning": "warning",
        "danger": "danger",
        "info": "info",
        "teal": "teal",
    }.get(status, "info")
    st.markdown(
        f"""<div class="cc-card {status_class}">
              <h4>{title}</h4>
              <p>{body}</p>
            </div>""",
        unsafe_allow_html=True,
    )


DEFAULT_BUSINESS_VALUE_POINTS = [
    (
        "Mandate Fit",
        "Designed for Canadian bank exposure oversight, not for replacing a high-growth equity index.",
        "Purpose",
    ),
    (
        "Risk Governance",
        "Turns noisy market data into auditable regime, exposure, cash, and stress-test decisions.",
        "Control",
    ),
    (
        "Concentration Awareness",
        "Shows when multiple bank holdings behave like one shared macro risk bucket.",
        "Diversification",
    ),
    (
        "Committee-Ready Evidence",
        "Explains what changed, why it matters, and which monitoring trigger should prompt action.",
        "Communication",
    ),
]


def business_value_panel(
    title: str = "Business Value Lens",
    intro: str | None = None,
    points: list[tuple[str, str, str]] | None = None,
    tone: str = "teal",
) -> None:
    """Show why the system matters to a business user beyond raw return ranking."""
    if intro:
        insight_card(title, intro, status=tone)

    cells = []
    for heading, body, kicker in points or DEFAULT_BUSINESS_VALUE_POINTS:
        cells.append(
            "<div class='cc-business-card'>"
            f"<div class='cc-kicker'>{html.escape(kicker)}</div>"
            f"<h4>{html.escape(heading)}</h4>"
            f"<p>{html.escape(body)}</p>"
            "</div>"
        )
    st.markdown("<div class='cc-business-grid'>" + "".join(cells) + "</div>", unsafe_allow_html=True)


def mandate_fit_table() -> None:
    """Explain when the dashboard should and should not be judged by total return alone."""
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Business Question": "Can this beat Nasdaq over 10 years?",
                    "Best Tool": "Passive benchmark comparison",
                    "How to Interpret": "If Nasdaq wins, it was the better growth allocation for that period.",
                },
                {
                    "Business Question": "Can we control Canadian bank concentration risk?",
                    "Best Tool": "Contagion score, network graph, CVaR, stress tests",
                    "How to Interpret": "This is where the project creates most of its business value.",
                },
                {
                    "Business Question": "Should we add, hold, trim, or hedge bank exposure today?",
                    "Best Tool": "Investment Decision Center",
                    "How to Interpret": "Use regime, bank stress, signal strength, and CVaR constraints together.",
                },
                {
                    "Business Question": "Can a risk committee understand the recommendation?",
                    "Best Tool": "Decision memos and plain-English guides",
                    "How to Interpret": "A useful model must be explainable, not just profitable in hindsight.",
                },
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )


def action_list(title: str, actions: list[str]) -> None:
    items_html = "".join(
        f"<div class='cc-action-item'><div class='cc-action-bullet'></div><span>{a}</span></div>"
        for a in actions
    )
    st.markdown(
        f"""<div class="cc-card info">
              <h4>{title}</h4>
              {items_html}
            </div>""",
        unsafe_allow_html=True,
    )


def regime_banner(label: str, summary: str, score: float, tone: str) -> None:
    bg_map = {
        "success": "#0d2318",
        "warning": "#1f1700",
        "danger": "#1f0a08",
        "info": "#0a1929",
    }
    border_map = {
        "success": "#00c853",
        "warning": "#ffb300",
        "danger": "#f44336",
        "info": "#1e88e5",
    }
    color_map = {
        "success": "#00c853",
        "warning": "#ffb300",
        "danger": "#f44336",
        "info": "#1e88e5",
    }
    bg = bg_map.get(tone, bg_map["info"])
    border = border_map.get(tone, border_map["info"])
    color = color_map.get(tone, color_map["info"])
    st.markdown(
        f"""<div class="regime-banner" style="background:{bg};border:1px solid {border};">
              <div class="regime-label" style="color:{color};">{score:.0f}/100</div>
              <div>
                <div style="font-weight:700;color:{color};margin-bottom:3px;">{label} Risk Regime</div>
                <div style="color:#c5d1db;font-size:0.88rem;">{summary}</div>
              </div>
            </div>""",
        unsafe_allow_html=True,
    )


def signal_badge(signal: str) -> str:
    """Return HTML badge for BUY / HOLD / REDUCE inline use."""
    if signal.upper() in ("BUY", "▲ BUY"):
        return "<span class='sig-buy'>▲ BUY</span>"
    if signal.upper() in ("REDUCE", "▼ REDUCE", "SELL"):
        return "<span class='sig-reduce'>▼ REDUCE</span>"
    return "<span class='sig-hold'>◆ HOLD</span>"


def conviction_stars_html(n: int) -> str:
    filled = "★" * n
    empty = "☆" * (5 - n)
    return f"<span class='cv-star-filled'>{filled}</span><span class='cv-star-empty'>{empty}</span>"


def signal_table(signals_df: pd.DataFrame) -> None:
    """Render the investment signals table with native Streamlit components."""
    if signals_df.empty:
        st.info("No signal data available.")
        return

    def readout(row: pd.Series) -> str:
        signal = str(row.get("Signal", "HOLD"))
        bank = row.get("Bank", "")
        delta = row.get("Weight Delta", 0.0)
        stress = row.get("Node Stress", 0.0)
        if signal == "BUY":
            return f"Add only if portfolio risk budget can absorb a {delta:+.1%} tilt; monitor stress above {max(60, stress + 10):.0f}."
        if signal == "REDUCE":
            return f"Trim or hedge first; bank-level stress is {stress:.0f}/100 and target is {row.get('Target Weight', 0):.1%}."
        return f"Hold {bank}; wait for a clearer score break or stress deterioration before changing weight."

    display = pd.DataFrame(
        {
            "Ticker": signals_df["Bank"],
            "Bank": signals_df["Name"],
            "Signal": signals_df["Signal"],
            "Conviction": signals_df["Conviction"].map(lambda x: "★" * int(x) + "☆" * (5 - int(x))),
            "Score": signals_df["Composite Score"].round(1),
            "Stress": signals_df["Node Stress"].round(1),
            "21D Return": signals_df["21D Return"].map(lambda x: f"{x:+.1%}" if pd.notna(x) else "N/A"),
            "Target Wt": signals_df["Target Weight"].map(lambda x: f"{x:.1%}" if pd.notna(x) else "N/A"),
            "Δ Wt": signals_df["Weight Delta"].map(lambda x: f"{x:+.1%}" if pd.notna(x) else "N/A"),
            "Decision Readout": signals_df.apply(readout, axis=1),
        }
    )

    st.dataframe(
        display,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Ticker": st.column_config.TextColumn("Ticker", width="small"),
            "Bank": st.column_config.TextColumn("Bank", width="medium"),
            "Signal": st.column_config.TextColumn("Signal", width="small"),
            "Conviction": st.column_config.TextColumn("Conviction", width="small"),
            "Score": st.column_config.ProgressColumn("Score", min_value=0, max_value=100, format="%.1f"),
            "Stress": st.column_config.ProgressColumn("Stress", min_value=0, max_value=100, format="%.1f"),
            "21D Return": st.column_config.TextColumn("21D Ret", width="small"),
            "Target Wt": st.column_config.TextColumn("Target Wt", width="small"),
            "Δ Wt": st.column_config.TextColumn("Δ Wt", width="small"),
            "Decision Readout": st.column_config.TextColumn("Decision Readout", width="large"),
        },
    )


# ── Risk badge ──────────────────────────────────────────────────────────────

def risk_badge(score: float) -> None:
    if score >= 80:
        st.error(f"Severe Stress — {score:.1f}/100")
    elif score >= 60:
        st.warning(f"High Stress — {score:.1f}/100")
    elif score >= 30:
        st.info(f"Moderate Stress — {score:.1f}/100")
    else:
        st.success(f"Low Stress — {score:.1f}/100")


# ── Plotly chart helpers ─────────────────────────────────────────────────────

def styled_line(
    df: pd.DataFrame,
    y: list[str] | str,
    title: str,
    yaxis_title: str = "",
    yaxis_format: str = "",
    height: int = 420,
) -> go.Figure:
    ys = [y] if isinstance(y, str) else y
    fig = go.Figure()
    colors = [PALETTE["blue"], PALETTE["green"], PALETTE["amber"], PALETTE["red"], PALETTE["teal"]]
    for i, col in enumerate(ys):
        if col in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df[col], mode="lines", name=col,
                line=dict(color=colors[i % len(colors)], width=2),
            ))
    fig.update_layout(
        title=title,
        yaxis_title=yaxis_title,
        yaxis_tickformat=yaxis_format,
        **PLOTLY_TEMPLATE["layout"].to_plotly_json(),
        height=height,
    )
    return fig


def styled_bar(
    x, y, title: str, orientation: str = "v",
    color: str = "#1e88e5", height: int = 420,
) -> go.Figure:
    if orientation == "h":
        fig = go.Figure(go.Bar(x=x, y=y, orientation="h", marker_color=color))
    else:
        fig = go.Figure(go.Bar(x=x, y=y, marker_color=color))
    fig.update_layout(
        title=title,
        **PLOTLY_TEMPLATE["layout"].to_plotly_json(),
        height=height,
    )
    return fig


def styled_heatmap(matrix: pd.DataFrame, title: str, height: int = 500) -> go.Figure:
    fig = px.imshow(
        matrix,
        text_auto=".2f",
        aspect="auto",
        title=title,
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
    )
    fig.update_layout(
        **PLOTLY_TEMPLATE["layout"].to_plotly_json(),
        height=height,
    )
    return fig


def plot_time_series(df: pd.DataFrame, y: str, title: str, explanation: str | None = None) -> None:
    fig = styled_line(df, y, title)
    st.plotly_chart(fig, use_container_width=True)
    if explanation:
        st.markdown(f"<p class='cc-caption'>{explanation}</p>", unsafe_allow_html=True)


def plot_heatmap(matrix: pd.DataFrame, title: str, explanation: str | None = None) -> None:
    fig = styled_heatmap(matrix, title)
    st.plotly_chart(fig, use_container_width=True)
    if explanation:
        st.markdown(f"<p class='cc-caption'>{explanation}</p>", unsafe_allow_html=True)


# ── Interpretation helpers ───────────────────────────────────────────────────

def decision_callout(plain_english: str, action: str, tone: str = "info") -> None:
    """Two-column card: plain-English meaning on the left, concrete action on the right."""
    border_map = {"success": "var(--green)", "warning": "var(--amber)", "danger": "var(--red)", "info": "var(--blue)", "teal": "var(--teal)"}
    bg_map = {"success": "#0d2318", "warning": "#1f1700", "danger": "#1f0a08", "info": "#0a1929", "teal": "#001f26"}
    border = border_map.get(tone, border_map["info"])
    bg = bg_map.get(tone, bg_map["info"])
    st.markdown(
        f"""<div class="cc-decision-grid" style="background:{bg};border:1px solid {border};border-radius:var(--radius);">
              <div class="cc-grid-cell" style="padding:1rem 1.2rem">
                <div style="font-size:0.7rem;text-transform:uppercase;letter-spacing:0.09em;
                            color:var(--muted);margin-bottom:6px;font-weight:600">What This Means</div>
                <div style="color:#c5d1db;font-size:0.9rem;line-height:1.5">{plain_english}</div>
              </div>
              <div class="cc-grid-cell" style="padding:1rem 1.2rem;border-left:1px solid {border}">
                <div style="font-size:0.7rem;text-transform:uppercase;letter-spacing:0.09em;
                            color:var(--muted);margin-bottom:6px;font-weight:600">Decision / Action</div>
                <div style="color:#e8edf2;font-size:0.9rem;line-height:1.5;font-weight:500">{action}</div>
              </div>
            </div>""",
        unsafe_allow_html=True,
    )


COMMON_PLAIN_ENGLISH_GUIDE = [
    (
        "Contagion risk",
        "The chance that stress in one bank or market channel spills into the rest of the banking group.",
    ),
    (
        "Volatility",
        "How much prices are moving around. Higher volatility means less certainty about near-term portfolio value.",
    ),
    (
        "Correlation",
        "How closely assets move together. High bank correlation means owning several banks may still behave like one large bank trade.",
    ),
    (
        "Drawdown",
        "The loss from a recent high point to the later low point. It captures the pain an investor had to sit through.",
    ),
    (
        "Sharpe ratio",
        "Return per unit of volatility. Higher is better, but it does not fully describe tail losses.",
    ),
    (
        "CVaR",
        "Conditional Value at Risk, or expected shortfall: the average loss in the worst tail of outcomes. Lower is better.",
    ),
    (
        "Turnover",
        "How much the portfolio trades. Higher turnover can make a strategy look good before costs but weaker after costs.",
    ),
    (
        "Cash weight",
        "The defensive portion held out of risky assets. Cash can reduce losses but can also drag returns in strong markets.",
    ),
]


def plain_english_expander(
    title: str = "Optional Plain-English Guide",
    items: list[tuple[str, str]] | None = None,
    expanded: bool = False,
) -> None:
    """Render an optional glossary for nontechnical finance readers."""
    guide_items = items if items is not None else COMMON_PLAIN_ENGLISH_GUIDE
    with st.expander(title, expanded=expanded):
        for term, explanation in guide_items:
            st.markdown(f"- **{term}**: {explanation}")


def page_intro(why: str, how: str) -> None:
    """Compact two-column intro card shown at the top of each analysis page."""
    st.markdown(
        f"""<div class="cc-intro-grid" style="background:var(--card);border:1px solid var(--border);border-radius:var(--radius);">
              <div class="cc-grid-cell" style="padding:0.85rem 1.1rem;border-left:3px solid var(--teal)">
                <div style="font-size:0.7rem;text-transform:uppercase;letter-spacing:0.09em;
                            color:var(--teal);margin-bottom:5px;font-weight:600">Why This Page Exists</div>
                <div style="color:#c5d1db;font-size:0.875rem;line-height:1.5">{why}</div>
              </div>
              <div class="cc-grid-cell" style="padding:0.85rem 1.1rem;border-left:3px solid var(--blue)">
                <div style="font-size:0.7rem;text-transform:uppercase;letter-spacing:0.09em;
                            color:var(--blue-light);margin-bottom:5px;font-weight:600">How To Read It</div>
                <div style="color:#c5d1db;font-size:0.875rem;line-height:1.5">{how}</div>
              </div>
            </div>""",
        unsafe_allow_html=True,
    )
    plain_english_expander()


def decision_memo(title: str, rows: list[dict[str, str]], tone: str = "info") -> None:
    """Render a compact observation-to-action table."""
    if not rows:
        return
    insight_card(title, "Each line converts a model observation into a decision implication and a monitoring trigger.", status=tone)
    st.dataframe(
        pd.DataFrame(rows),
        use_container_width=True,
        hide_index=True,
        column_config={
            "Observation": st.column_config.TextColumn("Observation", width="medium"),
            "Decision Implication": st.column_config.TextColumn("Decision Implication", width="large"),
            "Monitoring Trigger": st.column_config.TextColumn("Monitoring Trigger", width="large"),
        },
    )


def interpretation_box(title: str, bullets: list[str]) -> None:
    items = "".join(
        f"<div class='cc-action-item'><div class='cc-action-bullet'></div><span>{b}</span></div>"
        for b in bullets
    )
    st.markdown(
        f"""<div class="cc-card teal">
              <h4>{title}</h4>
              {items}
            </div>""",
        unsafe_allow_html=True,
    )


def explain_metric(label: str, value: str, explanation: str, delta=None) -> None:
    st.metric(label, value, delta=delta, help=explanation)


def format_table_percent(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = out[col].map(lambda x: f"{x:.2%}" if pd.notna(x) else "N/A")
    return out
