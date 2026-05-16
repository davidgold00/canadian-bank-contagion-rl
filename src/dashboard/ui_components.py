import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def apply_dashboard_style():
    st.markdown(
        """
        <style>
        :root {
            --ink: #18212f;
            --muted: #5d6877;
            --line: #d7dde5;
            --panel: #f7f9fc;
            --blue: #1d5f8f;
            --green: #1f7a5a;
            --amber: #ad6b00;
            --red: #b42318;
        }
        .block-container {
            padding-top: 2.1rem;
            padding-bottom: 3rem;
            max-width: 1360px;
        }
        h1, h2, h3 {
            letter-spacing: 0 !important;
            color: var(--ink);
        }
        div[data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 0.85rem 0.95rem;
            box-shadow: 0 1px 2px rgba(24, 33, 47, 0.04);
        }
        div[data-testid="stMetricLabel"] p {
            color: var(--muted);
            font-size: 0.86rem;
        }
        div[data-testid="stMetricValue"] {
            color: var(--ink);
            font-size: 1.6rem;
        }
        .analysis-card {
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 1rem 1.1rem;
            margin: 0.3rem 0 1rem 0;
        }
        .analysis-card h4 {
            margin: 0 0 0.45rem 0;
            color: var(--ink);
        }
        .analysis-card p, .analysis-card li {
            color: #354153;
            line-height: 1.45;
        }
        .source-pill {
            display: inline-block;
            padding: 0.25rem 0.55rem;
            border: 1px solid var(--line);
            border-radius: 999px;
            background: #ffffff;
            color: var(--muted);
            font-size: 0.82rem;
            margin-right: 0.35rem;
            margin-bottom: 0.35rem;
        }
        .small-note {
            color: var(--muted);
            font-size: 0.9rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def analyst_header(title, subtitle, date_text=None, source_text=None):
    st.title(title)
    st.markdown(f"### {subtitle}")
    details = []
    if date_text:
        details.append(f"<span class='source-pill'>Data through {date_text}</span>")
    if source_text:
        details.append(f"<span class='source-pill'>{source_text}</span>")
    if details:
        st.markdown(" ".join(details), unsafe_allow_html=True)


def page_header(title, subtitle, why_it_matters, how_to_read):
    st.title(title)
    st.markdown(f"### {subtitle}")

    c1, c2 = st.columns([1, 1])

    with c1:
        st.info(f"**Why this matters**\n\n{why_it_matters}")

    with c2:
        st.success(f"**How to read this page**\n\n{how_to_read}")

    st.divider()


def risk_badge(score):
    if score >= 75:
        st.error(f"High systemic stress: {score:.1f}/100")
    elif score >= 50:
        st.warning(f"Moderate systemic stress: {score:.1f}/100")
    else:
        st.success(f"Low systemic stress: {score:.1f}/100")


def explain_metric(label, value, explanation, delta=None):
    st.metric(label, value, delta=delta, help=explanation)


def insight_card(title, body, status="info"):
    colors = {
        "success": "#e9f7ef",
        "warning": "#fff4df",
        "danger": "#fdecec",
        "info": "#f7f9fc",
    }
    border = {
        "success": "#8fd2ad",
        "warning": "#e5b75c",
        "danger": "#e49a94",
        "info": "#d7dde5",
    }
    st.markdown(
        f"""
        <div class="analysis-card" style="background:{colors.get(status, colors['info'])}; border-color:{border.get(status, border['info'])};">
            <h4>{title}</h4>
            <p>{body}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def action_list(title, actions):
    st.markdown(f"#### {title}")
    for action in actions:
        st.markdown(f"- {action}")


def format_table_percent(df, columns):
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = out[col].map(lambda x: f"{x:.2%}" if pd.notna(x) else "N/A")
    return out


def plot_time_series(df, y, title, explanation=None):
    fig = px.line(df, y=y, title=title)
    fig.update_layout(
        height=420,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color="#18212f"),
    )
    st.plotly_chart(fig, use_container_width=True)
    if explanation:
        st.caption(explanation)


def plot_heatmap(matrix, title, explanation=None):
    fig = px.imshow(
        matrix,
        text_auto=".2f",
        aspect="auto",
        title=title,
        color_continuous_scale="RdBu_r",
    )
    fig.update_layout(
        height=500,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color="#18212f"),
    )
    st.plotly_chart(fig, use_container_width=True)
    if explanation:
        st.caption(explanation)


def interpretation_box(title, bullets):
    st.markdown(f"#### {title}")
    for b in bullets:
        st.markdown(f"- {b}")
