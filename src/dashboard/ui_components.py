import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


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


def plot_time_series(df, y, title, explanation=None):
    fig = px.line(df, y=y, title=title)
    fig.update_layout(height=420, margin=dict(l=20, r=20, t=50, b=20))
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
    fig.update_layout(height=500, margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(fig, use_container_width=True)
    if explanation:
        st.caption(explanation)


def interpretation_box(title, bullets):
    st.markdown(f"#### {title}")
    for b in bullets:
        st.markdown(f"- {b}")