from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


st.set_page_config(page_title="Model Validation", layout="wide")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


@st.cache_data
def load_dataset() -> pd.DataFrame:
    path = repo_root() / "data" / "processed" / "model_dataset.csv"

    if not path.exists():
        st.error("model_dataset.csv not found. Run scripts/build_features.py first.")
        st.stop()

    df = pd.read_csv(path)
    date_col = "date" if "date" in df.columns else df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col])
    return df.rename(columns={date_col: "date"}).set_index("date").sort_index()


def make_target(df: pd.DataFrame, horizon: int, threshold_quantile: float) -> pd.Series:
    if "contagion_risk_score" in df.columns:
        future_score = df["contagion_risk_score"].shift(-horizon)
        threshold = future_score.quantile(threshold_quantile)
        return (future_score >= threshold).astype(int)

    candidate_cols = [c for c in df.columns if "XFN.TO_ret" in c]
    if candidate_cols:
        future_return = df[candidate_cols[0]].shift(-horizon)
        threshold = future_return.quantile(1 - threshold_quantile)
        return (future_return <= threshold).astype(int)

    raise ValueError("No suitable target variable found.")


def prepare_xy(df: pd.DataFrame, horizon: int, threshold_quantile: float):
    y = make_target(df, horizon, threshold_quantile)

    excluded = ["contagion_risk_score"]
    feature_cols = [
        c
        for c in df.columns
        if c not in excluded and pd.api.types.is_numeric_dtype(df[c])
    ]

    X = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    dataset = pd.concat([X, y.rename("target")], axis=1).dropna()

    X = dataset[feature_cols]
    y = dataset["target"].astype(int)

    split = int(len(dataset) * 0.70)

    return X.iloc[:split], X.iloc[split:], y.iloc[:split], y.iloc[split:], feature_cols


def fit_models(X_train, X_test, y_train, y_test):
    models = {
        "Logistic Regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=2000, class_weight="balanced")),
            ]
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=250,
            max_depth=5,
            min_samples_leaf=10,
            random_state=42,
            class_weight="balanced",
        ),
    }

    fitted = {}
    rows = []

    for name, model in models.items():
        model.fit(X_train, y_train)
        fitted[name] = model

        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(X_test)[:, 1]
        else:
            prob = model.decision_function(X_test)

        pred = (prob >= 0.50).astype(int)
        fpr, tpr, _ = roc_curve(y_test, prob)
        auc_score = auc(fpr, tpr)

        rows.append(
            {
                "Model": name,
                "AUC": auc_score,
                "Accuracy": accuracy_score(y_test, pred),
                "Precision": precision_score(y_test, pred, zero_division=0),
                "Recall": recall_score(y_test, pred, zero_division=0),
                "Positive Rate": pred.mean(),
            }
        )

    return fitted, pd.DataFrame(rows).sort_values("AUC", ascending=False)


def plot_roc(fitted, X_test, y_test):
    fig = go.Figure()

    for name, model in fitted.items():
        prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, prob)
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{name} AUC={auc(fpr, tpr):.2f}"))

    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random", line=dict(dash="dash")))

    fig.update_layout(
        title="ROC Curve: Future Stress Event Prediction",
        xaxis_title="False positive rate",
        yaxis_title="True positive rate",
        height=460,
        margin=dict(l=20, r=20, t=50, b=20),
    )

    return fig


def feature_importance(model, feature_cols):
    if isinstance(model, Pipeline):
        inner = model.named_steps["model"]
    else:
        inner = model

    if hasattr(inner, "feature_importances_"):
        values = inner.feature_importances_
    elif hasattr(inner, "coef_"):
        values = np.abs(inner.coef_[0])
    else:
        return pd.DataFrame(columns=["Feature", "Importance"])

    out = pd.DataFrame({"Feature": feature_cols, "Importance": values})
    return out.sort_values("Importance", ascending=False).head(25)


def plot_feature_importance(importance_df):
    fig = go.Figure(
        go.Bar(
            x=importance_df["Importance"][::-1],
            y=importance_df["Feature"][::-1],
            orientation="h",
        )
    )
    fig.update_layout(
        title="Top Model Features",
        xaxis_title="Importance",
        height=650,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def confusion_table(model, X_test, y_test):
    prob = model.predict_proba(X_test)[:, 1]
    pred = (prob >= 0.50).astype(int)
    cm = confusion_matrix(y_test, pred)

    return pd.DataFrame(
        cm,
        index=["Actual calm", "Actual stress"],
        columns=["Predicted calm", "Predicted stress"],
    )


st.title("Model Validation")

st.info(
    """
    This page answers the credibility question: did the machine learning layer learn anything useful,
    or is it just producing attractive charts? Validation is done with a chronological train/test split,
    not a random shuffle, to reduce look-ahead bias.
    """
)

df = load_dataset()

st.sidebar.header("Validation Controls")
horizon = st.sidebar.slider("Prediction horizon, trading days", 1, 21, 5)
threshold_quantile = st.sidebar.slider("Stress-event percentile", 0.70, 0.95, 0.80, step=0.05)

X_train, X_test, y_train, y_test, feature_cols = prepare_xy(df, horizon, threshold_quantile)
fitted, metrics = fit_models(X_train, X_test, y_train, y_test)

best_model_name = metrics.iloc[0]["Model"]
best_model = fitted[best_model_name]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Best Model", best_model_name)
c2.metric("Best AUC", f"{metrics.iloc[0]['AUC']:.2f}")
c3.metric("Train Rows", f"{len(X_train):,}")
c4.metric("Test Rows", f"{len(X_test):,}")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "Model Metrics",
        "ROC / Discrimination",
        "Feature Importance",
        "Confusion Matrix",
        "Validation Methodology",
    ]
)

with tab1:
    st.subheader("Chronological Out-of-Sample Metrics")
    st.markdown(
        """
        These metrics are calculated on the later test period only.
        AUC above 0.50 means the model has some ability to rank future stress days above calm days.
        """
    )

    display = metrics.copy()
    for col in ["AUC", "Accuracy", "Precision", "Recall", "Positive Rate"]:
        display[col] = display[col].map(lambda x: f"{x:.3f}")

    st.dataframe(display, use_container_width=True, hide_index=True)

    if metrics.iloc[0]["AUC"] >= 0.65:
        st.success("The best model shows useful stress-discrimination ability.")
    elif metrics.iloc[0]["AUC"] >= 0.55:
        st.warning("The model shows modest signal. It may be useful as one input, not as a standalone predictor.")
    else:
        st.error("The model is close to random. More features, better targets, or regime-specific modeling are needed.")

with tab2:
    st.subheader("ROC Curve")
    st.markdown(
        """
        The ROC curve shows whether predicted stress probabilities rank actual stress events above calm periods.
        The diagonal line is random guessing.
        """
    )

    st.plotly_chart(plot_roc(fitted, X_test, y_test), use_container_width=True)

with tab3:
    st.subheader("What the Model Uses")
    st.markdown(
        """
        Feature importance makes the model more explainable.
        In a serious quant workflow, this helps determine whether the model is learning sensible risk drivers
        or spurious noise.
        """
    )

    chosen = st.selectbox("Model for feature importance", list(fitted.keys()))
    importance = feature_importance(fitted[chosen], feature_cols)

    st.plotly_chart(plot_feature_importance(importance), use_container_width=True)
    st.dataframe(importance, use_container_width=True, hide_index=True)

with tab4:
    st.subheader("Confusion Matrix")
    st.markdown(
        """
        This matrix shows classification behavior at a 50% probability threshold.

        - False positives: model warned about stress, but stress did not occur.
        - False negatives: stress occurred, but model missed it.

        In risk management, false negatives are usually more costly than false positives.
        """
    )

    cm_df = confusion_table(best_model, X_test, y_test)
    st.dataframe(cm_df, use_container_width=True)

    fig = go.Figure(
        data=go.Heatmap(
            z=cm_df.values,
            x=cm_df.columns,
            y=cm_df.index,
            text=cm_df.values,
            texttemplate="%{text}",
            colorscale="Blues",
        )
    )
    fig.update_layout(height=400, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig, use_container_width=True)

with tab5:
    st.subheader("Validation Methodology")
    st.markdown(
        """
        ### What is being predicted?

        The model predicts whether the system will enter a high-stress state over the next selected horizon.

        ### Why chronological split?

        Financial data is time-series data. Random train/test splitting leaks future market regimes into training.
        This project uses the earlier 70% of observations for training and the later 30% for testing.

        ### Why AUC?

        AUC measures ranking quality. A risk model does not need to be perfectly calibrated to be useful;
        it first needs to rank high-risk days above low-risk days.

        ### What would make this more senior-level?

        - Walk-forward validation across multiple folds.
        - Probability calibration.
        - Precision@K for top-risk days.
        - SHAP explanations.
        - Regime-specific validation.
        - Stress-period-only evaluation.
        - Bootstrap confidence intervals.
        - Feature leakage tests.
        - Comparison against naive baselines.

        ### Important limitation

        This is an educational research prototype. It is not investment advice, not a production credit model,
        and not a regulatory stress-testing system.
        """
    )