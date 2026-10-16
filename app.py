
from pathlib import Path
import json

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# =========================================================
# CONFIGURATION
# =========================================================
ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "threat_detection_pipeline.joblib"
METRICS_PATH = ROOT / "outputs" / "model_metrics.json"
MATRIX_PATH = ROOT / "outputs" / "confusion_matrix.csv"

st.set_page_config(
    page_title="Sentinel AI | Security Center",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# DESIGN SYSTEM
# =========================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
    --bg: #0a101c;
    --panel: #111b2b;
    --panel-hover: #162338;
    --border: #26364d;
    --text: #edf4ff;
    --muted: #91a2bc;
    --teal: #54e4c1;
    --blue: #75a7ff;
    --amber: #ffc76a;
    --red: #ff7585;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: var(--bg);
    color: var(--text);
}

[data-testid="stHeader"] {
    background: rgba(10, 16, 28, 0.94);
}

[data-testid="stSidebar"] {
    background: #0e1828;
    border-right: 1px solid var(--border);
}

[data-testid="stSidebar"] > div {
    padding-top: 1.6rem;
}

.block-container {
    max-width: 1500px;
    padding-top: 1.8rem;
    padding-bottom: 3rem;
}

h1, h2, h3 {
    color: var(--text);
    letter-spacing: -0.5px;
}

p, label {
    color: #c4d0e2;
}

.brand {
    font-size: 21px;
    font-weight: 800;
    letter-spacing: 1px;
    color: #f4f8ff;
}

.brand span {
    color: var(--teal);
}

.brand-sub {
    color: var(--muted);
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 2px;
    margin-top: 5px;
}

.side-label {
    color: #7f92ad;
    font-size: 10px;
    font-weight: 800;
    letter-spacing: 1.8px;
    margin: 25px 0 10px;
}

.hero {
    padding: 26px 30px;
    border: 1px solid #2a4056;
    border-radius: 17px;
    background:
        radial-gradient(ellipse at 90% 0%, #183b46 0%, transparent 43%),
        linear-gradient(120deg, #14233a, #101a2b);
    margin-bottom: 24px;
}

.eyebrow {
    font-size: 10px;
    font-weight: 800;
    letter-spacing: 2px;
    color: var(--teal);
    text-transform: uppercase;
}

.hero-title {
    font-size: clamp(25px, 3vw, 35px);
    font-weight: 800;
    letter-spacing: -1.2px;
    line-height: 1.2;
    color: #f7faff;
    margin: 10px 0;
}

.hero-desc {
    color: #a9bad1;
    font-size: 13px;
    max-width: 720px;
    line-height: 1.8;
}

.status-pill {
    display: inline-block;
    padding: 7px 11px;
    border-radius: 20px;
    background: #15372f;
    color: var(--teal);
    border: 1px solid #27604f;
    font-size: 11px;
    font-weight: 700;
}

.metric-card {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 13px;
    padding: 19px 20px;
    min-height: 130px;
    transition: border-color 0.2s ease;
}

.metric-card:hover {
    border-color: #4c7186;
}

.metric-top {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 8px;
}

.metric-label {
    color: #a5b6cd;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.8px;
}

.metric-icon {
    font-size: 16px;
}

.metric-value {
    color: #f4f7ff;
    font-size: 30px;
    font-weight: 800;
    letter-spacing: -1px;
    margin: 15px 0 7px;
}

.metric-caption {
    font-size: 11px;
    color: var(--muted);
}

.section-heading {
    font-size: 17px;
    font-weight: 750;
    color: #edf4ff;
    margin: 10px 0 4px;
}

.section-subtitle {
    font-size: 11px;
    color: var(--muted);
    margin-bottom: 15px;
}

.panel {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 20px;
    margin-bottom: 16px;
}

.info-row {
    display: flex;
    justify-content: space-between;
    gap: 12px;
    padding: 13px 0;
    border-bottom: 1px solid #26364a;
    font-size: 12px;
}

.info-row:last-child {
    border-bottom: none;
}

.info-label {
    color: #a1b2ca;
}

.info-value {
    color: #edf4ff;
    font-weight: 700;
    text-align: right;
}

.notice {
    background: #342d19;
    border: 1px solid #67572c;
    color: #ffda8e;
    padding: 14px 17px;
    border-radius: 11px;
    font-size: 12px;
    line-height: 1.7;
    margin: 18px 0;
}

.notice-danger {
    background: #351f29;
    border-color: #653443;
    color: #ffb0bb;
}

.notice-success {
    background: #14342e;
    border-color: #28584b;
    color: #82ead0;
}

.stButton button,
.stDownloadButton button {
    border-radius: 9px;
    font-weight: 700;
    min-height: 42px;
}

.stButton button[kind="primary"] {
    background: var(--teal);
    color: #082019;
    border: none;
}

div[data-testid="stFileUploader"] {
    background: var(--panel);
    border: 1px dashed #3b526e;
    border-radius: 12px;
    padding: 12px;
}

div[data-testid="stDataFrame"] {
    border: 1px solid var(--border);
    border-radius: 10px;
}

hr {
    border-color: var(--border);
}

.footer {
    border-top: 1px solid var(--border);
    margin-top: 28px;
    padding-top: 18px;
    display: flex;
    justify-content: space-between;
    gap: 12px;
    flex-wrap: wrap;
    color: #7789a4;
    font-size: 10px;
}

[data-testid="stMetric"] {
    background: var(--panel);
    padding: 15px;
    border: 1px solid var(--border);
    border-radius: 12px;
}

@media (max-width: 768px) {
    .hero {
        padding: 21px;
    }
    .metric-value {
        font-size: 25px;
    }
}
</style>
""", unsafe_allow_html=True)


# =========================================================
# DATA AND MODEL
# =========================================================
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metrics():
    with METRICS_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def metric_card(label, value, caption, icon):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-top">
                <div class="metric-label">{label}</div>
                <div class="metric-icon">{icon}</div>
            </div>
            <div class="metric-value">{value}</div>
            <div class="metric-caption">{caption}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_heading(title, subtitle=""):
    st.markdown(
        f'<div class="section-heading">{title}</div>'
        f'<div class="section-subtitle">{subtitle}</div>',
        unsafe_allow_html=True,
    )


def footer():
    st.markdown("""
    <div class="footer">
        <span>SENTINEL AI · CYBERSECURITY RESEARCH</span>
        <span>NSL-KDD BENCHMARK · EDUCATIONAL PROTOTYPE</span>
    </div>
    """, unsafe_allow_html=True)


# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown(
        '<div class="brand">🛡 SENTINEL <span>AI</span></div>'
        '<div class="brand-sub">SECURITY OPERATIONS CENTER</div>',
        unsafe_allow_html=True,
    )

    st.divider()

    st.markdown('<div class="side-label">WORKSPACE</div>',
                unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        [
            "Security Overview",
            "Threat Analysis",
            "Model Insights",
            "About Sentinel AI",
        ],
        label_visibility="collapsed",
    )

    st.markdown('<div class="side-label">SYSTEM</div>',
                unsafe_allow_html=True)

    artifacts_ready = MODEL_PATH.exists() and METRICS_PATH.exists()

    if artifacts_ready:
        st.markdown(
            '<div class="notice-success">● &nbsp; ML pipeline available</div>',
            unsafe_allow_html=True,
        )
    else:
        st.error("Required model artifacts are missing.")

    st.caption("Model · Random Forest")
    st.caption("Dataset · NSL-KDD")
    st.caption("Mode · Offline analysis")

    st.divider()
    st.caption("Sentinel AI · Project build")


# =========================================================
# VALIDATE ARTIFACTS
# =========================================================
if not artifacts_ready:
    st.error(
        "Model files are missing. Run `python src\\train_model.py` "
        "from your project directory, then restart the app."
    )
    st.stop()

model = load_model()
metrics = load_metrics()


# =========================================================
# PAGE HEADER
# =========================================================
if page == "Security Overview":
    eyebrow = "SECURITY INTELLIGENCE / OVERVIEW"
    title = "Threat Detection Center"
    description = (
        "A unified view of your machine-learning intrusion detection "
        "results, classification performance, and evaluation dataset."
    )
elif page == "Threat Analysis":
    eyebrow = "SECURITY INTELLIGENCE / ANALYSIS"
    title = "Network Threat Analysis"
    description = (
        "Upload a processed network-flow dataset to classify records "
        "and export the prediction results."
    )
elif page == "Model Insights":
    eyebrow = "SECURITY INTELLIGENCE / MODEL"
    title = "Model Performance"
    description = (
        "Review the measured performance and limitations of the "
        "Random Forest baseline."
    )
else:
    eyebrow = "SENTINEL AI / PROJECT"
    title = "About the Platform"
    description = (
        "Explore the technology, dataset, architecture, and scope "
        "of this cybersecurity research prototype."
    )

st.markdown(
    f"""
    <div class="hero">
        <div class="eyebrow">{eyebrow}</div>
        <div class="hero-title">{title}</div>
        <div class="hero-desc">{description}</div>
        <br>
        <span class="status-pill">● &nbsp; MODEL ARTIFACTS AVAILABLE</span>
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# SECURITY OVERVIEW
# =========================================================
if page == "Security Overview":

    section_heading(
        "Performance at a glance",
        "Measured on the separate NSL-KDD test dataset.",
    )

    cols = st.columns(4, gap="medium")

    card_data = [
        ("ACCURACY", f"{metrics['accuracy']:.2%}",
         "Correct classifications", "◈"),
        ("ATTACK PRECISION", f"{metrics['precision']:.2%}",
         "Predicted attacks that were attacks", "⌁"),
        ("ATTACK RECALL", f"{metrics['recall']:.2%}",
         "Attacks detected by the model", "◎"),
        ("ROC-AUC", f"{metrics['roc_auc']:.4f}",
         "Ranking discrimination score", "↗"),
    ]

    for col, (label, value, caption, icon) in zip(cols, card_data):
        with col:
            metric_card(label, value, caption, icon)

    st.markdown("")

    left, right = st.columns([1.35, 1], gap="large")

    with left:
        section_heading(
            "Classification performance",
            "Actual classes compared with model predictions.",
        )

        if MATRIX_PATH.exists():
            matrix = pd.read_csv(MATRIX_PATH, index_col=0)

            z = matrix.values
            labels = [
                [f"{int(v):,}" for v in row]
                for row in z
            ]

            fig = go.Figure(
                data=go.Heatmap(
                    z=z,
                    x=["Predicted normal", "Predicted attack"],
                    y=["Actual normal", "Actual attack"],
                    text=labels,
                    texttemplate="%{text}",
                    textfont=dict(size=15, color="#f4f8ff"),
                    colorscale=[
                        [0, "#14233a"],
                        [0.5, "#246c76"],
                        [1, "#54e4c1"],
                    ],
                    showscale=False,
                    xgap=5,
                    ygap=5,
                    hovertemplate=(
                        "Actual: %{y}<br>"
                        "Prediction: %{x}<br>"
                        "Records: %{z:,}<extra></extra>"
                    ),
                )
            )

            fig.update_layout(
                height=335,
                paper_bgcolor="#111b2b",
                plot_bgcolor="#111b2b",
                font=dict(color="#c8d5e8", size=11),
                margin=dict(l=8, r=8, t=15, b=8),
                xaxis=dict(side="bottom", fixedrange=True),
                yaxis=dict(autorange="reversed", fixedrange=True),
            )

            st.markdown('<div class="panel">', unsafe_allow_html=True)
            st.plotly_chart(
                fig,
                use_container_width=True,
                config={"displayModeBar": False},
            )
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("Confusion matrix is not available.")

    with right:
        section_heading(
            "Evaluation details",
            "Training and test execution summary.",
        )

        st.markdown(
            f"""
            <div class="panel">
                <div class="info-row">
                    <span class="info-label">Algorithm</span>
                    <span class="info-value">
                        {metrics.get("model_name", "Random Forest")}
                    </span>
                </div>
                <div class="info-row">
                    <span class="info-label">Training records</span>
                    <span class="info-value">
                        {metrics["training_rows"]:,}
                    </span>
                </div>
                <div class="info-row">
                    <span class="info-label">Testing records</span>
                    <span class="info-value">
                        {metrics["testing_rows"]:,}
                    </span>
                </div>
                <div class="info-row">
                    <span class="info-label">Input features</span>
                    <span class="info-value">
                        {metrics["feature_count_before_encoding"]}
                    </span>
                </div>
                <div class="info-row">
                    <span class="info-label">Training duration</span>
                    <span class="info-value">
                        {metrics["training_seconds"]:.2f} seconds
                    </span>
                </div>
                <div class="info-row">
                    <span class="info-label">Random seed</span>
                    <span class="info-value">
                        {metrics.get("random_state", 42)}
                    </span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        f"""
        <div class="notice">
            <strong>Baseline limitation</strong><br>
            Attack recall is {metrics['recall']:.2%}. The model misses some
            attacks on this benchmark and is not production-ready.
            These figures describe offline test data, not live network traffic.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("")
    section_heading(
        "Project capabilities",
        "Current functions available in this prototype.",
    )

    c1, c2, c3 = st.columns(3, gap="medium")

    with c1:
        st.markdown(
            '<div class="panel"><h3>01 · Classify</h3>'
            '<p>Predict normal versus attack for network-flow records.</p>'
            '</div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            '<div class="panel"><h3>02 · Evaluate</h3>'
            '<p>Review model metrics and confusion-matrix results.</p>'
            '</div>',
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            '<div class="panel"><h3>03 · Export</h3>'
            '<p>Download batch predictions for further analysis.</p>'
            '</div>',
            unsafe_allow_html=True,
        )


# =========================================================
# THREAT ANALYSIS
# =========================================================
elif page == "Threat Analysis":

    section_heading(
        "Batch analysis workspace",
        "Analyze a processed NSL-KDD-compatible CSV file.",
    )

    st.markdown(
        """
        <div class="panel">
            <strong>Input requirements</strong>
            <p>
            Upload a CSV containing the original NSL-KDD feature columns.
            The app ignores existing target and label columns when preparing
            model input. Use the processed dataset for the first test.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded = st.file_uploader(
        "Choose a processed CSV file",
        type=["csv"],
        help="For example: data/processed/test_processed.csv",
    )

    if uploaded is not None:
        try:
            data = pd.read_csv(uploaded)

            if data.empty:
                st.error("The uploaded CSV contains no records.")
            else:
                excluded = {"target", "label", "difficulty"}

                expected_columns = list(
                    model.named_steps["preprocessor"].feature_names_in_
                )

                missing = [
                    col for col in expected_columns
                    if col not in data.columns
                ]

                if missing:
                    st.error(
                        "The file is missing required features: "
                        + ", ".join(missing)
                    )
                else:
                    if len(data) > 200_000:
                        st.error(
                            "This file exceeds the 200,000-row limit "
                            "for this dashboard. Upload a smaller subset."
                        )
                        st.stop()

                    X = data[expected_columns].copy()

                    with st.spinner("Analyzing network-flow records..."):
                        predictions = model.predict(X)
                        probabilities = model.predict_proba(X)[:, 1]

                    results = data.copy()
                    results["predicted_class"] = [
                        "Potential Attack" if p == 1 else "Normal"
                        for p in predictions
                    ]
                    results["attack_probability"] = probabilities.round(4)

                    total = len(results)
                    attack_count = int((predictions == 1).sum())
                    normal_count = total - attack_count

                    st.markdown("")
                    cols = st.columns(3, gap="medium")

                    with cols[0]:
                        metric_card(
                            "RECORDS ANALYZED",
                            f"{total:,}",
                            "Uploaded records",
                            "▤",
                        )
                    with cols[1]:
                        metric_card(
                            "POTENTIAL ATTACKS",
                            f"{attack_count:,}",
                            "Predicted attack records",
                            "⚑",
                        )
                    with cols[2]:
                        rate = attack_count / total if total else 0
                        metric_card(
                            "FLAGGED SHARE",
                            f"{rate:.1%}",
                            "Share predicted as attacks",
                            "◉",
                        )

                    if attack_count:
                        st.markdown(
                            '<div class="notice-danger">'
                            '<strong>Potential attacks detected.</strong><br>'
                            'Review the predictions and probabilities below. '
                            'A model prediction alone does not confirm an incident.'
                            '</div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            '<div class="notice-success">'
                            'No records were classified as attacks in this batch. '
                            'This does not guarantee the absence of malicious traffic.'
                            '</div>',
                            unsafe_allow_html=True,
                        )

                    left, right = st.columns([1, 1.2], gap="large")

                    with left:
                        section_heading(
                            "Prediction distribution",
                            "Share of records assigned to each class.",
                        )

                        breakdown = pd.DataFrame({
                            "Class": ["Normal", "Potential Attack"],
                            "Records": [normal_count, attack_count],
                        })

                        fig = px.pie(
                            breakdown,
                            names="Class",
                            values="Records",
                            hole=0.68,
                            color="Class",
                            color_discrete_map={
                                "Normal": "#54e4c1",
                                "Potential Attack": "#ff7585",
                            },
                        )

                        fig.update_traces(
                            textposition="outside",
                            textinfo="percent+label",
                        )
                        fig.update_layout(
                            height=330,
                            paper_bgcolor="#111b2b",
                            plot_bgcolor="#111b2b",
                            font=dict(color="#dce7f6", size=11),
                            legend_title_text="",
                            margin=dict(l=8, r=8, t=20, b=10),
                        )

                        st.plotly_chart(
                            fig,
                            use_container_width=True,
                            config={"displayModeBar": False},
                        )

                    with right:
                        section_heading(
                            "Prediction results",
                            "Filter and inspect the model's output.",
                        )

                        filter_class = st.selectbox(
                            "Show records",
                            [
                                "All records",
                                "Potential Attack",
                                "Normal",
                            ],
                        )

                        min_probability = st.slider(
                            "Minimum attack probability",
                            min_value=0.0,
                            max_value=1.0,
                            value=0.0,
                            step=0.05,
                        )

                        filtered = results[
                            results["attack_probability"] >= min_probability
                        ]

                        if filter_class != "All records":
                            filtered = filtered[
                                filtered["predicted_class"] == filter_class
                            ]

                        display_columns = [
                            c for c in [
                                "predicted_class",
                                "attack_probability",
                                "protocol_type",
                                "service",
                                "flag",
                                "src_bytes",
                                "dst_bytes",
                                "label",
                            ]
                            if c in filtered.columns
                        ]

                        st.caption(
                            f"{len(filtered):,} records match the filters."
                        )

                        st.dataframe(
                            filtered[display_columns].head(500),
                            use_container_width=True,
                            hide_index=True,
                        )

                    st.download_button(
                        "Download complete prediction report",
                        data=results.to_csv(index=False).encode("utf-8"),
                        file_name="sentinel_ai_threat_analysis.csv",
                        mime="text/csv",
                        type="primary",
                        use_container_width=True,
                    )

                    st.caption(
                        "Probability is the model's estimated probability "
                        "of the attack class. Predictions are not verified incidents."
                    )

        except Exception as exc:
            st.error(f"Could not analyze the uploaded file: {exc}")


# =========================================================
# MODEL INSIGHTS
# =========================================================
elif page == "Model Insights":

    section_heading(
        "Model evaluation",
        "Measured performance on the separate NSL-KDD test set.",
    )

    metric_cols = st.columns(3, gap="medium")

    with metric_cols[0]:
        metric_card(
            "PRECISION",
            f"{metrics['precision']:.2%}",
            "Attack predictions that were attacks",
            "◎",
        )

    with metric_cols[1]:
        metric_card(
            "RECALL",
            f"{metrics['recall']:.2%}",
            "Attacks detected",
            "⌁",
        )

    with metric_cols[2]:
        metric_card(
            "F1 SCORE",
            f"{metrics['f1_score']:.2%}",
            "Precision-recall balance",
            "◈",
        )

    st.markdown("")
    section_heading("Classification report", "Per-class performance.")

    report_path = ROOT / "outputs" / "classification_report.txt"

    if report_path.exists():
        report_text = report_path.read_text(encoding="utf-8")
        st.code(report_text, language="text")
    else:
        st.info("Classification report is not available.")

    st.markdown("")
    section_heading("Interpretation", "What these measurements tell us.")

    st.markdown(
        f"""
        <div class="panel">
            <div class="info-row">
                <span class="info-label">Accuracy</span>
                <span class="info-value">{metrics['accuracy']:.2%}</span>
            </div>
            <div class="info-row">
                <span class="info-label">Attack precision</span>
                <span class="info-value">{metrics['precision']:.2%}</span>
            </div>
            <div class="info-row">
                <span class="info-label">Attack recall</span>
                <span class="info-value">{metrics['recall']:.2%}</span>
            </div>
            <div class="info-row">
                <span class="info-label">ROC-AUC</span>
                <span class="info-value">{metrics['roc_auc']:.4f}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="notice">
            <strong>Next model-development goal</strong><br>
            Improve attack recall while monitoring precision, F1-score,
            false positives, and false negatives. Any new model should be
            compared on the same held-out test set.
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# ABOUT
# =========================================================
elif page == "About Sentinel AI":

    section_heading(
        "Project overview",
        "A machine-learning cybersecurity portfolio project.",
    )

    st.markdown(
        """
        <div class="panel">
            <h3>What is Sentinel AI?</h3>
            <p>
            Sentinel AI demonstrates binary classification of network-flow
            records as normal traffic or potential attacks using the
            NSL-KDD intrusion-detection benchmark.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns(2, gap="large")

    with left:
        section_heading("Technology stack")
        st.markdown(
            """
            <div class="panel">
                <div class="info-row"><span class="info-label">Language</span>
                <span class="info-value">Python</span></div>
                <div class="info-row"><span class="info-label">ML</span>
                <span class="info-value">scikit-learn</span></div>
                <div class="info-row"><span class="info-label">Data</span>
                <span class="info-value">pandas / NumPy</span></div>
                <div class="info-row"><span class="info-label">Dashboard</span>
                <span class="info-value">Streamlit</span></div>
                <div class="info-row"><span class="info-label">Charts</span>
                <span class="info-value">Plotly</span></div>
                <div class="info-row"><span class="info-label">Model storage</span>
                <span class="info-value">Joblib</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        section_heading("Current capabilities")
        st.markdown(
            """
            <div class="panel">
                <div class="info-row"><span class="info-label">Classification</span>
                <span class="info-value">Normal / Attack</span></div>
                <div class="info-row"><span class="info-label">Model evaluation</span>
                <span class="info-value">Implemented</span></div>
                <div class="info-row"><span class="info-label">Batch CSV analysis</span>
                <span class="info-value">Implemented</span></div>
                <div class="info-row"><span class="info-label">Prediction export</span>
                <span class="info-value">Implemented</span></div>
                <div class="info-row"><span class="info-label">Live packet capture</span>
                <span class="info-value">Not implemented</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class="notice">
            <strong>Responsible use</strong><br>
            This is an educational research prototype trained on benchmark
            data. It does not monitor live networks, verify security incidents,
            or replace a production intrusion-detection system.
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# FOOTER
# =========================================================
footer()
