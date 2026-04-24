from __future__ import annotations

from pathlib import Path
from typing import Dict, List
import pickle

import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="PhishGuard AI",
    page_icon=":shield:",
    layout="wide",
    initial_sidebar_state="expanded",
)


APP_ROOT = Path(__file__).resolve().parent
DATA_PATH = APP_ROOT / "Network_Data" / "phisingData.csv"
MODEL_PATH = APP_ROOT / "final_model" / "model.pkl"
PREPROCESSOR_PATH = APP_ROOT / "final_model" / "preprocessor.pkl"

FEATURE_COLUMNS: List[str] = [
    "having_IP_Address",
    "URL_Length",
    "Shortining_Service",
    "having_At_Symbol",
    "double_slash_redirecting",
    "Prefix_Suffix",
    "having_Sub_Domain",
    "SSLfinal_State",
    "Domain_registeration_length",
    "Favicon",
    "port",
    "HTTPS_token",
    "Request_URL",
    "URL_of_Anchor",
    "Links_in_tags",
    "SFH",
    "Submitting_to_email",
    "Abnormal_URL",
    "Redirect",
    "on_mouseover",
    "RightClick",
    "popUpWidnow",
    "Iframe",
    "age_of_domain",
    "DNSRecord",
    "web_traffic",
    "Page_Rank",
    "Google_Index",
    "Links_pointing_to_page",
    "Statistical_report",
]

FEATURE_GROUPS: Dict[str, List[str]] = {
    "URL & Structure Signals": [
        "having_IP_Address",
        "URL_Length",
        "Shortining_Service",
        "having_At_Symbol",
        "double_slash_redirecting",
        "Prefix_Suffix",
        "having_Sub_Domain",
    ],
    "Security & Certificate Signals": [
        "SSLfinal_State",
        "Domain_registeration_length",
        "Favicon",
        "port",
        "HTTPS_token",
    ],
    "Content & Redirection Signals": [
        "Request_URL",
        "URL_of_Anchor",
        "Links_in_tags",
        "SFH",
        "Submitting_to_email",
        "Abnormal_URL",
        "Redirect",
    ],
    "Interaction Signals": [
        "on_mouseover",
        "RightClick",
        "popUpWidnow",
        "Iframe",
    ],
    "Trust & Reputation Signals": [
        "age_of_domain",
        "DNSRecord",
        "web_traffic",
        "Page_Rank",
        "Google_Index",
        "Links_pointing_to_page",
        "Statistical_report",
    ],
}

FEATURE_LABELS: Dict[str, str] = {
    "having_IP_Address": "Uses IP address instead of domain",
    "URL_Length": "Unusually long URL",
    "Shortining_Service": "Shortened URL service detected",
    "having_At_Symbol": "Contains @ symbol",
    "double_slash_redirecting": "Extra // redirection pattern",
    "Prefix_Suffix": "Hyphenated domain pattern",
    "having_Sub_Domain": "Suspicious sub-domain depth",
    "SSLfinal_State": "SSL certificate trust state",
    "Domain_registeration_length": "Short domain registration duration",
    "Favicon": "Favicon served from suspicious source",
    "port": "Suspicious port usage",
    "HTTPS_token": "HTTPS token used inside domain name",
    "Request_URL": "External resource loading pattern",
    "URL_of_Anchor": "Anchor tag destination quality",
    "Links_in_tags": "Link behavior inside tags",
    "SFH": "Server form handler behavior",
    "Submitting_to_email": "Form submits to email",
    "Abnormal_URL": "Abnormal URL structure",
    "Redirect": "Number of redirects",
    "on_mouseover": "Misleading mouseover behavior",
    "RightClick": "Right-click disabled",
    "popUpWidnow": "Popup window behavior",
    "Iframe": "Iframe usage",
    "age_of_domain": "Domain age trust level",
    "DNSRecord": "DNS record availability",
    "web_traffic": "Observed website traffic trust",
    "Page_Rank": "Page rank authority signal",
    "Google_Index": "Indexed by Google",
    "Links_pointing_to_page": "Backlink quality",
    "Statistical_report": "Statistical blacklist signal",
}

VALUE_OPTIONS = {
    -1: "High risk / phishing-like",
    0: "Suspicious / mixed",
    1: "Legitimate / safe",
}

RESULT_MAP = {
    -1: ("Phishing", "#ff6b6b"),
    0: ("Suspicious", "#f6c344"),
    1: ("Legitimate", "#34c38f"),
}


def inject_styles() -> None:
    st.markdown(
        """
        <style>
            .stApp {
                background:
                    radial-gradient(circle at top left, rgba(255, 140, 66, 0.18), transparent 28%),
                    radial-gradient(circle at top right, rgba(0, 118, 255, 0.14), transparent 22%),
                    linear-gradient(180deg, #f6f7fb 0%, #eef2f7 100%);
            }
            .block-container {
                padding-top: 1.5rem;
                padding-bottom: 2rem;
            }
            .hero-card {
                padding: 2rem 2.2rem;
                border-radius: 24px;
                background: linear-gradient(135deg, #132238 0%, #1c3f68 58%, #ff8c42 130%);
                color: white;
                box-shadow: 0 24px 80px rgba(19, 34, 56, 0.24);
                margin-bottom: 1.2rem;
            }
            .hero-title {
                font-size: 2.6rem;
                font-weight: 800;
                margin-bottom: 0.5rem;
                line-height: 1.1;
            }
            .hero-subtitle {
                font-size: 1.05rem;
                color: rgba(255, 255, 255, 0.88);
                max-width: 760px;
            }
            .stat-card {
                background: rgba(255, 255, 255, 0.82);
                border: 1px solid rgba(19, 34, 56, 0.08);
                border-radius: 20px;
                padding: 1rem 1.1rem;
                box-shadow: 0 14px 42px rgba(28, 63, 104, 0.08);
            }
            .stat-label {
                color: #5c677d;
                font-size: 0.9rem;
                margin-bottom: 0.2rem;
            }
            .stat-value {
                color: #132238;
                font-size: 1.7rem;
                font-weight: 800;
            }
            .section-title {
                color: #132238;
                font-size: 1.3rem;
                font-weight: 700;
                margin: 0.4rem 0 0.9rem 0;
            }
            .result-card {
                padding: 1.25rem;
                border-radius: 20px;
                border: 1px solid rgba(19, 34, 56, 0.08);
                background: rgba(255, 255, 255, 0.92);
                box-shadow: 0 14px 42px rgba(19, 34, 56, 0.08);
            }
            .tag {
                display: inline-block;
                padding: 0.2rem 0.6rem;
                border-radius: 999px;
                background: rgba(255,255,255,0.16);
                border: 1px solid rgba(255,255,255,0.24);
                margin-right: 0.5rem;
                font-size: 0.85rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_dataset() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


@st.cache_resource(show_spinner=False)
def load_artifacts():
    with PREPROCESSOR_PATH.open("rb") as preprocessor_file:
        preprocessor = pickle.load(preprocessor_file)
    with MODEL_PATH.open("rb") as model_file:
        model = pickle.load(model_file)
    return preprocessor, model


def predict_dataframe(df: pd.DataFrame):
    preprocessor, model = load_artifacts()
    transformed = preprocessor.transform(df[FEATURE_COLUMNS])
    predictions = model.predict(transformed)
    confidence = None
    if hasattr(model, "predict_proba"):
        try:
            probabilities = model.predict_proba(transformed)
            confidence = probabilities.max(axis=1)
        except Exception:
            confidence = None
    return predictions, confidence


def build_default_values(dataset: pd.DataFrame) -> Dict[str, int]:
    defaults: Dict[str, int] = {}
    feature_frame = dataset[FEATURE_COLUMNS]
    for column in FEATURE_COLUMNS:
        mode_series = feature_frame[column].mode(dropna=True)
        default_value = int(mode_series.iloc[0]) if not mode_series.empty else 0
        defaults[column] = default_value
    return defaults


def render_hero(dataset: pd.DataFrame) -> None:
    class_counts = dataset["Result"].value_counts().to_dict()
    phishing_count = int(class_counts.get(-1, 0))
    safe_count = int(class_counts.get(1, 0))
    sample_count = len(dataset)
    feature_count = len(FEATURE_COLUMNS)

    st.markdown(
        """
        <div class="hero-card">
            <div class="tag"></div>
            <div class="tag"></div>
            <div class="tag"></div>
            <div class="hero-title">PhishGuard AI</div>
            <div class="hero-subtitle">
                An interactive phishing detection dashboard for live prediction, batch testing.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            f'<div class="stat-card"><div class="stat-label">Dataset Rows</div><div class="stat-value">{sample_count}</div></div>',
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f'<div class="stat-card"><div class="stat-label">Model Features</div><div class="stat-value">{feature_count}</div></div>',
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f'<div class="stat-card"><div class="stat-label">Phishing Samples</div><div class="stat-value">{phishing_count}</div></div>',
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            f'<div class="stat-card"><div class="stat-label">Legitimate Samples</div><div class="stat-value">{safe_count}</div></div>',
            unsafe_allow_html=True,
        )


def render_sidebar() -> None:
    st.sidebar.title("Project Controls")
    st.sidebar.markdown(
        """
        Use this app during demo to:
        - test one website profile at a time
        - upload a CSV for batch prediction
        - show dataset insights and feature meanings
        """
    )
    st.sidebar.info(
        "Expected feature values are mostly `-1`, `0`, and `1`.\n\n"
        "`-1` = phishing-like, `0` = suspicious, `1` = legitimate."
    )
    st.sidebar.success("Run command: `streamlit run streamlit_app.py`")


def render_overview_tab(dataset: pd.DataFrame) -> None:
    st.markdown('<div class="section-title">Dataset Snapshot</div>', unsafe_allow_html=True)
    overview_col, chart_col = st.columns([1.25, 1])

    with overview_col:
        st.dataframe(dataset.head(8), use_container_width=True)

    with chart_col:
        distribution = (
            dataset["Result"]
            .map({-1: "Phishing", 1: "Legitimate", 0: "Suspicious"})
            .value_counts()
            .rename_axis("Class")
            .reset_index(name="Count")
            .set_index("Class")
        )
        st.bar_chart(distribution)

    


def render_single_prediction_tab(dataset: pd.DataFrame) -> None:
    st.markdown('<div class="section-title">Single Website Prediction</div>', unsafe_allow_html=True)
    defaults = build_default_values(dataset)

    with st.form("single_prediction_form"):
        inputs: Dict[str, int] = {}
        for group_name, columns in FEATURE_GROUPS.items():
            with st.expander(group_name, expanded=False):
                cols = st.columns(2)
                for idx, column in enumerate(columns):
                    with cols[idx % 2]:
                        current_value = defaults[column]
                        option_index = list(VALUE_OPTIONS.keys()).index(current_value)
                        selected_value = st.selectbox(
                            FEATURE_LABELS[column],
                            options=list(VALUE_OPTIONS.keys()),
                            index=option_index,
                            format_func=lambda value: f"{value} - {VALUE_OPTIONS[value]}",
                            help=column,
                            key=f"input_{column}",
                        )
                        inputs[column] = int(selected_value)

        submitted = st.form_submit_button("Analyze Risk", use_container_width=True)

    if submitted:
        input_df = pd.DataFrame([inputs], columns=FEATURE_COLUMNS)
        prediction, confidence = predict_dataframe(input_df)
        predicted_value = int(prediction[0])
        label, color = RESULT_MAP.get(predicted_value, ("Unknown", "#5c677d"))

        st.markdown(
            f"""
            <div class="result-card">
                <div style="font-size:0.95rem;color:#5c677d;">Prediction Result</div>
                <div style="font-size:2rem;font-weight:800;color:{color};margin:0.25rem 0;">{label}</div>
                <div style="color:#132238;">The model evaluated the entered feature profile and marked this website as <b>{label.lower()}</b>.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if confidence is not None:
            st.progress(float(confidence[0]))
            st.caption(f"Model confidence: {confidence[0] * 100:.2f}%")

        st.dataframe(input_df, use_container_width=True)


def render_batch_prediction_tab(dataset: pd.DataFrame) -> None:
    st.markdown('<div class="section-title">Batch CSV Prediction</div>', unsafe_allow_html=True)
    sample_csv = dataset[FEATURE_COLUMNS].head(25).to_csv(index=False).encode("utf-8")

    helper_col, upload_col = st.columns([1, 1.15])
    with helper_col:
        st.markdown(
            """
            Upload a CSV that contains the same 30 feature columns used during training.
            """
        )
        st.download_button(
            "Download Sample Input CSV",
            data=sample_csv,
            file_name="sample_phishing_input.csv",
            mime="text/csv",
            use_container_width=True,
        )
    

    with upload_col:
        uploaded_file = st.file_uploader("Upload feature CSV", type=["csv"])

    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        missing_columns = [column for column in FEATURE_COLUMNS if column not in batch_df.columns]
        if missing_columns:
            st.error(f"Missing required columns: {', '.join(missing_columns)}")
            return

        prediction_df = batch_df[FEATURE_COLUMNS].copy()
        predictions, confidence = predict_dataframe(prediction_df)
        prediction_df["Prediction"] = [RESULT_MAP.get(int(value), ("Unknown", ""))[0] for value in predictions]

        if confidence is not None:
            prediction_df["Confidence"] = [round(float(value), 4) for value in confidence]

        st.success(f"Processed {len(prediction_df)} rows successfully.")
        st.dataframe(prediction_df.head(20), use_container_width=True)
        st.download_button(
            "Download Prediction Results",
            data=prediction_df.to_csv(index=False).encode("utf-8"),
            file_name="prediction_results.csv",
            mime="text/csv",
            use_container_width=True,
        )


def render_feature_guide_tab() -> None:
    st.markdown('<div class="section-title">Feature Guide</div>', unsafe_allow_html=True)
    guide_rows = [
        {"Feature": feature, "Meaning": FEATURE_LABELS[feature], "Values": "-1 / 0 / 1"}
        for feature in FEATURE_COLUMNS
    ]
    st.dataframe(pd.DataFrame(guide_rows), use_container_width=True)
    

def validate_required_files() -> None:
    missing_files = [path.name for path in [DATA_PATH, MODEL_PATH, PREPROCESSOR_PATH] if not path.exists()]
    if missing_files:
        st.error(
            "Required project files are missing: "
            + ", ".join(missing_files)
            + ". Please keep the dataset and saved model artifacts inside the project folder."
        )
        st.stop()


def main() -> None:
    inject_styles()
    validate_required_files()
    dataset = load_dataset()

    render_hero(dataset)

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Dashboard", "Single Prediction", "Batch Prediction", "Feature Guide"]
    )
    with tab1:
        render_overview_tab(dataset)
    with tab2:
        render_single_prediction_tab(dataset)
    with tab3:
        render_batch_prediction_tab(dataset)
    with tab4:
        render_feature_guide_tab()


if __name__ == "__main__":
    main()
