import html
import io
import traceback
from typing import Dict, List, Tuple

import pandas as pd
import plotly.express as px
import streamlit as st

from config import (
    APP_DISCLAIMER,
    APP_TAGLINE,
    APP_TITLE,
    DEFAULT_NER_BATCH_SIZE,
    DEFAULT_PROCESSING_CHUNK_SIZE,
    INPUT_MODES,
    LARGE_DATASET_NER_BATCH_SIZE,
    LARGE_DATASET_PROCESSING_CHUNK_SIZE,
    LARGE_DATASET_THRESHOLD,
    VERY_LARGE_DATASET_NER_BATCH_SIZE,
    VERY_LARGE_DATASET_PROCESSING_CHUNK_SIZE,
    VERY_LARGE_DATASET_THRESHOLD,
    EXTREME_DATASET_NER_BATCH_SIZE,
    EXTREME_DATASET_PROCESSING_CHUNK_SIZE,
    EXTREME_DATASET_THRESHOLD,
    MAX_DATASET_PREVIEW_ROWS,
    MAX_RENDERED_PATIENTS,
    UPLOAD_FILE_TYPES,
    ENABLE_STREAMING_MODE,
)
from evaluation import LABEL_FIELD_ALIASES, compute_metrics, compute_metrics_for_patient_results
from intelligence import process_dataset
from pdf_utils import extract_text_from_uploaded_file
from preprocessing import normalize_patient_id, split_text_into_patient_records
from report_utils import generate_patient_csv, generate_patient_pdf
from utils import build_entity_table, resolve_input_text


TEXT_COLUMN_CANDIDATES = [
    "text",
    "note",
    "notes",
    "clinical_note",
    "clinical_notes",
    "report",
    "description",
    "discharge_summary",
    "summary",
]
PATIENT_ID_CANDIDATES = ["patient_id", "subject_id", "hadm_id", "mrn", "record_id", "id"]


@st.cache_data(show_spinner=False)
def load_evaluation_metrics() -> Dict[str, object]:
    return compute_metrics(run_inference=True)


def _read_uploaded_table(uploaded_file) -> pd.DataFrame:
    file_bytes = uploaded_file.getvalue()
    file_name = str(getattr(uploaded_file, "name", "") or "").lower()

    if file_name.endswith(".xlsx"):
        return pd.read_excel(io.BytesIO(file_bytes))

    if file_name.endswith(".jsonl"):
        return pd.read_json(io.BytesIO(file_bytes), lines=True)

    csv_kwargs = {"low_memory": True}
    if file_name.endswith(".tsv"):
        csv_kwargs["sep"] = "\t"

    try:
        return pd.read_csv(io.BytesIO(file_bytes), **csv_kwargs)
    except UnicodeDecodeError:
        return pd.read_csv(io.BytesIO(file_bytes), encoding="latin-1", **csv_kwargs)


def _build_row_metadata(row_dict: Dict[str, object], row_number: int, id_column: str | None) -> Dict[str, object]:
    metadata: Dict[str, object] = {"row_number": row_number}
    if id_column:
        metadata[id_column] = row_dict.get(id_column)

    label_columns = {alias for aliases in LABEL_FIELD_ALIASES.values() for alias in aliases}
    for column in label_columns:
        if column in row_dict:
            metadata[column] = row_dict.get(column)
    return metadata


def _candidate_text_columns(df: pd.DataFrame) -> List[str]:
    available_columns = {column.lower(): column for column in df.columns}
    matches = [available_columns[name] for name in TEXT_COLUMN_CANDIDATES if name in available_columns]
    if matches:
        return matches

    excluded_columns = set(PATIENT_ID_CANDIDATES)
    for aliases in LABEL_FIELD_ALIASES.values():
        excluded_columns.update(alias.lower() for alias in aliases)

    object_columns = [
        column
        for column in df.columns
        if df[column].dtype == "object" and column.lower() not in excluded_columns
    ]
    return object_columns[:3]


def dataframe_to_patient_records(df: pd.DataFrame, source_name: str = "uploaded_csv") -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    if df.empty:
        return [], {"text_columns": [], "patient_id_column": None}

    text_columns = _candidate_text_columns(df)
    id_column = next((column for column in df.columns if column.lower() in PATIENT_ID_CANDIDATES), None)
    records: List[Dict[str, object]] = []

    for index, row in df.iterrows():
        row_dict = row.to_dict()
        patient_id = normalize_patient_id(row_dict.get(id_column) if id_column else None, index + 1)
        text_parts = []
        for column in text_columns:
            value = row_dict.get(column)
            if pd.isna(value):
                continue
            text = str(value).strip()
            if text:
                text_parts.append(text)

        raw_text = "\n".join(text_parts).strip()
        if not raw_text:
            continue

        records.append(
            {
                "patient_id": patient_id,
                "record_index": len(records) + 1,
                "source": source_name,
                "raw_text": raw_text,
                "metadata": _build_row_metadata(row_dict, index + 1, id_column),
            }
        )

    return records, {"text_columns": text_columns, "patient_id_column": id_column}


def _resolve_processing_settings(record_count: int) -> Tuple[int, int, bool]:
    """
    Resolve optimal processing settings based on dataset size.
    
    Returns: (batch_size, chunk_size, use_streaming_mode)
    """
    if record_count >= EXTREME_DATASET_THRESHOLD:
        return EXTREME_DATASET_NER_BATCH_SIZE, EXTREME_DATASET_PROCESSING_CHUNK_SIZE, ENABLE_STREAMING_MODE and record_count >= 50000
    if record_count >= VERY_LARGE_DATASET_THRESHOLD:
        return VERY_LARGE_DATASET_NER_BATCH_SIZE, VERY_LARGE_DATASET_PROCESSING_CHUNK_SIZE, ENABLE_STREAMING_MODE and record_count >= 20000
    if record_count >= LARGE_DATASET_THRESHOLD:
        return LARGE_DATASET_NER_BATCH_SIZE, LARGE_DATASET_PROCESSING_CHUNK_SIZE, False
    return DEFAULT_NER_BATCH_SIZE, DEFAULT_PROCESSING_CHUNK_SIZE, False


def build_selected_records(
    typed_text: str,
    uploaded_file,
    uploaded_preview_text: str,
    uploaded_dataframe: pd.DataFrame | None,
    input_mode: str,
) -> Tuple[List[Dict[str, object]], str, Dict[str, object]]:
    typed = (typed_text or "").strip()
    typed_records = split_text_into_patient_records(typed, source_name="typed_text") if typed else []
    uploaded_records: List[Dict[str, object]] = []
    upload_meta: Dict[str, object] = {"text_columns": [], "patient_id_column": None}

    if uploaded_file is not None:
        if (uploaded_file.name or "").lower().endswith((".csv", ".tsv", ".xlsx", ".jsonl")):
            uploaded_records, upload_meta = dataframe_to_patient_records(
                uploaded_dataframe if uploaded_dataframe is not None else pd.DataFrame()
            )
        else:
            uploaded_records = (
                split_text_into_patient_records(uploaded_preview_text, source_name="uploaded_file")
                if uploaded_preview_text.strip()
                else []
            )

    selected_text, input_source = resolve_input_text(typed, uploaded_preview_text, input_mode)
    if input_mode == "Use typed text only":
        return typed_records, input_source, upload_meta
    if input_mode == "Use uploaded file only":
        return uploaded_records, input_source, upload_meta
    if not selected_text.strip():
        return [], "combined_inputs", upload_meta
    return typed_records + uploaded_records, "combined_inputs", upload_meta


def _render_model_status(model_meta: Dict[str, object]) -> None:
    st.subheader("Model Status")
    if model_meta.get("available"):
        st.success(
            f"Model loaded: {model_meta['model_name']} ({model_meta['role']}) on {model_meta.get('device', 'cpu')}.",
            icon="✅",
        )
    else:
        st.warning(
            "No biomedical model checkpoint could be loaded in this environment. Dictionary fallback is active.",
            icon="⚠️",
        )
    if model_meta.get("auth_enabled"):
        st.caption("Hugging Face authentication detected. Higher Hub rate limits are enabled.")
    else:
        st.caption("Set the `HF_TOKEN` environment variable to avoid unauthenticated Hugging Face Hub limits.")
    if model_meta.get("failures"):
        with st.expander("Model loading details", expanded=False):
            st.json(model_meta["failures"])


def _render_patient_section(index: int, patient_result: Dict[str, object]) -> None:
    risk_level = patient_result["risk"]["risk_level"]

    # Modern risk color coding
    risk_colors = {
        "High": ("#e53e3e", "#fed7d7", "🚨"),
        "Medium": ("#dd6b20", "#feebc8", "⚠️"),
        "Low": ("#38a169", "#c6f6d5", "✅")
    }
    risk_color, risk_bg, risk_icon = risk_colors.get(risk_level, ("#718096", "#f7fafc", "ℹ️"))

    title = f"{risk_icon} Patient {index}: {patient_result['patient_id']} | Risk: {risk_level}"

    with st.expander(title, expanded=index == 1):
        # Modern metrics cards
        st.markdown(f"""
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 16px; margin: 20px 0;">
            <div style="background: linear-gradient(135deg, {risk_bg} 0%, rgba(255,255,255,0.9) 100%); border-radius: 16px; padding: 20px; text-align: center; box-shadow: 0 4px 20px rgba(0,0,0,0.08); border-left: 4px solid {risk_color};">
                <div style="font-size: 2rem; margin-bottom: 8px;">🏥</div>
                <div style="font-size: 1.5rem; font-weight: 700; color: {risk_color};">{len(patient_result["entities"])}</div>
                <div style="color: #4a5568; font-size: 0.9rem;">Entities</div>
            </div>
            <div style="background: rgba(255,255,255,0.95); border-radius: 16px; padding: 20px; text-align: center; box-shadow: 0 4px 20px rgba(0,0,0,0.08);">
                <div style="font-size: 2rem; margin-bottom: 8px;">📝</div>
                <div style="font-size: 1.5rem; font-weight: 700; color: #2d3748;">{patient_result["preprocessing"]["token_count"]}</div>
                <div style="color: #4a5568; font-size: 0.9rem;">Tokens</div>
            </div>
            <div style="background: rgba(255,255,255,0.95); border-radius: 16px; padding: 20px; text-align: center; box-shadow: 0 4px 20px rgba(0,0,0,0.08);">
                <div style="font-size: 2rem; margin-bottom: 8px;">📏</div>
                <div style="font-size: 1.5rem; font-weight: 700; color: #2d3748;">{patient_result["preprocessing"]["char_count"]}</div>
                <div style="color: #4a5568; font-size: 0.9rem;">Characters</div>
            </div>
            <div style="background: rgba(255,255,255,0.95); border-radius: 16px; padding: 20px; text-align: center; box-shadow: 0 4px 20px rgba(0,0,0,0.08);">
                <div style="font-size: 2rem; margin-bottom: 8px;">📊</div>
                <div style="font-size: 1.5rem; font-weight: 700; color: #2d3748;">{str(patient_result["source"]).replace("_", " ").title()}</div>
                <div style="color: #4a5568; font-size: 0.9rem;">Source</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Modern download buttons
        st.markdown('<div style="display: flex; gap: 16px; margin: 24px 0;">', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label="📄 Download PDF Report",
                data=generate_patient_pdf(patient_result),
                file_name=f"patient_{patient_result['patient_id']}_report.pdf",
                mime="application/pdf",
                key=f"pdf_report_{patient_result['patient_id']}_{index}",
                use_container_width=True,
            )
        with col2:
            st.download_button(
                label="📊 Download CSV Report",
                data=generate_patient_csv(patient_result),
                file_name=f"patient_{patient_result['patient_id']}_report.csv",
                mime="text/csv",
                key=f"csv_report_{patient_result['patient_id']}_{index}",
                use_container_width=True,
            )
        st.markdown('</div>', unsafe_allow_html=True)

        # Modern sections with icons
        st.markdown("### 🔍 Extracted Medical Entities")
        entity_df = build_entity_table(patient_result["entities"])
        if entity_df.empty:
            st.warning("⚠️ No medical entities detected in this patient record.")
        else:
            st.dataframe(entity_df, use_container_width=True)

        st.markdown("### 📈 Risk Assessment")
        risk_explanation = patient_result["risk"]["explanation"]
        if risk_level == "High":
            st.error(f"🚨 {risk_explanation}")
        elif risk_level == "Medium":
            st.warning(f"⚠️ {risk_explanation}")
        else:
            st.success(f"✅ {risk_explanation}")

        st.markdown("### 💡 Clinical Insights")
        for insight in patient_result["insights"]:
            st.markdown(f'<div style="background: rgba(102, 126, 234, 0.1); border-radius: 12px; padding: 12px; margin: 8px 0; border-left: 4px solid #667eea;">• {insight}</div>', unsafe_allow_html=True)

        st.markdown("### 📋 Patient Summary")
        st.info(f"📝 {patient_result['summary']}")

        st.markdown("### 🎨 Highlighted Clinical Notes")
        st.markdown(patient_result["highlighted_html"], unsafe_allow_html=True)

        with st.expander("🔧 Structured JSON Data", expanded=False):
            st.json(
                {
                    "patient_id": patient_result["patient_id"],
                    "risk": patient_result["risk"],
                    "insights": patient_result["insights"],
                    "summary": patient_result["summary"],
                    "structured_data": patient_result["structured_data"],
                }
            )


def _normalize_search_keywords(search_input: str) -> List[str]:
    query = (search_input or "").lower().strip()
    return [keyword.strip() for keyword in query.split(",") if keyword.strip()]


def _patient_search_blob(patient_result: Dict[str, object]) -> str:
    entity_names = [str(entity.get("text", "")).strip() for entity in patient_result.get("entities", [])]
    parts = [
        str(patient_result.get("patient_id", "")),
        str(patient_result.get("raw_text", "")),
        str(patient_result.get("summary", "")),
        " ".join(entity_names),
    ]
    return " ".join(part for part in parts if part).lower()


def _matched_keywords(patient_result: Dict[str, object], keywords: List[str]) -> List[str]:
    if not keywords:
        return []

    searchable_text = _patient_search_blob(patient_result)
    return [keyword for keyword in keywords if keyword in searchable_text]


def _highlight_keyword_matches(keywords: List[str]) -> str:
    if not keywords:
        return ""
    badges = "".join(
        f"<span style='background:#eef5ff;border:1px solid #c9daf8;border-radius:999px;padding:2px 8px;margin-right:6px;'>{html.escape(keyword)}</span>"
        for keyword in keywords
    )
    return f"<div style='margin:6px 0 10px 0;'>Matched keywords: {badges}</div>"


def _ensure_session_defaults() -> None:
    defaults = {
        "analysis_ready": False,
        "analysis_results": None,
        "analysis_input_source": "",
        "analysis_upload_meta": {"text_columns": [], "patient_id_column": None},
        "analysis_uploaded_eval_metrics": None,
        "analysis_signature": None,
        "patient_search_term": "",
        "patient_risk_filter": "All",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _build_input_signature(
    typed_text: str,
    uploaded_file,
    uploaded_text: str,
    uploaded_dataframe: pd.DataFrame | None,
    input_mode: str,
) -> Tuple[object, ...]:
    file_name = ""
    file_size = 0
    if uploaded_file is not None:
        file_name = str(getattr(uploaded_file, "name", "") or "")
        file_size = int(getattr(uploaded_file, "size", 0) or 0)

    dataframe_shape = tuple(uploaded_dataframe.shape) if uploaded_dataframe is not None else None
    return (
        (typed_text or "").strip(),
        file_name,
        file_size,
        (uploaded_text or "").strip(),
        dataframe_shape,
        input_mode,
    )


def run_app() -> None:
    st.set_page_config(
        page_title="MedPehchaan AI+",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': 'https://github.com/your-repo',
            'Report a bug': 'https://github.com/your-repo/issues',
            'About': 'MedPehchaan AI+ - Intelligent Clinical Text Intelligence System'
        }
    )

    # Custom CSS for exceptionally beautiful and modern look
    st.markdown("""
    <style>
    /* Import Google Fonts for modern typography */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Poppins:wght@300;400;500;600;700&display=swap');

    /* Global styles */
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Main background with gradient */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        background-attachment: fixed;
    }

    .stAppViewContainer {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        background-attachment: fixed;
    }

    /* Title styling */
    h1 {
        font-family: 'Poppins', sans-serif;
        font-weight: 700;
        font-size: 3rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 1rem;
        text-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }

    h2, h3 {
        font-family: 'Poppins', sans-serif;
        font-weight: 600;
        color: #2d3748;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }

    /* Modern button styling */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 50px;
        padding: 16px 32px;
        font-size: 16px;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        text-transform: uppercase;
        letter-spacing: 0.5px;
        position: relative;
        overflow: hidden;
    }

    .stButton>button:before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
        transition: left 0.5s;
    }

    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.6);
    }

    .stButton>button:hover:before {
        left: 100%;
    }

    .stButton>button:active {
        transform: translateY(0);
        box-shadow: 0 2px 10px rgba(102, 126, 234, 0.4);
    }

    /* Input fields styling */
    .stTextInput>div>div>input,
    .stTextArea>div>div>textarea,
    .stSelectbox>div>div>div>div>select {
        border-radius: 20px;
        border: 2px solid #e2e8f0;
        padding: 16px 20px;
        background: rgba(255,255,255,0.95);
        backdrop-filter: blur(10px);
        font-size: 16px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
    }

    .stTextInput>div>div>input:focus,
    .stTextArea>div>div>textarea:focus,
    .stSelectbox>div>div>div>div>select:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        transform: translateY(-1px);
    }

    /* File uploader styling */
    .stFileUploader {
        border-radius: 20px;
        border: 2px dashed #cbd5e0;
        background: rgba(255,255,255,0.9);
        backdrop-filter: blur(10px);
        padding: 20px;
        transition: all 0.3s ease;
    }

    .stFileUploader:hover {
        border-color: #667eea;
        background: rgba(255,255,255,0.95);
    }

    /* Metric cards styling */
    .stMetric {
        background: linear-gradient(135deg, rgba(255,255,255,0.95) 0%, rgba(255,255,255,0.9) 100%);
        border-radius: 20px;
        padding: 24px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        border: 1px solid rgba(255,255,255,0.2);
        backdrop-filter: blur(10px);
        transition: transform 0.3s ease;
    }

    .stMetric:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 40px rgba(0,0,0,0.15);
    }

    /* Container styling */
    .stContainer {
        background: rgba(255,255,255,0.95);
        border-radius: 24px;
        padding: 32px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.2);
        margin: 20px 0;
    }

    /* Expander styling */
    .stExpander {
        background: rgba(255,255,255,0.95);
        border-radius: 16px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        border: 1px solid rgba(255,255,255,0.2);
        margin: 10px 0;
        overflow: hidden;
    }

    .stExpander > div:first-child {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        padding: 16px 20px;
        border-radius: 16px 16px 0 0;
    }

    /* Dataframe styling */
    .stDataFrame {
        border-radius: 16px;
        overflow: hidden;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
    }

    /* Progress bar styling */
    .stProgress > div > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }

    /* Alert styling */
    .stAlert {
        border-radius: 16px;
        border: none;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        backdrop-filter: blur(10px);
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(255,255,255,0.95);
        border-radius: 16px 16px 0 0;
        padding: 8px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 12px;
        margin: 0 4px;
        transition: all 0.3s ease;
    }

    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }

    /* Radio button styling */
    .stRadio > div {
        background: rgba(255,255,255,0.95);
        border-radius: 16px;
        padding: 16px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
    }

    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }

    /* Success/Warning/Error styling */
    .stSuccess, .stWarning, .stError, .stInfo {
        border-radius: 16px;
        border: none;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        backdrop-filter: blur(10px);
    }

    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }

    ::-webkit-scrollbar-track {
        background: rgba(255,255,255,0.1);
        border-radius: 10px;
    }

    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }

    /* Animation for loading */
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }

    .stSpinner > div > div {
        animation: pulse 1.5s ease-in-out infinite;
    }

    /* Chart styling */
    .js-plotly-plot {
        border-radius: 16px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        overflow: hidden;
    }

    /* Download button styling */
    .stDownloadButton>button {
        background: linear-gradient(135deg, #48bb78 0%, #38a169 100%);
        border-radius: 50px;
        padding: 12px 24px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(72, 187, 120, 0.4);
    }

    .stDownloadButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(72, 187, 120, 0.6);
    }

    /* Caption styling */
    .stCaption {
        color: #718096;
        font-style: italic;
        text-align: center;
        margin: 16px 0;
    }

    /* Header gradient text */
    .gradient-text {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 700;
        font-size: 2.5rem;
    }

    /* Card hover effects */
    .hover-card {
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }

    .hover-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 40px rgba(0,0,0,0.15);
    }

    /* Modern form styling */
    .modern-form {
        background: rgba(255,255,255,0.95);
        border-radius: 24px;
        padding: 32px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.2);
        margin: 20px 0;
    }

    /* Floating animation */
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
    }

    .float-animation {
        animation: float 3s ease-in-out infinite;
    }

    </style>
    """, unsafe_allow_html=True)

    _ensure_session_defaults()

    # Modern hero section
    st.markdown("""
    <div style="text-align: center; padding: 40px 20px; margin-bottom: 40px;">
        <div class="float-animation">
            <h1 style="font-size: 4rem; margin-bottom: 10px;">🏥 MedPehchaan AI+</h1>
            <p style="font-size: 1.3rem; color: #4a5568; margin-top: 10px; font-weight: 300;">
                Intelligent Clinical Text Intelligence System
            </p>
        </div>
        <div style="margin-top: 30px;">
            <div style="display: inline-block; background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%); padding: 20px; border-radius: 20px; backdrop-filter: blur(10px);">
                <p style="color: #2d3748; font-size: 1.1rem; margin: 0; font-weight: 500;">
                    ⚠️ <strong>Medical Disclaimer:</strong> This tool is for research and educational purposes only. Not for clinical diagnosis.
                </p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        """
        <div style="background: rgba(255,255,255,0.9); border-radius: 20px; padding: 24px; margin: 20px 0; box-shadow: 0 4px 20px rgba(0,0,0,0.08);">
        <h3 style="color: #2d3748; margin-top: 0;">🔬 How It Works</h3>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-top: 20px;">
            <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%); border-radius: 16px;">
                <div style="font-size: 2rem; margin-bottom: 10px;">📝</div>
                <strong>Split & Clean</strong><br>
                <small>Process patient records individually</small>
            </div>
            <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, rgba(72, 187, 120, 0.1) 0%, rgba(56, 161, 105, 0.1) 100%); border-radius: 16px;">
                <div style="font-size: 2rem; margin-bottom: 10px;">🔍</div>
                <strong>Extract Entities</strong><br>
                <small>Biomedical NER with AI</small>
            </div>
            <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, rgba(237, 137, 54, 0.1) 0%, rgba(221, 107, 32, 0.1) 100%); border-radius: 16px;">
                <div style="font-size: 2rem; margin-bottom: 10px;">📊</div>
                <strong>Analyze & Risk Assess</strong><br>
                <small>Generate insights & summaries</small>
            </div>
            <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, rgba(229, 62, 62, 0.1) 0%, rgba(197, 48, 48, 0.1) 100%); border-radius: 16px;">
                <div style="font-size: 2rem; margin-bottom: 10px;">📋</div>
                <strong>Report Generation</strong><br>
                <small>Comprehensive patient reports</small>
            </div>
        </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    with st.container():
        st.markdown('<div class="modern-form">', unsafe_allow_html=True)
        st.markdown('<h2 style="color: #2d3748; text-align: center; margin-bottom: 30px;">📥 Input Your Clinical Data</h2>', unsafe_allow_html=True)

        left, right = st.columns([2, 1])

        with left:
            st.markdown("### ✍️ Type or Paste Text")
            typed_text = st.text_area(
                "",
                key="typed_text_input",
                height=250,
                placeholder=(
                    "Patient ID: 1001\nDiabetes with chest pain. Aspirin prescribed.\n\n"
                    "Patient ID: 1002\nAsthma with fatigue. ECG advised."
                ),
                label_visibility="collapsed"
            )

        uploaded_text = ""
        uploaded_dataframe = None
        uploaded_file = None
        with right:
            st.markdown("### 📎 Upload File")
            uploaded_file = st.file_uploader(
                "",
                key="uploaded_file_input",
                type=UPLOAD_FILE_TYPES,
                help="Supports: TXT, PDF, CSV, TSV, XLSX, JSONL",
                label_visibility="collapsed"
            )

            if uploaded_file is not None:
                file_name = (uploaded_file.name or "").lower()
                try:
                    if file_name.endswith((".csv", ".tsv", ".xlsx", ".jsonl")):
                        uploaded_dataframe = _read_uploaded_table(uploaded_file)
                        st.success(f"✅ Dataset loaded: {len(uploaded_dataframe)} rows")
                        if len(uploaded_dataframe) >= LARGE_DATASET_THRESHOLD:
                            st.info("💡 Large dataset detected. Optimized processing enabled.")
                        with st.expander("👀 Preview Dataset", expanded=False):
                            st.dataframe(
                                uploaded_dataframe.head(MAX_DATASET_PREVIEW_ROWS),
                                use_container_width=True,
                            )
                    else:
                        uploaded_text = extract_text_from_uploaded_file(uploaded_file)
                        st.success("✅ File processed successfully")
                        with st.expander("👀 Preview Text", expanded=False):
                            st.write((uploaded_text[:1200] + "...") if len(uploaded_text) > 1200 else uploaded_text)
                except Exception as exc:
                    st.error(f"❌ Error reading file: {exc}")

        # Input mode selection with better styling
        if typed_text.strip() and (uploaded_text.strip() or uploaded_dataframe is not None):
            st.markdown("### 🎯 Input Mode")
            input_mode = st.radio(
                "",
                INPUT_MODES,
                key="input_mode_selection",
                help="Choose how to combine your inputs",
                label_visibility="collapsed"
            )
        elif typed_text.strip():
            input_mode = "Use typed text only"
            st.info("📝 Using typed text only")
        elif uploaded_text.strip() or uploaded_dataframe is not None:
            input_mode = "Use uploaded file only"
            st.info("📎 Using uploaded file only")
        else:
            input_mode = "Use typed text only"

        # Modern analyze button
        st.markdown('<div style="text-align: center; margin-top: 30px;">', unsafe_allow_html=True)
        analyze = st.button("🚀 Analyze Patients", type="primary", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    current_input_signature = _build_input_signature(
        typed_text=typed_text,
        uploaded_file=uploaded_file,
        uploaded_text=uploaded_text,
        uploaded_dataframe=uploaded_dataframe,
        input_mode=input_mode,
    )

    if analyze:
        try:
            patient_records, input_source, upload_meta = build_selected_records(
                typed_text=typed_text,
                uploaded_file=uploaded_file,
                uploaded_preview_text=uploaded_text,
                uploaded_dataframe=uploaded_dataframe,
                input_mode=input_mode,
            )

            if not patient_records:
                st.session_state["analysis_ready"] = False
                st.session_state["analysis_results"] = None
                st.warning("No patient records found. Please enter text or upload a valid TXT, PDF, or CSV file.")
                return

            batch_size, processing_chunk_size, use_streaming = _resolve_processing_settings(len(patient_records))
            with st.spinner("Processing patient records independently..."):
                progress_bar = st.progress(0.0)
                progress_text = st.empty()

                def _on_progress(chunk_index: int, total_chunks: int, processed_count: int) -> None:
                    progress_bar.progress(
                        min(chunk_index / max(total_chunks, 1), 1.0),
                        text=f"Processed {processed_count} / {len(patient_records)} patient records",
                    )
                    progress_text.caption(
                        f"Chunk {chunk_index} of {total_chunks} completed using NER batch size {batch_size}."
                    )

                # Use streaming mode for very large datasets
                if use_streaming:
                    from intelligence import process_dataset_streaming
                    patient_list = []
                    aggregate_state = {}
                    model_meta = {}
                    
                    for patient_result in process_dataset_streaming(
                        patient_records,
                        batch_size=batch_size,
                        processing_chunk_size=processing_chunk_size,
                        progress_callback=_on_progress,
                    ):
                        patient_list.append(patient_result)
                    
                    # This will contain the final aggregate report
                    results = {
                        "patients": patient_list,
                        "aggregate_report": {"total_patients_processed": len(patient_list)},
                        "model_meta": model_meta,
                    }
                else:
                    from intelligence import process_dataset
                    results = process_dataset(
                        patient_records,
                        batch_size=batch_size,
                        processing_chunk_size=processing_chunk_size,
                        progress_callback=_on_progress,
                    )
                progress_bar.empty()
                progress_text.empty()
                uploaded_eval_metrics = compute_metrics_for_patient_results(results["patients"])

            st.session_state["analysis_ready"] = True
            st.session_state["analysis_results"] = results
            st.session_state["analysis_input_source"] = input_source
            st.session_state["analysis_upload_meta"] = upload_meta
            st.session_state["analysis_uploaded_eval_metrics"] = uploaded_eval_metrics
            st.session_state["analysis_signature"] = current_input_signature
            st.session_state["patient_search_term"] = ""
            st.session_state["patient_risk_filter"] = "All"
        except Exception as exc:
            st.session_state["analysis_ready"] = False
            st.session_state["analysis_results"] = None
            st.error("The app hit an unexpected error while processing this input.")
            st.code(f"{type(exc).__name__}: {exc}")
            st.text(traceback.format_exc())
            return

    if not st.session_state["analysis_ready"]:
        return

    if st.session_state["analysis_signature"] != current_input_signature:
        st.info("Inputs changed after the last analysis. Click `Analyze Patients` again to refresh the results.")
        return

    try:
        results = st.session_state["analysis_results"]
        input_source = st.session_state["analysis_input_source"]
        upload_meta = st.session_state["analysis_upload_meta"]
        uploaded_eval_metrics = st.session_state["analysis_uploaded_eval_metrics"]

        st.markdown('<h2 style="text-align: center; margin-bottom: 30px;">📊 Processing Overview</h2>', unsafe_allow_html=True)
        overview_cols = st.columns(4)
        metrics_data = [
            (results["aggregate_report"]["total_patients_processed"], "👥 Patients", "#667eea"),
            (results["aggregate_report"]["total_diseases_detected"], "🦠 Diseases", "#e53e3e"),
            (results["aggregate_report"]["patients_with_no_entities"], "📋 No-Entity Patients", "#718096"),
            (input_source.replace("_", " ").title(), "📥 Input Source", "#38a169")
        ]

        for i, (value, label, color) in enumerate(metrics_data):
            with overview_cols[i]:
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, rgba(255,255,255,0.95) 0%, rgba(255,255,255,0.9) 100%); border-radius: 20px; padding: 24px; text-align: center; box-shadow: 0 8px 32px rgba(0,0,0,0.1); border: 1px solid rgba(255,255,255,0.2);">
                    <div style="font-size: 2.5rem; margin-bottom: 12px;">{label.split()[0]}</div>
                    <div style="font-size: 2rem; font-weight: 700; color: {color}; margin-bottom: 8px;">{value}</div>
                    <div style="color: #4a5568; font-size: 0.9rem; font-weight: 500;">{label}</div>
                </div>
                """, unsafe_allow_html=True)

        if upload_meta["text_columns"]:
            st.caption(
                f"📋 CSV text columns used: {', '.join(upload_meta['text_columns'])}"
                + (f" | 🆔 Patient ID column: {upload_meta['patient_id_column']}" if upload_meta["patient_id_column"] else "")
            )

        _render_model_status(results["model_meta"])

        # Modern tabs
        patient_tab, aggregate_tab = st.tabs(["👤 Patient-wise View", "📈 Aggregate Analysis"])

        with patient_tab:
            st.markdown('<div style="background: rgba(255,255,255,0.95); border-radius: 16px; padding: 24px; margin: 20px 0; box-shadow: 0 4px 20px rgba(0,0,0,0.08);">', unsafe_allow_html=True)
            filter_left, filter_right = st.columns([2, 1])
            with filter_left:
                st.markdown("### 🔍 Search & Filter")
                search_term = st.text_input(
                    "",
                    key="patient_search_term",
                    placeholder="Search by patient ID, symptoms, or conditions...",
                    label_visibility="collapsed"
                )
            with filter_right:
                risk_filter = st.selectbox(
                    "",
                    ["All", "High", "Medium", "Low"],
                    key="patient_risk_filter",
                    label_visibility="collapsed"
                )
            st.markdown('</div>', unsafe_allow_html=True)

            filtered_results = []
            keywords = _normalize_search_keywords(search_term)
            for patient_result in results["patients"]:
                matched_keywords = _matched_keywords(patient_result, keywords)
                matches_query = not keywords or bool(matched_keywords)
                matches_risk = risk_filter == "All" or patient_result["risk"]["risk_level"] == risk_filter
                if matches_query and matches_risk:
                    patient_with_match_meta = dict(patient_result)
                    patient_with_match_meta["search_matches"] = matched_keywords
                    patient_with_match_meta["highlighted_html"] = (
                        _highlight_keyword_matches(matched_keywords) + patient_result["highlighted_html"]
                    )
                    filtered_results.append(patient_with_match_meta)

            if not filtered_results:
                st.warning("🔍 No matching patients found with the current filters.")
            else:
                if len(filtered_results) > MAX_RENDERED_PATIENTS:
                    st.info(
                        f"📋 Showing first {MAX_RENDERED_PATIENTS} of {len(filtered_results)} matching patients."
                    )
                for index, patient_result in enumerate(filtered_results[:MAX_RENDERED_PATIENTS], start=1):
                    _render_patient_section(index, patient_result)

        with aggregate_tab:
            st.markdown("### 📊 Comprehensive Analysis Dashboard")

            # Risk distribution section
            risk_df = pd.DataFrame(
                [
                    {"Risk Level": level, "Patients": count}
                    for level, count in results["aggregate_report"]["overall_risk_distribution"].items()
                ]
            )
            symptom_df = pd.DataFrame(results["aggregate_report"]["most_common_symptoms"])

            risk_cols = st.columns([1, 1])
            with risk_cols[0]:
                st.markdown("#### 🎯 Risk Distribution")
                st.dataframe(risk_df, use_container_width=True, hide_index=True)
                if not risk_df.empty:
                    st.bar_chart(risk_df.set_index("Risk Level"), color="#667eea")

            with risk_cols[1]:
                st.markdown("#### 🩺 Most Common Symptoms")
                if symptom_df.empty:
                    st.info("ℹ️ No symptoms detected in the processed patients.")
                else:
                    st.dataframe(symptom_df, use_container_width=True, hide_index=True)

            # Patient summary table
            st.markdown("### 📋 Patient Summary Overview")
            summary_rows = []
            for patient_result in results["patients"]:
                summary_rows.append(
                    {
                        "Patient ID": patient_result["patient_id"],
                        "Risk Level": patient_result["risk"]["risk_level"],
                        "Diseases": len(patient_result["structured_data"]["diseases"]),
                        "Symptoms": len(patient_result["structured_data"]["symptoms"]),
                        "Medications": len(patient_result["structured_data"]["medications"]),
                        "Procedures": len(patient_result["structured_data"]["procedures"]),
                        "Summary": patient_result["summary"],
                    }
                )
            st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)

            # Evaluation metrics with modern styling
            evaluation_metrics = uploaded_eval_metrics or {}
            if uploaded_eval_metrics:
                st.markdown("### 🎯 Model Performance Metrics")
                metrics_cols = st.columns(5)
                metrics_info = [
                    (uploaded_eval_metrics['precision'], "Precision", "#667eea"),
                    (uploaded_eval_metrics['recall'], "Recall", "#38a169"),
                    (uploaded_eval_metrics['f1_score'], "F1 Score", "#dd6b20"),
                    (uploaded_eval_metrics['accuracy'], "Accuracy", "#e53e3e"),
                    (uploaded_eval_metrics.get('exact_match_rate', 0.0), "Exact Match", "#805ad5")
                ]

                for i, (value, label, color) in enumerate(metrics_info):
                    with metrics_cols[i]:
                        st.markdown(f"""
                        <div style="background: linear-gradient(135deg, rgba(255,255,255,0.95) 0%, rgba(255,255,255,0.9) 100%); border-radius: 16px; padding: 20px; text-align: center; box-shadow: 0 4px 20px rgba(0,0,0,0.08);">
                            <div style="font-size: 1.8rem; font-weight: 700; color: {color}; margin-bottom: 8px;">{value:.1f}%</div>
                            <div style="color: #4a5568; font-size: 0.9rem; font-weight: 500;">{label}</div>
                        </div>
                        """, unsafe_allow_html=True)

                # Enhanced metrics chart
                fig = px.bar(
                    x=['Precision', 'Recall', 'F1 Score', 'Accuracy', 'Exact Match'],
                    y=[
                        uploaded_eval_metrics['precision'],
                        uploaded_eval_metrics['recall'],
                        uploaded_eval_metrics['f1_score'],
                        uploaded_eval_metrics['accuracy'],
                        uploaded_eval_metrics.get('exact_match_rate', 0.0),
                    ],
                    title="📊 Model Evaluation Metrics",
                    labels={'x': 'Metric', 'y': 'Score (%)'},
                    color=['Precision', 'Recall', 'F1 Score', 'Accuracy', 'Exact Match'],
                    color_discrete_sequence=px.colors.qualitative.Vivid
                )
                fig.update_layout(
                    plot_bgcolor='rgba(255,255,255,0.9)',
                    paper_bgcolor='rgba(255,255,255,0)',
                    margin=dict(t=50, r=20, b=20, l=20),
                    yaxis=dict(range=[0, 100]),
                    font=dict(size=14),
                    title_font=dict(size=20, color='#2d3748')
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info(
                    "💡 Upload evaluation data with gold standard labels to see detailed performance metrics."
                )

            with st.expander("🔧 Complete JSON Report", expanded=False):
                st.json(
                    {
                        "aggregate_report": results["aggregate_report"],
                        "patients": [
                            {
                                "patient_id": patient_result["patient_id"],
                                "structured_data": patient_result["structured_data"],
                                "risk": patient_result["risk"],
                                "insights": patient_result["insights"],
                                "summary": patient_result["summary"],
                            }
                            for patient_result in results["patients"]
                        ],
                        "evaluation": evaluation_metrics,
                    }
                )

    except Exception as exc:
        st.error("The app hit an unexpected error while processing this input.")
        st.code(f"{type(exc).__name__}: {exc}")
        st.text(traceback.format_exc())
