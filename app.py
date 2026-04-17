# App.py  –  Website Prediksi Tingkat Stres Mahasiswa
# Light UI – Clean, Compact, Informatif

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import builtins
import ast
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ──────────────────────────────────────────────
# PAGE CONFIG (harus pertama)
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="StressCheck – Prediksi Tingkat Stres",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# CUSTOM CSS – Light Theme
# ──────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

/* ── Reset & Base ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    color: #1a1f2e;
}

.stApp {
    background: #f5f6fa;
    min-height: 100vh;
}

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 3rem;
    max-width: 1100px;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 1px solid #e5e8ef;
    box-shadow: 2px 0 8px rgba(0,0,0,0.04);
}

section[data-testid="stSidebar"] label {
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    color: #4b5563 !important;
}

section[data-testid="stSidebar"] .stSlider > div > div > div {
    background: #3b82f6 !important;
}

/* Sidebar header */
.sidebar-header {
    padding: 0.25rem 0 1.25rem 0;
    border-bottom: 1px solid #e5e8ef;
    margin-bottom: 1rem;
}
.sidebar-app-name {
    font-size: 1rem;
    font-weight: 700;
    color: #1a1f2e;
    letter-spacing: -0.01em;
}
.sidebar-tagline {
    font-size: 0.75rem;
    color: #6b7280;
    margin-top: 0.2rem;
}

/* ── Hero ── */
.hero {
    background: #ffffff;
    border: 1px solid #e5e8ef;
    border-radius: 14px;
    padding: 1.5rem 2rem;
    margin-bottom: 1.25rem;
    display: flex;
    align-items: center;
    gap: 1.5rem;
}
.hero-text h1 {
    font-size: 1.5rem;
    font-weight: 700;
    color: #1a1f2e;
    margin: 0 0 0.25rem 0;
    letter-spacing: -0.02em;
}
.hero-text p {
    font-size: 0.85rem;
    color: #6b7280;
    margin: 0;
    line-height: 1.6;
    max-width: 580px;
}
.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: #eff6ff;
    border: 1px solid #bfdbfe;
    color: #2563eb;
    font-size: 0.72rem;
    font-weight: 600;
    padding: 3px 10px;
    border-radius: 20px;
    margin-bottom: 0.6rem;
    font-family: 'DM Mono', monospace;
}
.hero-icon {
    font-size: 2.5rem;
    flex-shrink: 0;
    opacity: 0.8;
}

/* ── Instruction Steps ── */
.steps-row {
    display: flex;
    gap: 0.75rem;
    margin-bottom: 1.25rem;
}
.step-item {
    flex: 1;
    background: #ffffff;
    border: 1px solid #e5e8ef;
    border-radius: 10px;
    padding: 0.85rem 1rem;
    display: flex;
    align-items: flex-start;
    gap: 0.75rem;
}
.step-num {
    width: 22px;
    height: 22px;
    background: #3b82f6;
    color: #fff;
    font-size: 0.72rem;
    font-weight: 700;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
    margin-top: 1px;
}
.step-title {
    font-size: 0.82rem;
    font-weight: 600;
    color: #1a1f2e;
    margin-bottom: 0.15rem;
}
.step-desc {
    font-size: 0.75rem;
    color: #6b7280;
    line-height: 1.4;
}

/* ── Section Headers ── */
.section-header {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin-bottom: 1rem;
    padding-bottom: 0.6rem;
    border-bottom: 1px solid #e5e8ef;
}
.section-tag {
    background: #eff6ff;
    color: #2563eb;
    font-size: 0.7rem;
    font-weight: 700;
    padding: 2px 8px;
    border-radius: 4px;
    font-family: 'DM Mono', monospace;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.section-title {
    font-size: 1rem;
    font-weight: 700;
    color: #1a1f2e;
    margin: 0;
}
.section-subtitle {
    font-size: 0.78rem;
    color: #6b7280;
    margin-left: auto;
}

/* ── Result Card ── */
.result-card {
    border-radius: 12px;
    padding: 1.5rem 1.75rem;
    text-align: center;
    border: 1.5px solid;
}
.result-card-low    { background: #f0fdf4; border-color: #86efac; }
.result-card-medium { background: #fffbeb; border-color: #fcd34d; }
.result-card-high   { background: #fff1f2; border-color: #fca5a5; }

.result-level {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 0.3rem;
    font-family: 'DM Mono', monospace;
}
.result-level-low    { color: #16a34a; }
.result-level-medium { color: #d97706; }
.result-level-high   { color: #dc2626; }

.result-label {
    font-size: 1.4rem;
    font-weight: 700;
    color: #1a1f2e;
    margin: 0 0 0.4rem 0;
    letter-spacing: -0.02em;
}
.result-desc {
    font-size: 0.82rem;
    color: #4b5563;
    line-height: 1.5;
    margin: 0;
}

/* ── Probability Bars ── */
.prob-container {
    background: #ffffff;
    border: 1px solid #e5e8ef;
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
}
.prob-title {
    font-size: 0.72rem;
    font-weight: 700;
    color: #6b7280;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 1rem;
    font-family: 'DM Mono', monospace;
}
.prob-row {
    margin-bottom: 0.9rem;
}
.prob-label-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.3rem;
}
.prob-label     { font-size: 0.82rem; color: #374151; font-weight: 500; }
.prob-label-bold { font-weight: 700; color: #1a1f2e; }
.prob-pct       { font-family: 'DM Mono', monospace; font-size: 0.82rem; font-weight: 600; color: #1a1f2e; }
.prob-bar-bg    { background: #e5e8ef; border-radius: 4px; height: 8px; overflow: hidden; }
.prob-bar-low    { height: 100%; background: #22c55e; border-radius: 4px; }
.prob-bar-medium { height: 100%; background: #f59e0b; border-radius: 4px; }
.prob-bar-high   { height: 100%; background: #ef4444; border-radius: 4px; }

.confidence-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    margin-top: 0.75rem;
    padding: 5px 12px;
    background: #f0fdf4;
    border: 1px solid #bbf7d0;
    border-radius: 6px;
    font-size: 0.78rem;
    color: #15803d;
    font-weight: 600;
    font-family: 'DM Mono', monospace;
}

/* ── SHAP Intro Card ── */
.shap-info-card {
    background: #eff6ff;
    border: 1px solid #bfdbfe;
    border-radius: 10px;
    padding: 1rem 1.25rem;
    margin-bottom: 1.25rem;
    font-size: 0.83rem;
    color: #374151;
    line-height: 1.6;
}
.shap-info-title {
    font-weight: 700;
    color: #1d4ed8;
    margin-bottom: 0.35rem;
    font-size: 0.85rem;
}
.shap-legend {
    display: flex;
    gap: 1.25rem;
    margin-top: 0.6rem;
    flex-wrap: wrap;
}
.shap-legend-item {
    display: flex;
    align-items: center;
    gap: 0.4rem;
    font-size: 0.8rem;
    color: #374151;
}
.dot-red  { width: 11px; height: 11px; background: #ef4444; border-radius: 2px; flex-shrink: 0; }
.dot-blue { width: 11px; height: 11px; background: #3b82f6; border-radius: 2px; flex-shrink: 0; }

/* ── Driver Cards ── */
.driver-group-title {
    font-size: 0.78rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    margin-bottom: 0.6rem;
    font-family: 'DM Mono', monospace;
}
.driver-group-title-red  { color: #dc2626; }
.driver-group-title-blue { color: #2563eb; }

.driver-card {
    background: #ffffff;
    border: 1px solid #e5e8ef;
    border-radius: 8px;
    padding: 0.65rem 0.85rem;
    margin-bottom: 0.45rem;
    display: flex;
    align-items: center;
    gap: 0.7rem;
    border-left: 3px solid transparent;
}
.driver-card-up   { border-left-color: #ef4444; }
.driver-card-down { border-left-color: #3b82f6; }

.driver-rank {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    color: #9ca3af;
    min-width: 18px;
}
.driver-name {
    font-size: 0.82rem;
    font-weight: 500;
    color: #1a1f2e;
    flex: 1;
    line-height: 1.3;
}
.driver-pill {
    font-size: 0.68rem;
    font-weight: 700;
    padding: 2px 8px;
    border-radius: 10px;
}
.driver-pill-up   { background: #fee2e2; color: #dc2626; }
.driver-pill-down { background: #dbeafe; color: #2563eb; }

.driver-val {
    font-family: 'DM Mono', monospace;
    font-size: 0.76rem;
    color: #6b7280;
    min-width: 50px;
    text-align: right;
}

/* ── Info Box ── */
.info-box {
    background: #f9fafb;
    border: 1px solid #e5e8ef;
    border-radius: 8px;
    padding: 0.8rem 1rem;
    font-size: 0.81rem;
    color: #4b5563;
    line-height: 1.6;
}
.info-box strong { color: #1a1f2e; }

/* ── Metric Cards (engineered features) ── */
.metric-mini {
    background: #ffffff;
    border: 1px solid #e5e8ef;
    border-radius: 8px;
    padding: 0.85rem 1rem;
    margin-bottom: 0.5rem;
}
.metric-mini-label {
    font-size: 0.72rem;
    font-weight: 600;
    color: #6b7280;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 0.2rem;
}
.metric-mini-val {
    font-family: 'DM Mono', monospace;
    font-size: 1.1rem;
    font-weight: 600;
    color: #1a1f2e;
}

/* ── Streamlit overrides ── */
.stDataFrame      { border-radius: 8px; overflow: hidden; border: 1px solid #e5e8ef; }
div[data-testid="stMetric"] { background: #fff; border: 1px solid #e5e8ef; border-radius: 8px; padding: 0.75rem; }
.stAlert          { border-radius: 8px; }
hr                { border-color: #e5e8ef !important; }
.stPlotlyChart, .stPyplot { background: transparent !important; }

/* Expander styling in sidebar */
section[data-testid="stSidebar"] details {
    background: #f9fafb;
    border: 1px solid #e5e8ef;
    border-radius: 8px;
    margin-bottom: 0.5rem;
}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# LOAD ARTIFACTS
# ──────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model             = joblib.load("xgb_model.pkl")
    scaler            = joblib.load("scaler.pkl")
    label_encoder     = joblib.load("label_encoder.pkl")
    selected_features = joblib.load("selected_features.pkl")
    return model, scaler, label_encoder, selected_features

model, scaler, le, selected_features = load_artifacts()

STRESS_CONFIG = {
    0: {
        "label": "Stres Rendah",
        "level": "LEVEL RENDAH",
        "css_class": "result-card-low",
        "level_class": "result-level-low",
        "color": "#22c55e",
        "bar_class": "prob-bar-low",
        "desc": "Kondisi mental kamu tergolong baik. Pertahankan pola istirahat, aktivitas, dan dukungan sosialmu.",
    },
    1: {
        "label": "Stres Sedang",
        "level": "LEVEL SEDANG",
        "css_class": "result-card-medium",
        "level_class": "result-level-medium",
        "color": "#f59e0b",
        "bar_class": "prob-bar-medium",
        "desc": "Ada beberapa faktor pemicu stres. Perhatikan kualitas tidur dan manajemen waktu belajarmu.",
    },
    2: {
        "label": "Stres Tinggi",
        "level": "LEVEL TINGGI",
        "css_class": "result-card-high",
        "level_class": "result-level-high",
        "color": "#ef4444",
        "bar_class": "prob-bar-high",
        "desc": "Tingkat stres cukup tinggi. Pertimbangkan untuk berbicara dengan konselor atau profesional kesehatan jiwa.",
    },
}

FEATURE_LABELS = {
    "anxiety_level":               "Tingkat Kecemasan",
    "self_esteem":                 "Harga Diri",
    "mental_health_history":       "Riwayat Masalah Mental",
    "depression":                  "Tingkat Depresi",
    "headache":                    "Frekuensi Sakit Kepala",
    "blood_pressure":              "Tekanan Darah",
    "sleep_quality":               "Kualitas Tidur",
    "breathing_problem":           "Masalah Pernapasan",
    "noise_level":                 "Kebisingan Lingkungan",
    "living_conditions":           "Kondisi Tempat Tinggal",
    "safety":                      "Rasa Aman",
    "basic_needs":                 "Kebutuhan Dasar",
    "academic_performance":        "Performa Akademik",
    "study_load":                  "Beban Belajar",
    "teacher_student_relationship":"Hub. Dosen-Mahasiswa",
    "future_career_concerns":      "Kekhawatiran Karir",
    "social_support":              "Dukungan Sosial",
    "peer_pressure":               "Tekanan Teman Sebaya",
    "extracurricular_activities":  "Aktivitas Ekskul",
    "bullying":                    "Perundungan",
    "academic_stress_index":       "Indeks Stres Akademik",
    "environment_quality_index":   "Indeks Kualitas Lingkungan",
    "social_stress_score":         "Skor Stres Sosial",
}


# ──────────────────────────────────────────────
# SHAP Monkey-Patch (XGBoost 3.x multiclass)
# ──────────────────────────────────────────────
def patch_shap_for_xgb_multiclass():
    import shap.explainers._tree as _tree_mod
    _OrigLoader = _tree_mod.XGBTreeModelLoader
    _orig_init  = _OrigLoader.__init__
    _orig_float = builtins.float
    if getattr(_OrigLoader, "_patched_for_multiclass", False):
        return

    class _ArrayAwareFloat(float):
        def __new__(cls, x=0):
            if isinstance(x, str):
                try:
                    return _orig_float.__new__(cls, x)
                except (ValueError, TypeError):
                    try:
                        arr = ast.literal_eval(x)
                        return _orig_float.__new__(cls, np.mean(arr))
                    except Exception:
                        return _orig_float.__new__(cls, 0.5)
            return _orig_float.__new__(cls, x)

    def _patched_init(self, xgb_model):
        builtins.float = _ArrayAwareFloat
        try:
            _orig_init(self, xgb_model)
        finally:
            builtins.float = _orig_float

    _OrigLoader.__init__ = _patched_init
    _OrigLoader._patched_for_multiclass = True


@st.cache_resource
def get_shap_explainer(_model):
    import shap
    patch_shap_for_xgb_multiclass()
    return shap.TreeExplainer(_model)


# ──────────────────────────────────────────────
# NORMALISASI & INPUT BUILDER
# ──────────────────────────────────────────────
MINMAX_RANGE = {
    "study_load": (0, 5), "future_career_concerns": (0, 5),
    "academic_performance": (0, 5), "peer_pressure": (0, 5),
    "bullying": (0, 5), "social_support": (0, 3),
}

def minmax_norm_single(value, feat):
    lo, hi = MINMAX_RANGE[feat]
    return (value - lo) / (hi - lo) if hi != lo else 0.0

def build_input_row(raw):
    w_study, w_career, w_acad = 0.6342, 0.7426, 0.7209
    total_w = w_study + w_career + w_acad
    academic_stress_index = (
        (w_study  / total_w) * minmax_norm_single(raw["study_load"], "study_load")
        + (w_career / total_w) * minmax_norm_single(raw["future_career_concerns"], "future_career_concerns")
        + (w_acad   / total_w) * (1 - minmax_norm_single(raw["academic_performance"], "academic_performance"))
    )
    environment_quality_index = (
        raw["noise_level"]
        + (5 - raw["living_conditions"])
        + (5 - raw["safety"])
        + (5 - raw["basic_needs"])
    )
    social_stress_score = (
        minmax_norm_single(raw["peer_pressure"], "peer_pressure")
        + minmax_norm_single(raw["bullying"], "bullying")
        + (1 - minmax_norm_single(raw["social_support"], "social_support"))
    )
    full = {**raw,
            "academic_stress_index": academic_stress_index,
            "environment_quality_index": environment_quality_index,
            "social_stress_score": social_stress_score}
    return pd.DataFrame([full])[selected_features]


# ══════════════════════════════════════════════
# SIDEBAR INPUT
# ══════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div class="sidebar-header">
        <div class="sidebar-app-name">StressCheck</div>
        <div class="sidebar-tagline">Prediksi tingkat stres mahasiswa berbasis AI</div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("Psikologis", expanded=True):
        anxiety_level         = st.slider("Tingkat Kecemasan",  0, 21, 10, help="0 = tidak cemas sama sekali, 21 = sangat cemas")
        self_esteem           = st.slider("Harga Diri",          0, 30, 15, help="0 = sangat rendah, 30 = sangat tinggi")
        mental_health_history = st.selectbox("Riwayat Masalah Mental", [0, 1],
                                              format_func=lambda x: "Tidak ada" if x == 0 else "Ada riwayat")
        depression            = st.slider("Tingkat Depresi",     0, 27, 10, help="0 = tidak ada, 27 = sangat berat")

    with st.expander("Kesehatan Fisik", expanded=False):
        headache          = st.slider("Frekuensi Sakit Kepala", 0, 5, 2, help="0=tidak pernah, 5=sangat sering")
        blood_pressure    = st.slider("Tekanan Darah",          1, 3, 2, help="1=rendah, 2=normal, 3=tinggi")
        sleep_quality     = st.slider("Kualitas Tidur",         1, 5, 3, help="1=sangat buruk, 5=sangat baik")
        breathing_problem = st.slider("Masalah Pernapasan",     0, 5, 1, help="0=tidak ada, 5=sangat sering")

    with st.expander("Lingkungan", expanded=False):
        noise_level       = st.slider("Tingkat Kebisingan",        0, 5, 2, help="0=sangat tenang, 5=sangat bising")
        living_conditions = st.slider("Kondisi Tempat Tinggal",    1, 5, 3, help="1=sangat buruk, 5=sangat baik")
        safety            = st.slider("Rasa Aman di Lingkungan",   1, 5, 3, help="1=tidak aman, 5=sangat aman")
        basic_needs       = st.slider("Pemenuhan Kebutuhan Dasar", 1, 5, 3, help="1=tidak terpenuhi, 5=terpenuhi")

    with st.expander("Akademik", expanded=False):
        academic_performance         = st.slider("Performa Akademik",    1, 5, 3, help="1=sangat buruk, 5=sangat baik")
        study_load                   = st.slider("Beban Belajar",        1, 5, 3, help="1=sangat ringan, 5=sangat berat")
        teacher_student_relationship = st.slider("Hub. Dosen-Mahasiswa", 1, 5, 3, help="1=sangat buruk, 5=sangat baik")
        future_career_concerns       = st.slider("Kekhawatiran Karir",   1, 5, 3, help="1=tidak khawatir, 5=sangat khawatir")

    with st.expander("Sosial", expanded=False):
        social_support             = st.slider("Dukungan Sosial",           0, 3, 2, help="0=tidak ada, 3=sangat banyak")
        peer_pressure              = st.slider("Tekanan Teman Sebaya",      1, 5, 3, help="1=tidak ada, 5=sangat tinggi")
        extracurricular_activities = st.slider("Aktivitas Ekstrakurikuler", 0, 5, 2, help="0=tidak ada, 5=sangat aktif")
        bullying                   = st.slider("Tingkat Perundungan",       0, 5, 1, help="0=tidak ada, 5=sangat parah")

    st.markdown("<div style='height:0.75rem'></div>", unsafe_allow_html=True)
    predict_btn = st.button("Analisis Sekarang", use_container_width=True, type="primary")
    st.markdown("""
    <div style='margin-top:0.75rem;font-size:0.73rem;color:#9ca3af;text-align:center;line-height:1.5;'>
        Geser slider sesuai kondisimu saat ini, lalu tekan tombol di atas.
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════
# MAIN CONTENT
# ══════════════════════════════════════════════

# ── Hero ──
st.markdown("""
<div class="hero">
    <div class="hero-text">
        <div class="hero-badge">XGBoost + SHAP · Research Tool</div>
        <h1>Prediksi Tingkat Stres Mahasiswa</h1>
        <p>
            Isi data kondisimu di sidebar kiri, lalu lihat hasil prediksi beserta
            penjelasan faktor-faktor yang paling mempengaruhi tingkat stresmu.
            Semua analisis dijalankan secara lokal, data tidak dikirim ke mana pun.
        </p>
    </div>
</div>
""", unsafe_allow_html=True)

# ── How to use steps ──
st.markdown("""
<div class="steps-row">
    <div class="step-item">
        <div class="step-num">1</div>
        <div>
            <div class="step-title">Isi Data di Sidebar</div>
            <div class="step-desc">Geser slider sesuai kondisi psikologis, fisik, akademik, dan sosialmu saat ini.</div>
        </div>
    </div>
    <div class="step-item">
        <div class="step-num">2</div>
        <div>
            <div class="step-title">Klik Analisis</div>
            <div class="step-desc">Tekan tombol "Analisis Sekarang" untuk menjalankan model prediksi XGBoost.</div>
        </div>
    </div>
    <div class="step-item">
        <div class="step-num">3</div>
        <div>
            <div class="step-title">Baca Hasil & Penjelasan</div>
            <div class="step-desc">Lihat tingkat stres, probabilitas per kelas, dan faktor-faktor terbesar melalui SHAP.</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ── Proses data input ──
raw_input = {
    "anxiety_level": anxiety_level, "self_esteem": self_esteem,
    "mental_health_history": mental_health_history, "depression": depression,
    "headache": headache, "blood_pressure": blood_pressure,
    "sleep_quality": sleep_quality, "breathing_problem": breathing_problem,
    "noise_level": noise_level, "living_conditions": living_conditions,
    "safety": safety, "basic_needs": basic_needs,
    "academic_performance": academic_performance, "study_load": study_load,
    "teacher_student_relationship": teacher_student_relationship,
    "future_career_concerns": future_career_concerns,
    "social_support": social_support, "peer_pressure": peer_pressure,
    "extracurricular_activities": extracurricular_activities, "bullying": bullying,
}
df_input_raw    = build_input_row(raw_input)
df_input_scaled = pd.DataFrame(scaler.transform(df_input_raw), columns=selected_features)
prediction       = model.predict(df_input_scaled)
prediction_proba = model.predict_proba(df_input_scaled)
pred_class       = int(prediction[0])
cfg              = STRESS_CONFIG[pred_class]


# ══════════════════════════════════════════════
# SECTION 1: HASIL PREDIKSI
# ══════════════════════════════════════════════
st.markdown("""
<div class="section-header">
    <span class="section-tag">01</span>
    <span class="section-title">Hasil Prediksi</span>
    <span class="section-subtitle">Berdasarkan 23 faktor input</span>
</div>
""", unsafe_allow_html=True)

col_result, col_prob = st.columns([1, 1.5], gap="large")

with col_result:
    st.markdown(f"""
    <div class="{cfg['css_class']} result-card">
        <div class="result-level {cfg['level_class']}">{cfg['level']}</div>
        <div class="result-label">{cfg['label']}</div>
        <p class="result-desc">{cfg['desc']}</p>
    </div>
    """, unsafe_allow_html=True)

with col_prob:
    proba_labels  = ["Stres Rendah", "Stres Sedang", "Stres Tinggi"]
    bar_classes   = ["prob-bar-low", "prob-bar-medium", "prob-bar-high"]
    confidence    = max(prediction_proba[0])

    st.markdown(f"""
    <div class="prob-container">
        <div class="prob-title">Distribusi Probabilitas Kelas</div>
    """, unsafe_allow_html=True)

    for i, (lbl, bar_cls) in enumerate(zip(proba_labels, bar_classes)):
        pct    = prediction_proba[0][i]
        w      = int(pct * 100)
        bold   = "prob-label-bold" if i == pred_class else "prob-label"
        st.markdown(f"""
        <div class="prob-row">
            <div class="prob-label-row">
                <span class="{bold}">{lbl}</span>
                <span class="prob-pct">{pct:.1%}</span>
            </div>
            <div class="prob-bar-bg">
                <div class="{bar_cls}" style="width:{w}%"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(f"""
        <div class="confidence-badge">Keyakinan model: {confidence:.1%}</div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════
# SECTION 2: SHAP EXPLANATION
# ══════════════════════════════════════════════
st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
st.markdown("""
<div class="section-header">
    <span class="section-tag">02</span>
    <span class="section-title">Penjelasan Prediksi (SHAP)</span>
    <span class="section-subtitle">Faktor yang paling mempengaruhi hasil</span>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="shap-info-card">
    <div class="shap-info-title">Cara membaca penjelasan SHAP</div>
    SHAP (SHapley Additive exPlanations) menghitung seberapa besar kontribusi tiap faktor terhadap prediksi.
    Faktor dengan nilai absolut lebih besar berarti pengaruhnya lebih dominan.
    <div class="shap-legend">
        <div class="shap-legend-item">
            <div class="dot-red"></div>
            <span><strong>Merah</strong> — mendorong prediksi ke stres lebih tinggi</span>
        </div>
        <div class="shap-legend-item">
            <div class="dot-blue"></div>
            <span><strong>Biru</strong> — mendorong prediksi ke stres lebih rendah</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

if hasattr(model, "get_booster"):
    try:
        import shap

        shap_explainer = get_shap_explainer(model)
        shap_vals      = shap_explainer.shap_values(df_input_scaled)

        if isinstance(shap_vals, list):
            sv = shap_vals[pred_class][0]
            ev = shap_explainer.expected_value[pred_class]
        else:
            sv = shap_vals[0, :, pred_class]
            ev = (shap_explainer.expected_value[pred_class]
                  if hasattr(shap_explainer.expected_value, "__len__")
                  else shap_explainer.expected_value)

        # ── Top Driver Cards ──────────────────────────────────────
        shap_series  = pd.Series(sv, index=selected_features)
        top_positive = shap_series[shap_series > 0].sort_values(ascending=False).head(5)
        top_negative = shap_series[shap_series < 0].sort_values(ascending=True).head(5)

        col_up, col_down = st.columns(2, gap="large")

        with col_up:
            st.markdown("""
            <div class="driver-group-title driver-group-title-red">Meningkatkan stres</div>
            """, unsafe_allow_html=True)
            for rank, (feat, val) in enumerate(top_positive.items(), 1):
                feat_label = FEATURE_LABELS.get(feat, feat)
                st.markdown(f"""
                <div class="driver-card driver-card-up">
                    <span class="driver-rank">#{rank}</span>
                    <span class="driver-name">{feat_label}</span>
                    <span class="driver-pill driver-pill-up">Naik</span>
                    <span class="driver-val">+{val:.3f}</span>
                </div>
                """, unsafe_allow_html=True)

        with col_down:
            st.markdown("""
            <div class="driver-group-title driver-group-title-blue">Menurunkan stres</div>
            """, unsafe_allow_html=True)
            if top_negative.empty:
                st.markdown("""
                <div class="info-box">Tidak ada faktor yang secara signifikan menurunkan stres saat ini.</div>
                """, unsafe_allow_html=True)
            else:
                for rank, (feat, val) in enumerate(top_negative.items(), 1):
                    feat_label = FEATURE_LABELS.get(feat, feat)
                    st.markdown(f"""
                    <div class="driver-card driver-card-down">
                        <span class="driver-rank">#{rank}</span>
                        <span class="driver-name">{feat_label}</span>
                        <span class="driver-pill driver-pill-down">Turun</span>
                        <span class="driver-val">{val:.3f}</span>
                    </div>
                    """, unsafe_allow_html=True)

        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

        # ── SHAP Bar Chart ────────────────────────────────────────
        st.markdown("""
        <div class="section-header" style="margin-top:0.25rem;">
            <span class="section-tag">02A</span>
            <span class="section-title">Kontribusi Semua Faktor</span>
            <span class="section-subtitle">Semakin panjang batang, semakin besar pengaruhnya</span>
        </div>
        """, unsafe_allow_html=True)

        sorted_idx   = np.argsort(np.abs(sv))
        sorted_feats = [FEATURE_LABELS.get(selected_features[i], selected_features[i]) for i in sorted_idx]
        sorted_vals  = sv[sorted_idx]
        bar_colors   = ["#ef4444" if v > 0 else "#3b82f6" for v in sorted_vals]

        fig, ax = plt.subplots(figsize=(9, 6.5))
        fig.patch.set_facecolor("#f9fafb")
        ax.set_facecolor("#f9fafb")

        bars = ax.barh(sorted_feats, sorted_vals, color=bar_colors, height=0.6,
                       edgecolor="none", zorder=3, alpha=0.88)

        for bar, val in zip(bars, sorted_vals):
            x_pos = val + (0.002 if val >= 0 else -0.002)
            ha    = "left" if val >= 0 else "right"
            ax.text(x_pos, bar.get_y() + bar.get_height() / 2,
                    f"{val:+.3f}", va="center", ha=ha,
                    fontsize=7, color="#374151", alpha=0.85)

        ax.axvline(0, color="#9ca3af", linewidth=0.8, zorder=2)
        ax.set_xlabel("SHAP Value — kontribusi terhadap prediksi", color="#4b5563", fontsize=8.5, labelpad=8)
        ax.tick_params(axis="y", colors="#374151", labelsize=8)
        ax.tick_params(axis="x", colors="#9ca3af", labelsize=7.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#e5e8ef")
        ax.spines["bottom"].set_color("#e5e8ef")
        ax.grid(axis="x", color="#e5e8ef", linestyle="--", zorder=1, linewidth=0.6)

        red_patch  = mpatches.Patch(color="#ef4444", label="Mendorong stres naik", alpha=0.88)
        blue_patch = mpatches.Patch(color="#3b82f6", label="Mendorong stres turun", alpha=0.88)
        ax.legend(handles=[red_patch, blue_patch], loc="lower right",
                  framealpha=0.8, labelcolor="#374151", fontsize=7.5,
                  facecolor="#f9fafb", edgecolor="#e5e8ef")

        plt.tight_layout()
        st.pyplot(fig, transparent=False)
        plt.close(fig)

        # ── Waterfall ─────────────────────────────────────────────
        st.markdown("""
        <div class="section-header" style="margin-top:1.25rem;">
            <span class="section-tag">02B</span>
            <span class="section-title">Waterfall — Alur Pembentukan Prediksi</span>
            <span class="section-subtitle">Dari nilai dasar model hingga prediksi akhirmu</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="info-box" style="margin-bottom:1rem;">
            <strong>Cara membaca:</strong> Mulai dari nilai dasar model (E[f(x)]), setiap baris
            menunjukkan satu faktor yang "mendorong" prediksi naik (merah) atau turun (biru),
            hingga mencapai prediksi akhir f(x) di bagian atas.
        </div>
        """, unsafe_allow_html=True)

        exp = shap.Explanation(
            values=sv,
            base_values=float(ev),
            data=df_input_scaled.iloc[0].values,
            feature_names=[FEATURE_LABELS.get(f, f) for f in selected_features],
        )
        fig2, _ = plt.subplots(figsize=(10, 6.5))
        fig2.patch.set_facecolor("#f9fafb")
        shap.plots.waterfall(exp, show=False, max_display=15)
        ax2 = plt.gca()
        ax2.set_facecolor("#f9fafb")
        ax2.tick_params(colors="#374151")
        for spine in ax2.spines.values():
            spine.set_edgecolor("#e5e8ef")
        plt.tight_layout()
        st.pyplot(fig2, transparent=False)
        plt.close(fig2)

    except Exception as e:
        st.error(f"SHAP gagal dijalankan: {e}")
        st.code(
            "Pastikan requirements.txt menggunakan:\n"
            "  shap==0.47.0\n"
            "  numpy==1.26.4",
            language="text",
        )


# ══════════════════════════════════════════════
# SECTION 3: DATA INPUT SUMMARY
# ══════════════════════════════════════════════
st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
st.markdown("""
<div class="section-header">
    <span class="section-tag">03</span>
    <span class="section-title">Ringkasan Data Input</span>
    <span class="section-subtitle">Nilai yang dimasukkan + fitur turunan otomatis</span>
</div>
""", unsafe_allow_html=True)

col_a, col_b = st.columns([2.2, 1], gap="large")

with col_a:
    raw_df = pd.DataFrame(
        [(FEATURE_LABELS.get(k, k), v) for k, v in raw_input.items()],
        columns=["Faktor", "Nilai"]
    )
    st.dataframe(raw_df, use_container_width=True, hide_index=True, height=300)

with col_b:
    st.markdown("""
    <div style='font-size:0.72rem;font-weight:700;color:#6b7280;text-transform:uppercase;
                letter-spacing:0.08em;margin-bottom:0.65rem;font-family:DM Mono,monospace;'>
        Fitur Turunan
    </div>
    """, unsafe_allow_html=True)

    engineered = {
        "academic_stress_index":     df_input_raw["academic_stress_index"].values[0],
        "environment_quality_index": df_input_raw["environment_quality_index"].values[0],
        "social_stress_score":       df_input_raw["social_stress_score"].values[0],
    }
    for feat, val in engineered.items():
        label = FEATURE_LABELS.get(feat, feat)
        st.markdown(f"""
        <div class="metric-mini">
            <div class="metric-mini-label">{label}</div>
            <div class="metric-mini-val">{val:.4f}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box" style="margin-top:0.5rem;">
        Fitur-fitur ini dihitung otomatis dari input kamu menggunakan formula berbasis bobot yang
        telah dikalibrasi pada dataset pelatihan.
    </div>
    """, unsafe_allow_html=True)


# ── Footer ──
st.markdown("""
<div style='text-align:center;margin-top:2.5rem;padding-top:1.25rem;
            border-top:1px solid #e5e8ef;'>
    <span style='font-family:DM Mono,monospace;font-size:0.72rem;color:#9ca3af;letter-spacing:0.06em;'>
        Model: XGBoost &nbsp;·&nbsp; Explainability: SHAP TreeExplainer
        &nbsp;·&nbsp; Framework: CRISP-DM &nbsp;·&nbsp; UI: Streamlit
    </span>
</div>
""", unsafe_allow_html=True)