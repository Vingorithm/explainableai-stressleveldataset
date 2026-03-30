# app.py  –  Aplikasi Prediksi Tingkat Stres Mahasiswa
# Versi: Enhanced UI + SHAP yang informatif untuk user awam

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
    page_title="Stress Level Predictor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# CUSTOM CSS
# ──────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Space+Mono:wght@400;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
}

/* Background */
.stApp {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    min-height: 100vh;
}

/* Main container */
.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* ── Hero Banner ── */
.hero-banner {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    border: 1px solid rgba(99, 179, 237, 0.2);
    border-radius: 20px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(99,179,237,0.08) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-title {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 2.4rem;
    font-weight: 800;
    color: #ffffff;
    margin: 0 0 0.5rem 0;
    line-height: 1.2;
}
.hero-subtitle {
    font-size: 1rem;
    color: rgba(255,255,255,0.65);
    margin: 0;
    line-height: 1.6;
    max-width: 600px;
}
.hero-badge {
    display: inline-block;
    background: rgba(99,179,237,0.15);
    border: 1px solid rgba(99,179,237,0.4);
    color: #63b3ed;
    font-size: 0.75rem;
    font-weight: 600;
    padding: 4px 12px;
    border-radius: 20px;
    margin-bottom: 1rem;
    font-family: 'Space Mono', monospace;
    letter-spacing: 0.05em;
}

/* ── Metric Cards ── */
.metric-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 16px;
    padding: 1.5rem;
    text-align: center;
    transition: transform 0.2s, border-color 0.2s;
}
.metric-card:hover {
    transform: translateY(-2px);
    border-color: rgba(99,179,237,0.3);
}
.metric-label {
    font-size: 0.75rem;
    font-weight: 600;
    color: rgba(255,255,255,0.5);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.5rem;
}
.metric-value {
    font-family: 'Space Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    color: #fff;
    line-height: 1;
}
.metric-sub {
    font-size: 0.8rem;
    color: rgba(255,255,255,0.5);
    margin-top: 0.3rem;
}

/* ── Stress Result Card ── */
.result-card-low {
    background: linear-gradient(135deg, #065f46, #064e3b);
    border: 2px solid #10b981;
    border-radius: 20px;
    padding: 2rem 2.5rem;
    text-align: center;
    box-shadow: 0 0 30px rgba(16,185,129,0.2);
}
.result-card-medium {
    background: linear-gradient(135deg, #78350f, #92400e);
    border: 2px solid #f59e0b;
    border-radius: 20px;
    padding: 2rem 2.5rem;
    text-align: center;
    box-shadow: 0 0 30px rgba(245,158,11,0.2);
}
.result-card-high {
    background: linear-gradient(135deg, #7f1d1d, #991b1b);
    border: 2px solid #ef4444;
    border-radius: 20px;
    padding: 2rem 2.5rem;
    text-align: center;
    box-shadow: 0 0 30px rgba(239,68,68,0.2);
}
.result-emoji { font-size: 3.5rem; margin-bottom: 0.5rem; }
.result-label {
    font-size: 1.8rem;
    font-weight: 800;
    color: #fff;
    margin: 0;
}
.result-desc {
    font-size: 0.9rem;
    color: rgba(255,255,255,0.75);
    margin-top: 0.5rem;
}

/* ── Section Headers ── */
.section-header {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin-bottom: 1.25rem;
    padding-bottom: 0.75rem;
    border-bottom: 1px solid rgba(255,255,255,0.08);
}
.section-icon {
    font-size: 1.4rem;
}
.section-title {
    font-size: 1.2rem;
    font-weight: 700;
    color: #fff;
    margin: 0;
}
.section-subtitle {
    font-size: 0.8rem;
    color: rgba(255,255,255,0.5);
    margin: 0;
}

/* ── SHAP Explanation Cards ── */
.shap-intro-card {
    background: rgba(99,179,237,0.06);
    border: 1px solid rgba(99,179,237,0.2);
    border-radius: 14px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1.5rem;
}
.shap-legend-row {
    display: flex;
    gap: 1.5rem;
    margin-top: 0.75rem;
    flex-wrap: wrap;
}
.shap-legend-item {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 0.85rem;
    color: rgba(255,255,255,0.8);
}
.legend-dot-red {
    width: 14px; height: 14px;
    background: #f87171;
    border-radius: 3px;
    flex-shrink: 0;
}
.legend-dot-blue {
    width: 14px; height: 14px;
    background: #60a5fa;
    border-radius: 3px;
    flex-shrink: 0;
}

/* ── Feature Insight Cards (Top drivers) ── */
.driver-card {
    background: rgba(255,255,255,0.04);
    border-radius: 10px;
    padding: 0.9rem 1.1rem;
    margin-bottom: 0.6rem;
    display: flex;
    align-items: center;
    gap: 0.85rem;
    border-left: 4px solid transparent;
}
.driver-card-up { border-left-color: #f87171; }
.driver-card-down { border-left-color: #60a5fa; }
.driver-rank {
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    color: rgba(255,255,255,0.35);
    min-width: 20px;
}
.driver-name {
    font-size: 0.9rem;
    font-weight: 600;
    color: #fff;
    flex: 1;
}
.driver-direction {
    font-size: 0.75rem;
    padding: 2px 10px;
    border-radius: 20px;
    font-weight: 600;
}
.driver-direction-up {
    background: rgba(248,113,113,0.15);
    color: #f87171;
}
.driver-direction-down {
    background: rgba(96,165,250,0.15);
    color: #60a5fa;
}
.driver-value {
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem;
    color: rgba(255,255,255,0.5);
    min-width: 55px;
    text-align: right;
}

/* ── Info Box ── */
.info-box {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 1rem 1.25rem;
    font-size: 0.85rem;
    color: rgba(255,255,255,0.65);
    line-height: 1.6;
}
.info-box strong { color: rgba(255,255,255,0.9); }

/* ── Probability Bar ── */
.prob-row {
    margin-bottom: 0.85rem;
}
.prob-label-row {
    display: flex;
    justify-content: space-between;
    margin-bottom: 0.3rem;
}
.prob-label { font-size: 0.85rem; color: rgba(255,255,255,0.8); font-weight: 500; }
.prob-pct   { font-family: 'Space Mono', monospace; font-size: 0.85rem; color: #fff; font-weight: 700; }
.prob-bar-bg {
    background: rgba(255,255,255,0.08);
    border-radius: 6px;
    height: 10px;
    overflow: hidden;
}
.prob-bar-fill-low    { height: 100%; background: linear-gradient(90deg,#10b981,#34d399); border-radius: 6px; transition: width 0.8s ease; }
.prob-bar-fill-medium { height: 100%; background: linear-gradient(90deg,#f59e0b,#fbbf24); border-radius: 6px; transition: width 0.8s ease; }
.prob-bar-fill-high   { height: 100%; background: linear-gradient(90deg,#ef4444,#f87171); border-radius: 6px; transition: width 0.8s ease; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    border-right: 1px solid rgba(255,255,255,0.07);
}
section[data-testid="stSidebar"] .stSlider > div > div > div { background: #63b3ed !important; }

/* ── Charts background transparent ── */
.stPlotlyChart, .stPyplot { background: transparent !important; }

/* ── Streamlit overrides ── */
.stDataFrame { border-radius: 10px; overflow: hidden; }
div[data-testid="stMetric"] { background: rgba(255,255,255,0.04); border-radius: 12px; padding: 1rem; }
.stAlert { border-radius: 12px; }
hr { border-color: rgba(255,255,255,0.08) !important; }
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
        "emoji": "😌",
        "css_class": "result-card-low",
        "color": "#10b981",
        "bar_class": "prob-bar-fill-low",
        "desc": "Kondisi mental Anda tergolong baik. Tetap pertahankan pola hidup sehat!",
    },
    1: {
        "label": "Stres Sedang",
        "emoji": "😐",
        "css_class": "result-card-medium",
        "color": "#f59e0b",
        "bar_class": "prob-bar-fill-medium",
        "desc": "Ada beberapa faktor pemicu stres. Perhatikan istirahat dan manajemen waktu.",
    },
    2: {
        "label": "Stres Tinggi",
        "emoji": "😰",
        "css_class": "result-card-high",
        "color": "#ef4444",
        "bar_class": "prob-bar-fill-high",
        "desc": "Tingkat stres cukup tinggi. Disarankan untuk berbicara dengan konselor atau profesional.",
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
# [FIX] SHAP Monkey-Patch (XGBoost 3.x multiclass)
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
    "bullying": (0, 5), "social_support": (0, 5),
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
    <div style='padding:1rem 0 1.5rem 0;'>
        <div style='font-family:Space Mono,monospace;font-size:0.7rem;color:rgba(255,255,255,0.4);
                    letter-spacing:0.15em;margin-bottom:0.5rem;'>INPUT MAHASISWA</div>
        <div style='font-size:1.1rem;font-weight:800;color:#fff;'>Isi Data Kamu</div>
        <div style='font-size:0.8rem;color:rgba(255,255,255,0.5);margin-top:0.2rem;'>
            Geser slider sesuai kondisi kamu saat ini
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("🧠 Faktor Psikologis", expanded=True):
        anxiety_level         = st.slider("Tingkat Kecemasan",        0, 21, 10,
                                          help="0 = tidak cemas sama sekali, 21 = sangat cemas")
        self_esteem           = st.slider("Harga Diri",               0, 30, 15,
                                          help="0 = sangat rendah, 30 = sangat tinggi")
        mental_health_history = st.selectbox("Riwayat Masalah Mental", [0, 1],
                                              format_func=lambda x: "Tidak ada" if x == 0 else "Ada riwayat")
        depression            = st.slider("Tingkat Depresi",           0, 27, 10,
                                          help="0 = tidak ada, 27 = sangat berat")

    with st.expander("🏥 Kesehatan Fisik", expanded=False):
        headache          = st.slider("Frekuensi Sakit Kepala", 0, 5, 2, help="0=tidak pernah, 5=sangat sering")
        blood_pressure    = st.slider("Tekanan Darah",          1, 3, 2, help="1=rendah, 2=normal, 3=tinggi")
        sleep_quality     = st.slider("Kualitas Tidur",         1, 5, 3, help="1=sangat buruk, 5=sangat baik")
        breathing_problem = st.slider("Masalah Pernapasan",     0, 5, 1, help="0=tidak ada, 5=sangat sering")

    with st.expander("🏠 Lingkungan", expanded=False):
        noise_level       = st.slider("Tingkat Kebisingan",        0, 5, 2, help="0=sangat tenang, 5=sangat bising")
        living_conditions = st.slider("Kondisi Tempat Tinggal",    1, 5, 3, help="1=sangat buruk, 5=sangat baik")
        safety            = st.slider("Rasa Aman di Lingkungan",   1, 5, 3, help="1=tidak aman, 5=sangat aman")
        basic_needs       = st.slider("Pemenuhan Kebutuhan Dasar", 1, 5, 3, help="1=tidak terpenuhi, 5=sangat terpenuhi")

    with st.expander("📚 Akademik", expanded=False):
        academic_performance         = st.slider("Performa Akademik",    1, 5, 3, help="1=sangat buruk, 5=sangat baik")
        study_load                   = st.slider("Beban Belajar",        1, 5, 3, help="1=sangat ringan, 5=sangat berat")
        teacher_student_relationship = st.slider("Hub. Dosen-Mahasiswa", 1, 5, 3, help="1=sangat buruk, 5=sangat baik")
        future_career_concerns       = st.slider("Kekhawatiran Karir",   1, 5, 3, help="1=tidak khawatir, 5=sangat khawatir")

    with st.expander("👥 Sosial", expanded=False):
        social_support             = st.slider("Dukungan Sosial",           1, 5, 3, help="1=tidak ada, 5=sangat banyak")
        peer_pressure              = st.slider("Tekanan Teman Sebaya",      1, 5, 3, help="1=tidak ada, 5=sangat tinggi")
        extracurricular_activities = st.slider("Aktivitas Ekstrakurikuler", 0, 5, 2, help="0=tidak ada, 5=sangat aktif")
        bullying                   = st.slider("Tingkat Perundungan",       0, 5, 1, help="0=tidak ada, 5=sangat parah")

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    predict_btn = st.button("🔍 Analisis Sekarang", use_container_width=True, type="primary")


# ══════════════════════════════════════════════
# MAIN CONTENT
# ══════════════════════════════════════════════

# ── Hero Banner ──
st.markdown("""
<div class="hero-banner">
    <div class="hero-badge">🎓 RESEARCH TOOL · XGBoost + SHAP</div>
    <h1 class="hero-title">Prediksi Tingkat Stres<br>Mahasiswa</h1>
    <p class="hero-subtitle">
        Isi data kondisi kamu di sidebar kiri, lalu lihat hasil prediksi berbasis AI
        beserta penjelasan faktor-faktor yang paling mempengaruhi tingkat stresmu.
    </p>
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
    <span class="section-icon">🎯</span>
    <div>
        <div class="section-title">Hasil Prediksi</div>
        <div class="section-subtitle">Berdasarkan 23 faktor yang kamu masukkan</div>
    </div>
</div>
""", unsafe_allow_html=True)

col_result, col_prob = st.columns([1, 1.4], gap="large")

with col_result:
    st.markdown(f"""
    <div class="{cfg['css_class']}">
        <div class="result-emoji">{cfg['emoji']}</div>
        <div class="result-label">{cfg['label']}</div>
        <div class="result-desc">{cfg['desc']}</div>
    </div>
    """, unsafe_allow_html=True)

with col_prob:
    st.markdown("""
    <div style='margin-bottom:0.75rem;'>
        <span style='font-size:0.8rem;font-weight:600;color:rgba(255,255,255,0.5);
                     text-transform:uppercase;letter-spacing:0.1em;'>
            Probabilitas per Kelas
        </span>
    </div>
    """, unsafe_allow_html=True)

    proba_labels = ["😌 Stres Rendah", "😐 Stres Sedang", "😰 Stres Tinggi"]
    bar_classes  = ["prob-bar-fill-low", "prob-bar-fill-medium", "prob-bar-fill-high"]

    for i, (lbl, bar_cls) in enumerate(zip(proba_labels, bar_classes)):
        pct  = prediction_proba[0][i]
        w    = int(pct * 100)
        bold = "font-weight:800;color:#fff;" if i == pred_class else ""
        st.markdown(f"""
        <div class="prob-row">
            <div class="prob-label-row">
                <span class="prob-label" style="{bold}">{lbl}</span>
                <span class="prob-pct">{pct:.1%}</span>
            </div>
            <div class="prob-bar-bg">
                <div class="{bar_cls}" style="width:{w}%"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    confidence = max(prediction_proba[0])
    st.markdown(f"""
    <div class="info-box" style="margin-top:1rem;">
        <strong>Tingkat Keyakinan Model:</strong> {confidence:.1%}<br>
        <span style="font-size:0.8rem;">
            Model XGBoost menganalisis 23 fitur untuk menghasilkan prediksi ini.
        </span>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════
# SECTION 2: SHAP EXPLANATION
# ══════════════════════════════════════════════
st.markdown("---")
st.markdown("""
<div class="section-header">
    <span class="section-icon">🔬</span>
    <div>
        <div class="section-title">Mengapa Prediksi Ini?</div>
        <div class="section-subtitle">Penjelasan berbasis SHAP — faktor apa yang paling mempengaruhi hasil</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Penjelasan SHAP untuk user awam
st.markdown("""
<div class="shap-intro-card">
    <strong style="color:#63b3ed;font-size:0.95rem;">💡 Apa itu SHAP?</strong>
    <p style="margin:0.5rem 0 0 0;font-size:0.88rem;color:rgba(255,255,255,0.75);line-height:1.7;">
        SHAP adalah teknologi <em>Explainable AI</em> yang menjelaskan <strong style="color:#fff">mengapa</strong>
        model menghasilkan prediksi tertentu — bukan hanya <em>apa</em> hasilnya.
        Setiap faktor diberi skor: semakin besar nilainya, semakin besar pengaruhnya terhadap prediksimu.
    </p>
    <div class="shap-legend-row">
        <div class="shap-legend-item">
            <div class="legend-dot-red"></div>
            <span><strong style="color:#f87171">Mendorong NAIK</strong> — faktor ini meningkatkan risiko stresmu</span>
        </div>
        <div class="shap-legend-item">
            <div class="legend-dot-blue"></div>
            <span><strong style="color:#60a5fa">Mendorong TURUN</strong> — faktor ini mengurangi risiko stresmu</span>
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

        # ── Top Drivers Cards (user-friendly) ──────────────────────
        shap_series  = pd.Series(sv, index=selected_features)
        top_positive = shap_series[shap_series > 0].sort_values(ascending=False).head(5)
        top_negative = shap_series[shap_series < 0].sort_values(ascending=True).head(5)

        col_up, col_down = st.columns(2, gap="large")

        with col_up:
            st.markdown("""
            <div style='margin-bottom:0.75rem;'>
                <span style='color:#f87171;font-weight:700;font-size:0.95rem;'>
                    🔺 Faktor yang Meningkatkan Stres
                </span><br>
                <span style='font-size:0.78rem;color:rgba(255,255,255,0.45);'>
                    Kondisi ini mendorong prediksi ke arah stres lebih tinggi
                </span>
            </div>
            """, unsafe_allow_html=True)
            for rank, (feat, val) in enumerate(top_positive.items(), 1):
                feat_label = FEATURE_LABELS.get(feat, feat)
                st.markdown(f"""
                <div class="driver-card driver-card-up">
                    <span class="driver-rank">#{rank}</span>
                    <span class="driver-name">{feat_label}</span>
                    <span class="driver-direction driver-direction-up">▲ Naik</span>
                    <span class="driver-value">+{val:.3f}</span>
                </div>
                """, unsafe_allow_html=True)

        with col_down:
            st.markdown("""
            <div style='margin-bottom:0.75rem;'>
                <span style='color:#60a5fa;font-weight:700;font-size:0.95rem;'>
                    🔻 Faktor yang Menurunkan Stres
                </span><br>
                <span style='font-size:0.78rem;color:rgba(255,255,255,0.45);'>
                    Kondisi ini mendorong prediksi ke arah stres lebih rendah
                </span>
            </div>
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
                        <span class="driver-direction driver-direction-down">▼ Turun</span>
                        <span class="driver-value">{val:.3f}</span>
                    </div>
                    """, unsafe_allow_html=True)

        st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)

        # ── SHAP Bar Chart (matplotlib) ─────────────────────────────
        st.markdown("""
        <div class="section-header" style="margin-top:0.5rem;">
            <span class="section-icon">📊</span>
            <div>
                <div class="section-title">Kontribusi Semua Faktor</div>
                <div class="section-subtitle">Semakin panjang batang, semakin besar pengaruhnya</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        sorted_idx   = np.argsort(np.abs(sv))
        sorted_feats = [FEATURE_LABELS.get(selected_features[i], selected_features[i]) for i in sorted_idx]
        sorted_vals  = sv[sorted_idx]
        bar_colors   = ["#f87171" if v > 0 else "#60a5fa" for v in sorted_vals]

        fig, ax = plt.subplots(figsize=(9, 7))
        fig.patch.set_alpha(0)
        ax.set_facecolor("none")

        bars = ax.barh(sorted_feats, sorted_vals, color=bar_colors, height=0.65,
                       edgecolor="none", zorder=3)

        # Value labels
        for bar, val in zip(bars, sorted_vals):
            x_pos = val + (0.002 if val >= 0 else -0.002)
            ha    = "left" if val >= 0 else "right"
            ax.text(x_pos, bar.get_y() + bar.get_height() / 2,
                    f"{val:+.3f}", va="center", ha=ha,
                    fontsize=7.5, color="white", alpha=0.75)

        ax.axvline(0, color=(1, 1, 1, 0.25), linewidth=1, zorder=2)
        ax.set_xlabel("SHAP Value (pengaruh terhadap prediksi)", color="white", fontsize=9, labelpad=8)
        ax.tick_params(axis="y", colors="white", labelsize=8.5)
        ax.tick_params(axis="x", colors=(1, 1, 1, 0.5))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color((1, 1, 1, 0.1))
        ax.spines["bottom"].set_color((1, 1, 1, 0.1))
        ax.grid(axis="x", color="white", alpha=0.05, linestyle="--", zorder=1)

        red_patch  = mpatches.Patch(color="#f87171", label="Mendorong stres naik")
        blue_patch = mpatches.Patch(color="#60a5fa", label="Mendorong stres turun")
        ax.legend(handles=[red_patch, blue_patch], loc="lower right",
                  framealpha=0.15, labelcolor="white", fontsize=8,
                  facecolor="black", edgecolor="rgba(255,255,255,0.1)")

        plt.tight_layout()
        st.pyplot(fig, transparent=True)
        plt.close(fig)

        # ── Waterfall Plot ──────────────────────────────────────────
        st.markdown("""
        <div class="section-header" style="margin-top:1.5rem;">
            <span class="section-icon">🌊</span>
            <div>
                <div class="section-title">Waterfall — Alur Pembentukan Prediksi</div>
                <div class="section-subtitle">
                    Bagaimana nilai awal (E[f(x)]) bergerak step-by-step ke prediksi akhir (f(x))
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="info-box" style="margin-bottom:1rem;">
            <strong>Cara membaca grafik ini:</strong> Mulai dari nilai dasar model (E[f(x)]),
            setiap baris adalah satu faktor yang "mendorong" prediksi naik (merah) atau turun (biru),
            hingga mencapai prediksi akhirmu f(x) di bagian atas.
        </div>
        """, unsafe_allow_html=True)

        exp = shap.Explanation(
            values=sv,
            base_values=float(ev),
            data=df_input_scaled.iloc[0].values,
            feature_names=[FEATURE_LABELS.get(f, f) for f in selected_features],
        )
        fig2, _ = plt.subplots(figsize=(10, 7))
        fig2.patch.set_alpha(0)
        shap.plots.waterfall(exp, show=False, max_display=15)
        ax2 = plt.gca()
        ax2.set_facecolor("none")
        ax2.tick_params(colors="white")
        for spine in ax2.spines.values():
            spine.set_edgecolor("rgba(255,255,255,0.15)")
        plt.tight_layout()
        st.pyplot(fig2, transparent=True)
        plt.close(fig2)

    except Exception as e:
        st.error(f"⚠️ SHAP gagal dijalankan: {e}")
        st.code(
            "Pastikan requirements.txt menggunakan:\n"
            "  shap==0.47.0\n"
            "  numpy==1.26.4",
            language="text",
        )


# ══════════════════════════════════════════════
# SECTION 3: DATA INPUT SUMMARY
# ══════════════════════════════════════════════
st.markdown("---")
st.markdown("""
<div class="section-header">
    <span class="section-icon">📋</span>
    <div>
        <div class="section-title">Ringkasan Data Input</div>
        <div class="section-subtitle">Semua nilai yang dimasukkan + fitur turunan yang dihitung otomatis</div>
    </div>
</div>
""", unsafe_allow_html=True)

col_a, col_b = st.columns([2, 1], gap="large")

with col_a:
    raw_df = pd.DataFrame(
        [(FEATURE_LABELS.get(k, k), v) for k, v in raw_input.items()],
        columns=["Faktor", "Nilai Kamu"]
    )
    st.dataframe(raw_df, use_container_width=True, hide_index=True, height=320)

with col_b:
    st.markdown("<div style='font-size:0.85rem;font-weight:600;color:rgba(255,255,255,0.6);margin-bottom:0.75rem;'>Fitur Turunan (Dihitung Otomatis)</div>", unsafe_allow_html=True)

    engineered = {
        "academic_stress_index": df_input_raw["academic_stress_index"].values[0],
        "environment_quality_index": df_input_raw["environment_quality_index"].values[0],
        "social_stress_score": df_input_raw["social_stress_score"].values[0],
    }
    for feat, val in engineered.items():
        label = FEATURE_LABELS.get(feat, feat)
        st.markdown(f"""
        <div class="metric-card" style="margin-bottom:0.75rem;text-align:left;padding:1rem 1.25rem;">
            <div class="metric-label">{label}</div>
            <div class="metric-value" style="font-size:1.4rem;">{val:.4f}</div>
        </div>
        """, unsafe_allow_html=True)


# ── Footer ──
st.markdown("""
<div style='text-align:center;margin-top:3rem;padding-top:1.5rem;
            border-top:1px solid rgba(255,255,255,0.08);'>
    <span style='font-family:Space Mono,monospace;font-size:0.75rem;
                 color:rgba(255,255,255,0.25);letter-spacing:0.08em;'>
        MODEL: XGBoost &nbsp;·&nbsp; EXPLAINABILITY: SHAP TreeExplainer
        &nbsp;·&nbsp; FRAMEWORK: CRISP-DM &nbsp;·&nbsp; UI: Streamlit
    </span>
</div>
""", unsafe_allow_html=True)