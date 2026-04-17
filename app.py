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
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# CUSTOM CSS – Light Theme
# ──────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Reset & Base ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    color: #1f2937;
}

.stApp {
    background: #f8fafc;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1080px;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 1px solid #e2e8f0;
}

section[data-testid="stSidebar"] label {
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    color: #475569 !important;
}

/* Sidebar header */
.sidebar-header {
    padding: 0.5rem 0 1rem 0;
    border-bottom: 1px solid #f1f5f9;
    margin-bottom: 1rem;
}
.sidebar-app-name {
    font-size: 1.1rem;
    font-weight: 700;
    color: #0f172a;
    letter-spacing: -0.01em;
}
.sidebar-tagline {
    font-size: 0.75rem;
    color: #64748b;
    margin-top: 0.2rem;
}

/* ── Typography & Tags ── */
.section-title {
    font-size: 1.15rem;
    font-weight: 600;
    color: #0f172a;
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #e2e8f0;
}

/* ── Result Card ── */
.result-card {
    border-radius: 8px;
    padding: 1.5rem;
    text-align: center;
    border: 1px solid;
    box-shadow: 0 1px 3px rgba(0,0,0,0.02);
}
.result-card-low    { background: #f0fdf4; border-color: #bbf7d0; }
.result-card-medium { background: #fffbeb; border-color: #fef08a; }
.result-card-high   { background: #fef2f2; border-color: #fecaca; }

.result-level {
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
    font-family: 'JetBrains Mono', monospace;
}
.result-level-low    { color: #15803d; }
.result-level-medium { color: #b45309; }
.result-level-high   { color: #b91c1c; }

.result-label {
    font-size: 1.5rem;
    font-weight: 700;
    color: #0f172a;
    margin: 0 0 0.5rem 0;
}
.result-desc {
    font-size: 0.85rem;
    color: #475569;
    line-height: 1.5;
    margin: 0;
}

/* ── Probability Bars ── */
.prob-container {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 1.25rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.02);
}
.prob-title {
    font-size: 0.8rem;
    font-weight: 600;
    color: #475569;
    margin-bottom: 1rem;
}
.prob-row {
    margin-bottom: 0.8rem;
}
.prob-label-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.3rem;
}
.prob-label      { font-size: 0.85rem; color: #475569; font-weight: 400; }
.prob-label-bold { font-weight: 600; color: #0f172a; font-size: 0.85rem; }
.prob-pct        { font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; font-weight: 500; color: #0f172a; }
.prob-bar-bg     { background: #f1f5f9; border-radius: 4px; height: 6px; overflow: hidden; }
.prob-bar-low    { height: 100%; background: #22c55e; border-radius: 4px; }
.prob-bar-medium { height: 100%; background: #f59e0b; border-radius: 4px; }
.prob-bar-high   { height: 100%; background: #ef4444; border-radius: 4px; }

/* ── SHAP Info Box ── */
.shap-info-card {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 6px;
    padding: 0.85rem 1rem;
    margin-bottom: 1rem;
    font-size: 0.85rem;
    color: #475569;
}
.shap-legend {
    display: flex;
    gap: 1rem;
    margin-top: 0.5rem;
}
.shap-legend-item {
    display: flex;
    align-items: center;
    gap: 0.4rem;
    font-size: 0.8rem;
}
.dot-red  { width: 8px; height: 8px; background: #ef4444; border-radius: 50%; }
.dot-blue { width: 8px; height: 8px; background: #3b82f6; border-radius: 50%; }

/* ── Driver Cards ── */
.driver-group-title {
    font-size: 0.8rem;
    font-weight: 600;
    margin-bottom: 0.5rem;
}
.driver-group-title-red  { color: #b91c1c; }
.driver-group-title-blue { color: #1d4ed8; }

.driver-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 6px;
    padding: 0.6rem 0.8rem;
    margin-bottom: 0.4rem;
    display: flex;
    align-items: center;
    gap: 0.75rem;
}
.driver-rank {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    color: #94a3b8;
    min-width: 20px;
}
.driver-name {
    font-size: 0.85rem;
    font-weight: 500;
    color: #1e293b;
    flex: 1;
}
.driver-val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    font-weight: 500;
}
.val-up { color: #ef4444; }
.val-down { color: #3b82f6; }

/* ── Streamlit Tabs Styling Overrides ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 2rem;
    border-bottom: 1px solid #e2e8f0;
}
.stTabs [data-baseweb="tab"] {
    padding-top: 1rem;
    padding-bottom: 1rem;
    height: auto;
}
.stTabs [aria-selected="true"] {
    font-weight: 600;
}

</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# LOAD ARTIFACTS
# ──────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    # Placeholder error handling jika file pkl tidak ada untuk demo
    try:
        model             = joblib.load("xgb_model.pkl")
        scaler            = joblib.load("scaler.pkl")
        label_encoder     = joblib.load("label_encoder.pkl")
        selected_features = joblib.load("selected_features.pkl")
    except:
        st.warning("Model artifacts belum ditemukan. Pastikan file .pkl tersedia di direktori yang sama.")
        st.stop()
    return model, scaler, label_encoder, selected_features

model, scaler, le, selected_features = load_artifacts()

STRESS_CONFIG = {
    0: {
        "label": "Stres Rendah",
        "level": "Level Rendah",
        "css_class": "result-card-low",
        "level_class": "result-level-low",
        "bar_class": "prob-bar-low",
        "desc": "Kondisi mental terpantau stabil. Pertahankan pola istirahat, aktivitas, dan dukungan sosial yang positif.",
    },
    1: {
        "label": "Stres Sedang",
        "level": "Level Sedang",
        "css_class": "result-card-medium",
        "level_class": "result-level-medium",
        "bar_class": "prob-bar-medium",
        "desc": "Terdapat beberapa indikasi pemicu stres. Perhatikan kualitas tidur dan manajemen beban akademik.",
    },
    2: {
        "label": "Stres Tinggi",
        "level": "Level Tinggi",
        "css_class": "result-card-high",
        "level_class": "result-level-high",
        "bar_class": "prob-bar-high",
        "desc": "Tingkat stres terdeteksi cukup tinggi. Disarankan untuk berkonsultasi dengan layanan konseling akademik atau kesehatan jiwa.",
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
        <div class="sidebar-app-name">StressCheck AI</div>
        <div class="sidebar-tagline">Analisis Kondisi Mahasiswa</div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("Psikologis", expanded=True):
        anxiety_level         = st.slider("Tingkat Kecemasan",  0, 21, 10, help="0 = Tidak cemas, 21 = Sangat cemas")
        self_esteem           = st.slider("Harga Diri",         0, 30, 15, help="0 = Sangat rendah, 30 = Sangat tinggi")
        mental_health_history = st.selectbox("Riwayat Masalah Mental", [0, 1], format_func=lambda x: "Tidak ada" if x == 0 else "Ada riwayat")
        depression            = st.slider("Tingkat Depresi",     0, 27, 10, help="0 = Tidak ada, 27 = Sangat berat")

    with st.expander("Kesehatan Fisik", expanded=False):
        headache          = st.slider("Frekuensi Sakit Kepala", 0, 5, 2)
        blood_pressure    = st.slider("Tekanan Darah",          1, 3, 2, help="1 = Rendah, 2 = Normal, 3 = Tinggi")
        sleep_quality     = st.slider("Kualitas Tidur",         1, 5, 3)
        breathing_problem = st.slider("Masalah Pernapasan",     0, 5, 1)

    with st.expander("Lingkungan", expanded=False):
        noise_level       = st.slider("Tingkat Kebisingan",      0, 5, 2)
        living_conditions = st.slider("Kondisi Tempat Tinggal",  1, 5, 3)
        safety            = st.slider("Rasa Aman di Lingkungan", 1, 5, 3)
        basic_needs       = st.slider("Kebutuhan Dasar",         1, 5, 3)

    with st.expander("Akademik", expanded=False):
        academic_performance         = st.slider("Performa Akademik",    1, 5, 3)
        study_load                   = st.slider("Beban Belajar",        1, 5, 3)
        teacher_student_relationship = st.slider("Hub. Dosen-Mahasiswa", 1, 5, 3)
        future_career_concerns       = st.slider("Kekhawatiran Karir",   1, 5, 3)

    with st.expander("Sosial", expanded=False):
        social_support             = st.slider("Dukungan Sosial",           0, 3, 2)
        peer_pressure              = st.slider("Tekanan Teman Sebaya",      1, 5, 3)
        extracurricular_activities = st.slider("Aktivitas Ekstrakurikuler", 0, 5, 2)
        bullying                   = st.slider("Tingkat Perundungan",       0, 5, 1)

    st.markdown("<br>", unsafe_allow_html=True)
    st.button("Jalankan Prediksi", use_container_width=True, type="primary")


# ══════════════════════════════════════════════
# DATA PROCESSING (Pre-computations)
# ══════════════════════════════════════════════
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
prediction      = model.predict(df_input_scaled)
prediction_proba = model.predict_proba(df_input_scaled)
pred_class      = int(prediction[0])
cfg             = STRESS_CONFIG[pred_class]


# ══════════════════════════════════════════════
# MAIN CONTENT (Tabs Layout)
# ══════════════════════════════════════════════
st.markdown("### Prediksi Tingkat Stres Mahasiswa")
st.markdown("<p style='color:#64748b; font-size:0.9rem; margin-top:-0.5rem;'>Sesuaikan parameter di sidebar untuk melihat pembaruan prediksi secara real-time.</p>", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["Hasil Analisis", "Detail Model (SHAP)", "Ringkasan Data"])

# ── TAB 1: HASIL PREDIKSI ──
with tab1:
    st.markdown("<br>", unsafe_allow_html=True)
    col_result, col_prob = st.columns([1, 1.2], gap="large")

    with col_result:
        st.markdown(f"""
        <div class="section-title">Kesimpulan Model</div>
        <div class="{cfg['css_class']} result-card">
            <div class="result-level {cfg['level_class']}">{cfg['level']}</div>
            <div class="result-label">{cfg['label']}</div>
            <p class="result-desc">{cfg['desc']}</p>
        </div>
        """, unsafe_allow_html=True)

    with col_prob:
        proba_labels  = ["Stres Rendah", "Stres Sedang", "Stres Tinggi"]
        bar_classes   = ["prob-bar-low", "prob-bar-medium", "prob-bar-high"]

        st.markdown(f"""
        <div class="section-title">Probabilitas Kelas</div>
        <div class="prob-container">
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

        st.markdown("</div>", unsafe_allow_html=True)


# ── TAB 2: SHAP EXPLANATION ──
with tab2:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="shap-info-card">
        <strong>Interpretasi Model (SHAP)</strong><br>
        Menampilkan fitur mana yang paling kuat mendorong hasil prediksi. 
        <div class="shap-legend">
            <div class="shap-legend-item"><div class="dot-red"></div> Mendorong stres lebih tinggi</div>
            <div class="shap-legend-item"><div class="dot-blue"></div> Mendorong stres lebih rendah</div>
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

            shap_series  = pd.Series(sv, index=selected_features)
            top_positive = shap_series[shap_series > 0].sort_values(ascending=False).head(5)
            top_negative = shap_series[shap_series < 0].sort_values(ascending=True).head(5)

            col_up, col_down = st.columns(2, gap="medium")

            with col_up:
                st.markdown("<div class='driver-group-title driver-group-title-red'>Top Faktor Pendorong (+)</div>", unsafe_allow_html=True)
                if top_positive.empty:
                    st.info("Tidak ada pendorong signifikan.")
                else:
                    for rank, (feat, val) in enumerate(top_positive.items(), 1):
                        st.markdown(f"""
                        <div class="driver-card">
                            <span class="driver-rank">#{rank}</span>
                            <span class="driver-name">{FEATURE_LABELS.get(feat, feat)}</span>
                            <span class="driver-val val-up">+{val:.3f}</span>
                        </div>
                        """, unsafe_allow_html=True)

            with col_down:
                st.markdown("<div class='driver-group-title driver-group-title-blue'>Top Faktor Penekan (-)</div>", unsafe_allow_html=True)
                if top_negative.empty:
                    st.info("Tidak ada penekan signifikan.")
                else:
                    for rank, (feat, val) in enumerate(top_negative.items(), 1):
                        st.markdown(f"""
                        <div class="driver-card">
                            <span class="driver-rank">#{rank}</span>
                            <span class="driver-name">{FEATURE_LABELS.get(feat, feat)}</span>
                            <span class="driver-val val-down">{val:.3f}</span>
                        </div>
                        """, unsafe_allow_html=True)

            # Waterfall Chart
            st.markdown("<div class='section-title' style='margin-top: 2rem;'>Alur Prediksi (Waterfall Chart)</div>", unsafe_allow_html=True)
            
            exp = shap.Explanation(
                values=sv,
                base_values=float(ev),
                data=df_input_scaled.iloc[0].values,
                feature_names=[FEATURE_LABELS.get(f, f) for f in selected_features],
            )
            fig, ax = plt.subplots(figsize=(8, 5))
            fig.patch.set_facecolor("#ffffff")
            shap.plots.waterfall(exp, show=False, max_display=12)
            
            # Matplotlib styling for light mode
            plt.gca().set_facecolor("#ffffff")
            plt.gca().tick_params(colors="#1f2937")
            for spine in plt.gca().spines.values():
                spine.set_edgecolor("#e2e8f0")
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        except Exception as e:
            st.error(f"Grafik SHAP tidak dapat ditampilkan: {e}")


# ── TAB 3: DATA INPUT SUMMARY ──
with tab3:
    st.markdown("<br>", unsafe_allow_html=True)
    col_a, col_b = st.columns([2, 1], gap="large")

    with col_a:
        st.markdown("<div class='section-title'>Parameter Input</div>", unsafe_allow_html=True)
        raw_df = pd.DataFrame(
            [(FEATURE_LABELS.get(k, k), v) for k, v in raw_input.items()],
            columns=["Faktor / Parameter", "Nilai Input"]
        )
        st.dataframe(raw_df, use_container_width=True, hide_index=True, height=400)

    with col_b:
        st.markdown("<div class='section-title'>Fitur Kalkulasi</div>", unsafe_allow_html=True)
        engineered = {
            "academic_stress_index":     df_input_raw["academic_stress_index"].values[0],
            "environment_quality_index": df_input_raw["environment_quality_index"].values[0],
            "social_stress_score":       df_input_raw["social_stress_score"].values[0],
        }
        for feat, val in engineered.items():
            st.metric(label=FEATURE_LABELS.get(feat, feat), value=f"{val:.3f}")
        
        st.caption("Fitur-fitur ini dihasilkan secara otomatis dari data sidebar berdasarkan pembobotan algoritma untuk diteruskan ke dalam model XGBoost.")