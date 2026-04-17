# App.py  –  Website Prediksi Tingkat Stres Mahasiswa
# Light UI – Clean, Compact, Feature-Rich (Thesis Edition)

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import builtins
import ast
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import altair as alt
from datetime import datetime

# ──────────────────────────────────────────────
# PAGE CONFIG & SESSION STATE
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Prediksi Stress Mahasiswa - 220711789",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inisialisasi session state untuk riwayat komparasi "What-If"
if "history" not in st.session_state:
    st.session_state.history = []

# ──────────────────────────────────────────────
# CUSTOM CSS – Light Theme
# ──────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    color: #1f2937; /* Diperbaiki dari #fffff */
}
.stApp { background: #f8fafc; }
.block-container { padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1100px; }

/* Sidebar */
section[data-testid="stSidebar"] { background: #ffffff; border-right: 1px solid #e2e8f0; }
section[data-testid="stSidebar"] label { font-size: 0.85rem !important; font-weight: 500 !important; color: #475569 !important; }
.sidebar-header { padding: 0.5rem 0 1rem 0; border-bottom: 1px solid #f1f5f9; margin-bottom: 1rem; }
.sidebar-app-name { font-size: 1.2rem; font-weight: 700; color: #0f172a; }
.sidebar-tagline { font-size: 0.75rem; color: #64748b; margin-top: 0.2rem; line-height: 1.4;}

/* Section Title */
.section-title { font-size: 1.15rem; font-weight: 600; color: #0f172a; margin-bottom: 1rem; padding-bottom: 0.5rem; border-bottom: 1px solid #e2e8f0; }

/* Cards */
.result-card { border-radius: 8px; padding: 1.5rem; text-align: center; border: 1px solid; box-shadow: 0 1px 3px rgba(0,0,0,0.02); }
.result-card-low    { background: #f0fdf4; border-color: #bbf7d0; }
.result-card-medium { background: #fffbeb; border-color: #fef08a; }
.result-card-high   { background: #fef2f2; border-color: #fecaca; }

.result-level { font-size: 0.75rem; font-weight: 700; letter-spacing: 0.05em; text-transform: uppercase; margin-bottom: 0.5rem; font-family: 'JetBrains Mono', monospace; }
.result-level-low    { color: #15803d; }
.result-level-medium { color: #b45309; }
.result-level-high   { color: #b91c1c; }

.result-label { font-size: 1.5rem; font-weight: 700; color: #0f172a; margin: 0 0 0.5rem 0; }
.result-desc { font-size: 0.85rem; color: #475569; line-height: 1.5; margin: 0; }

/* Probabilities */
.prob-container { background: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 1.25rem; }
.prob-row { margin-bottom: 0.8rem; }
.prob-label-row { display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.3rem; }
.prob-label      { font-size: 0.85rem; color: #475569; font-weight: 400; }
.prob-label-bold { font-weight: 600; color: #0f172a; font-size: 0.85rem; }
.prob-pct        { font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; font-weight: 500; color: #0f172a; }
.prob-bar-bg     { background: #f1f5f9; border-radius: 4px; height: 6px; overflow: hidden; }
.prob-bar-low    { height: 100%; background: #22c55e; border-radius: 4px; }
.prob-bar-medium { height: 100%; background: #f59e0b; border-radius: 4px; }
.prob-bar-high   { height: 100%; background: #ef4444; border-radius: 4px; }

/* SHAP & Drivers */
.shap-info-card { background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 6px; padding: 0.85rem 1rem; margin-bottom: 1rem; font-size: 0.85rem; color: #475569; }
.driver-group-title { font-size: 0.8rem; font-weight: 600; margin-bottom: 0.5rem; }
.driver-group-title-red  { color: #b91c1c; }
.driver-group-title-blue { color: #1d4ed8; }
.driver-card { background: #ffffff; border: 1px solid #e2e8f0; border-radius: 6px; padding: 0.6rem 0.8rem; margin-bottom: 0.4rem; display: flex; align-items: center; gap: 0.75rem; }
.driver-rank { font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; color: #94a3b8; min-width: 20px; }
.driver-name { font-size: 0.85rem; font-weight: 500; color: #1e293b; flex: 1; }
.driver-val { font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; font-weight: 500; }
.val-up { color: #ef4444; }
.val-down { color: #3b82f6; }

/* Recommendation Box */
.recommendation-box { background: #e0f2fe; border-left: 4px solid #0284c7; padding: 1rem; border-radius: 0 8px 8px 0; margin-top: 1rem; }
.recommendation-title { font-size: 0.9rem; font-weight: 700; color: #0369a1; margin-bottom: 0.3rem; }
.recommendation-text { font-size: 0.85rem; color: #0c4a6e; line-height: 1.5; }

/* Metrics Custom */
.thesis-metric-box { background: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 1.25rem; text-align: center; box-shadow: 0 1px 2px rgba(0,0,0,0.01); }
.thesis-metric-title { font-size: 0.8rem; font-weight: 600; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.5rem; }
.thesis-metric-val { font-family: 'JetBrains Mono', monospace; font-size: 1.75rem; font-weight: 700; color: #0f172a; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { gap: 2rem; border-bottom: 1px solid #e2e8f0; }
.stTabs [data-baseweb="tab"] { padding-top: 1rem; padding-bottom: 1rem; height: auto; }
.stTabs [aria-selected="true"] { font-weight: 600; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# DICTIONARY REKOMENDASI (Prescriptive Analytics)
# ──────────────────────────────────────────────
ADVICE_MAP = {
    "sleep_quality": "Kualitas tidur yang buruk merupakan penyumbang utama stres. Pertimbangkan untuk menerapkan *sleep hygiene*, membatasi kafein setelah sore hari, dan menjaga jadwal tidur yang konsisten.",
    "study_load": "Beban belajar terlihat membebani. Cobalah teknik manajemen waktu seperti Pomodoro, atau bicarakan dengan penasihat akademik mengenai pengaturan SKS semester depan.",
    "anxiety_level": "Tingkat kecemasan menjadi faktor pendorong utama. Berlatih teknik relaksasi (seperti pernapasan dalam atau *mindfulness*) dapat membantu. Jangan ragu menghubungi layanan konseling kampus.",
    "future_career_concerns": "Kekhawatiran akan karir menyumbang stres secara signifikan. Mengunjungi pusat pengembangan karir kampus atau berdiskusi dengan mentor/alumni mungkin bisa memberikan kejelasan arah.",
    "peer_pressure": "Tekanan teman sebaya terdeteksi tinggi. Fokuslah pada *boundaries* (batasan) pribadi dan kelilingi diri dengan lingkungan pertemanan yang lebih suportif secara emosional.",
    "depression": "Indikator depresi muncul sebagai faktor dominan. Sangat disarankan untuk berkonsultasi dengan psikolog atau layanan kesehatan jiwa kampus untuk mendapatkan bantuan profesional.",
    "social_support": "Kurangnya dukungan sosial berperan besar pada tingkat stres. Cobalah untuk lebih terbuka dengan keluarga, teman terdekat, atau bergabung dengan komunitas hobi di kampus.",
    "default": "Tetap perhatikan keseimbangan antara aktivitas akademik dan waktu istirahat. Mengelola stres sedini mungkin akan sangat membantu produktivitas jangka panjang."
}

# ──────────────────────────────────────────────
# LOAD ARTIFACTS
# ──────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    try:
        model             = joblib.load("xgb_model.pkl") # Disesuaikan dengan penamaan file joblib di script utama
        scaler            = joblib.load("scaler.pkl")
        label_encoder     = joblib.load("label_encoder.pkl")
        selected_features = joblib.load("selected_features.pkl")
    except:
        st.warning("⚠️ Artefak model (xgboost_model.pkl, scaler.pkl, dll) belum ditemukan di direktori saat ini.")
        st.stop()
    return model, scaler, label_encoder, selected_features

model, scaler, le, selected_features = load_artifacts()

STRESS_CONFIG = {
    0: {"label": "Stres Rendah", "level": "Level Rendah", "css_class": "result-card-low", "level_class": "result-level-low", "bar_class": "prob-bar-low", "desc": "Kondisi mental terpantau stabil. Pertahankan pola istirahat dan dukungan sosial yang positif."},
    1: {"label": "Stres Sedang", "level": "Level Sedang", "css_class": "result-card-medium", "level_class": "result-level-medium", "bar_class": "prob-bar-medium", "desc": "Terdapat beberapa indikasi pemicu stres. Perhatikan kualitas tidur dan manajemen beban akademik."},
    2: {"label": "Stres Tinggi", "level": "Level Tinggi", "css_class": "result-card-high", "level_class": "result-level-high", "bar_class": "prob-bar-high", "desc": "Tingkat stres terdeteksi cukup tinggi. Disarankan untuk berkonsultasi dengan layanan konseling akademik atau kesehatan jiwa."},
}

FEATURE_LABELS = {
    "anxiety_level": "Tingkat Kecemasan", "self_esteem": "Harga Diri", "mental_health_history": "Riwayat Masalah Mental",
    "depression": "Tingkat Depresi", "headache": "Frekuensi Sakit Kepala", "blood_pressure": "Tekanan Darah",
    "sleep_quality": "Kualitas Tidur", "breathing_problem": "Masalah Pernapasan", "noise_level": "Kebisingan Lingkungan",
    "living_conditions": "Kondisi Tempat Tinggal", "safety": "Rasa Aman", "basic_needs": "Kebutuhan Dasar",
    "academic_performance": "Performa Akademik", "study_load": "Beban Belajar", "teacher_student_relationship": "Hub. Dosen-Mahasiswa",
    "future_career_concerns": "Kekhawatiran Karir", "social_support": "Dukungan Sosial", "peer_pressure": "Tekanan Teman Sebaya",
    "extracurricular_activities": "Aktivitas Ekskul", "bullying": "Perundungan", "academic_stress_index": "Indeks Stres Akademik",
    "environment_quality_index": "Indeks Kualitas Lingkungan", "social_stress_score": "Skor Stres Sosial",
}

# ──────────────────────────────────────────────
# SHAP Monkey-Patch & NORMALIZATION
# ──────────────────────────────────────────────
def patch_shap_for_xgb_multiclass():
    import shap.explainers._tree as _tree_mod
    _OrigLoader = _tree_mod.XGBTreeModelLoader
    _orig_init  = _OrigLoader.__init__
    _orig_float = builtins.float
    if getattr(_OrigLoader, "_patched_for_multiclass", False): return

    class _ArrayAwareFloat(float):
        def __new__(cls, x=0):
            if isinstance(x, str):
                try: return _orig_float.__new__(cls, x)
                except:
                    try: return _orig_float.__new__(cls, np.mean(ast.literal_eval(x)))
                    except: return _orig_float.__new__(cls, 0.5)
            return _orig_float.__new__(cls, x)

    def _patched_init(self, xgb_model):
        builtins.float = _ArrayAwareFloat
        try: _orig_init(self, xgb_model)
        finally: builtins.float = _orig_float

    _OrigLoader.__init__ = _patched_init
    _OrigLoader._patched_for_multiclass = True

@st.cache_resource
def get_shap_explainer(_model):
    import shap
    patch_shap_for_xgb_multiclass()
    return shap.TreeExplainer(_model)

MINMAX_RANGE = { "study_load": (0, 5), "future_career_concerns": (0, 5), "academic_performance": (0, 5), "peer_pressure": (0, 5), "bullying": (0, 5), "social_support": (0, 3) }

def minmax_norm_single(value, feat):
    lo, hi = MINMAX_RANGE[feat]
    return (value - lo) / (hi - lo) if hi != lo else 0.0

def build_input_row(raw):
    w_study, w_career, w_acad = 0.6342, 0.7426, 0.7209
    total_w = w_study + w_career + w_acad
    academic_stress_index = ((w_study / total_w) * minmax_norm_single(raw["study_load"], "study_load") + (w_career / total_w) * minmax_norm_single(raw["future_career_concerns"], "future_career_concerns") + (w_acad / total_w) * (1 - minmax_norm_single(raw["academic_performance"], "academic_performance")))
    environment_quality_index = (raw["noise_level"] + (5 - raw["living_conditions"]) + (5 - raw["safety"]) + (5 - raw["basic_needs"]))
    social_stress_score = (minmax_norm_single(raw["peer_pressure"], "peer_pressure") + minmax_norm_single(raw["bullying"], "bullying") + (1 - minmax_norm_single(raw["social_support"], "social_support")))

    full = {**raw, "academic_stress_index": academic_stress_index, "environment_quality_index": environment_quality_index, "social_stress_score": social_stress_score}
    return pd.DataFrame([full])[selected_features]


# ══════════════════════════════════════════════
# SIDEBAR INPUT & IDENTITAS
# ══════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div class="sidebar-header">
        <div class="sidebar-app-name">Prediksi Stress Mahasiswa</div>
        <div class="sidebar-tagline">Explainable AI untuk Prediksi Tingkat Stres Mahasiswa</div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("🧠 Psikologis", expanded=True):
        anxiety_level         = st.slider("Kecemasan", 0, 21, 10, help="Skala 0 (Tidak cemas sama sekali) hingga 21 (Sangat cemas/Kecemasan berat)")
        self_esteem           = st.slider("Harga Diri", 0, 30, 15, help="Skala 0 (Sangat rendah/Tidak percaya diri) hingga 30 (Sangat tinggi)")
        mental_health_history = st.selectbox("Riwayat Mental", [0, 1], format_func=lambda x: "Tidak ada" if x == 0 else "Ada riwayat", help="0 = Tidak pernah memiliki masalah mental. 1 = Memiliki riwayat masalah mental sebelumnya.")
        depression            = st.slider("Depresi", 0, 27, 10, help="Skala 0 (Tidak ada gejala depresi) hingga 27 (Depresi sangat berat)")

    with st.expander("🩺 Kesehatan Fisik", expanded=False):
        headache          = st.slider("Sakit Kepala", 0, 5, 2, help="Frekuensi: 0 (Tidak pernah) hingga 5 (Sangat sering / hampir setiap hari)")
        blood_pressure    = st.slider("Tekanan Darah", 1, 3, 2, help="Kondisi: 1 (Hipotensi/Rendah), 2 (Normal), 3 (Hipertensi/Tinggi)")
        sleep_quality     = st.slider("Kualitas Tidur", 1, 5, 3, help="Skala 1 (Sangat buruk / insomnia) hingga 5 (Sangat nyenyak dan berkualitas)")
        breathing_problem = st.slider("Pernapasan", 0, 5, 1, help="Gangguan napas/sesak: 0 (Tidak pernah bermasalah) hingga 5 (Sering bermasalah/sesak)")

    with st.expander("🏠 Lingkungan", expanded=False):
        noise_level       = st.slider("Kebisingan", 0, 5, 2, help="Skala 0 (Lingkungan sangat tenang) hingga 5 (Sangat bising / mengganggu)")
        living_conditions = st.slider("Kondisi Tempat", 1, 5, 3, help="Kelayakan: 1 (Sangat buruk/tidak layak) hingga 5 (Sangat baik dan nyaman)")
        safety            = st.slider("Rasa Aman", 1, 5, 3, help="Skala 1 (Sangat merasa tidak aman) hingga 5 (Sangat aman dari gangguan)")
        basic_needs       = st.slider("Kebutuhan Dasar", 1, 5, 3, help="Pemenuhan makan/minum/fasilitas: 1 (Sering tidak terpenuhi) hingga 5 (Selalu terpenuhi dengan baik)")

    with st.expander("📚 Akademik", expanded=False):
        academic_performance         = st.slider("Performa", 1, 5, 3, help="Kinerja akademik: 1 (Sangat buruk/IPK rendah) hingga 5 (Sangat baik/IPK sangat tinggi)")
        study_load                   = st.slider("Beban Belajar", 1, 5, 3, help="Skala 1 (Beban tugas sangat ringan) hingga 5 (Beban tugas sangat berat)")
        teacher_student_relationship = st.slider("Hub. Dosen", 1, 5, 3, help="Interaksi dengan dosen: 1 (Sangat buruk/sering konflik) hingga 5 (Sangat baik/suportif)")
        future_career_concerns       = st.slider("Khawatir Karir", 1, 5, 3, help="Kecemasan masa depan: 1 (Tidak khawatir sama sekali) hingga 5 (Sangat cemas akan karir)")

    with st.expander("👥 Sosial", expanded=False):
        social_support             = st.slider("Dukungan Sosial", 0, 3, 2, help="Tingkat dukungan keluarga/teman: 0 (Tidak ada) hingga 3 (Dukungan penuh)")
        peer_pressure              = st.slider("Tekanan Teman", 1, 5, 3, help="Tuntutan pertemanan (Peer pressure): 1 (Tidak ada tekanan) hingga 5 (Tekanan sangat tinggi)")
        extracurricular_activities = st.slider("Ekskul", 0, 5, 2, help="Keterlibatan organisasi/ekskul: 0 (Tidak aktif sama sekali) hingga 5 (Sangat aktif/sibuk)")
        bullying                   = st.slider("Perundungan", 0, 5, 1, help="Tingkat perundungan (Bullying): 0 (Tidak pernah mengalami) hingga 5 (Sering menjadi korban)")

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Tombol Update hanya me-refresh UI
    st.button("Update Prediksi", use_container_width=True, type="primary")

    # Identitas Peneliti di Sidebar
    st.markdown("<hr style='margin: 2rem 0 1rem 0;'>", unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; color: #64748b; font-size: 0.8rem; line-height: 1.5;">
        <b>Penelitian Tugas Akhir</b><br>
        Kevin Philips Tanamas<br>
        <span style="font-family: 'JetBrains Mono', monospace;">NPM: 220711789</span>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════
# DATA PROCESSING
# ══════════════════════════════════════════════
raw_input = {
    "anxiety_level": anxiety_level, "self_esteem": self_esteem, "mental_health_history": mental_health_history,
    "depression": depression, "headache": headache, "blood_pressure": blood_pressure, "sleep_quality": sleep_quality,
    "breathing_problem": breathing_problem, "noise_level": noise_level, "living_conditions": living_conditions,
    "safety": safety, "basic_needs": basic_needs, "academic_performance": academic_performance, "study_load": study_load,
    "teacher_student_relationship": teacher_student_relationship, "future_career_concerns": future_career_concerns,
    "social_support": social_support, "peer_pressure": peer_pressure, "extracurricular_activities": extracurricular_activities,
    "bullying": bullying,
}
df_input_raw     = build_input_row(raw_input)
df_input_scaled  = pd.DataFrame(scaler.transform(df_input_raw), columns=selected_features)
prediction       = model.predict(df_input_scaled)
prediction_proba = model.predict_proba(df_input_scaled)
pred_class       = int(prediction[0])
cfg              = STRESS_CONFIG[pred_class]


# ══════════════════════════════════════════════
# MAIN CONTENT
# ══════════════════════════════════════════════
st.markdown("### 🎓 Prediksi Tingkat Stres Mahasiswa (XGBoost + SHAP)")
st.markdown("<p style='color:#64748b; font-size:0.9rem; margin-top:-0.5rem;'>Implementasi Machine Learning berdasarkan metodologi CRISP-DM.</p>", unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Prediksi Individu", "Interpretasi & Saran", "Ringkasan Data", "Prediksi Batch", "Performa Model"])

# ── TAB 1: HASIL PREDIKSI & SKENARIO ──
with tab1:
    st.markdown("<br>", unsafe_allow_html=True)
    col_result, col_prob = st.columns([1, 1.2], gap="large")

    with col_result:
        st.markdown("<div class='section-title'>Kesimpulan Model</div>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class="{cfg['css_class']} result-card">
            <div class="result-level {cfg['level_class']}">{cfg['level']}</div>
            <div class="result-label">{cfg['label']}</div>
            <p class="result-desc">{cfg['desc']}</p>
        </div>
        """, unsafe_allow_html=True)

        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            # Fitur Simpan Skenario (Live Demo)
            if st.button("💾 Simpan ke Skenario", use_container_width=True):
                timestamp = datetime.now().strftime("%H:%M:%S")
                st.session_state.history.append({
                    "Waktu": timestamp,
                    "Kecemasan": anxiety_level,
                    "Beban_Belajar": study_load,
                    "Tidur": sleep_quality,
                    "Hasil_Prediksi": cfg['label'],
                    "Keyakinan": f"{max(prediction_proba[0]):.1%}"
                })
                st.success("Tersimpan!")
                
        with col_btn2:
            # Ekspor Hasil Individu
            result_df = pd.DataFrame([raw_input])
            result_df["Hasil_Prediksi"] = cfg['label']
            csv = result_df.to_csv(index=False).encode('utf-8')
            st.download_button(label="📥 Unduh (CSV)", data=csv, file_name='laporan_stres_individu.csv', mime='text/csv', use_container_width=True)

    with col_prob:
        st.markdown("<div class='section-title'>Probabilitas Kelas</div>", unsafe_allow_html=True)
        st.markdown("<div class='prob-container'>", unsafe_allow_html=True)
        proba_labels = ["Stres Rendah", "Stres Sedang", "Stres Tinggi"]
        bar_classes  = ["prob-bar-low", "prob-bar-medium", "prob-bar-high"]

        for i, (lbl, bar_cls) in enumerate(zip(proba_labels, bar_classes)):
            pct = prediction_proba[0][i]
            bold = "prob-label-bold" if i == pred_class else "prob-label"
            st.markdown(f"""
            <div class="prob-row">
                <div class="prob-label-row"><span class="{bold}">{lbl}</span><span class="prob-pct">{pct:.1%}</span></div>
                <div class="prob-bar-bg"><div class="{bar_cls}" style="width:{int(pct * 100)}%"></div></div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Tabel Riwayat Skenario (Muncul jika ada data)
    if st.session_state.history:
        st.markdown("<div class='section-title' style='margin-top: 2rem;'>Riwayat Komparasi Skenario (Live Demo)</div>", unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(st.session_state.history), use_container_width=True)
        if st.button("Hapus Riwayat"):
            st.session_state.history = []
            st.rerun()


# ── TAB 2: SHAP EXPLANATION & PRESCRIPTIVE ANALYTICS ──
with tab2:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="shap-info-card">
        <strong>Interpretasi Model (SHAP) & Rekomendasi</strong><br>
        Teknologi <i>Explainable AI</i> mengungkap faktor dominan. Sistem juga memberikan rekomendasi tindak lanjut berdasarkan analisis.
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
            sv = shap_vals[pred_class][0] if isinstance(shap_vals, list) else shap_vals[0, :, pred_class]
            ev = shap_explainer.expected_value[pred_class] if isinstance(shap_vals, list) else (shap_explainer.expected_value[pred_class] if hasattr(shap_explainer.expected_value, "__len__") else shap_explainer.expected_value)

            shap_series  = pd.Series(sv, index=selected_features)
            top_positive = shap_series[shap_series > 0].sort_values(ascending=False).head(4)
            top_negative = shap_series[shap_series < 0].sort_values(ascending=True).head(4)

            # --- Prescriptive Analytics (Menampilkan Saran berdasarkan pendorong stres terbesar) ---
            if not top_positive.empty and pred_class > 0: # Hanya beri saran jika stres Sedang/Tinggi
                top_driver = top_positive.index[0]
                advice = ADVICE_MAP.get(top_driver, ADVICE_MAP["default"])
                
                st.markdown(f"""
                <div class="recommendation-box">
                    <div class="recommendation-title">💡 Saran Tindak Lanjut: Fokus pada "{FEATURE_LABELS.get(top_driver, top_driver)}"</div>
                    <div class="recommendation-text">{advice}</div>
                </div>
                <br>
                """, unsafe_allow_html=True)

            col_up, col_down = st.columns(2, gap="medium")
            with col_up:
                st.markdown("<div class='driver-group-title driver-group-title-red'>Top Faktor Pendorong (+)</div>", unsafe_allow_html=True)
                for rank, (feat, val) in enumerate(top_positive.items(), 1):
                    st.markdown(f'<div class="driver-card"><span class="driver-rank">#{rank}</span><span class="driver-name">{FEATURE_LABELS.get(feat, feat)}</span><span class="driver-val val-up">+{val:.3f}</span></div>', unsafe_allow_html=True)
            with col_down:
                st.markdown("<div class='driver-group-title driver-group-title-blue'>Top Faktor Penekan (-)</div>", unsafe_allow_html=True)
                for rank, (feat, val) in enumerate(top_negative.items(), 1):
                    st.markdown(f'<div class="driver-card"><span class="driver-rank">#{rank}</span><span class="driver-name">{FEATURE_LABELS.get(feat, feat)}</span><span class="driver-val val-down">{val:.3f}</span></div>', unsafe_allow_html=True)

            st.markdown("<div class='section-title' style='margin-top: 2rem;'>Alur Prediksi (Waterfall Chart)</div>", unsafe_allow_html=True)
            exp = shap.Explanation(values=sv, base_values=float(ev), data=df_input_scaled.iloc[0].values, feature_names=[FEATURE_LABELS.get(f, f) for f in selected_features])
            fig, ax = plt.subplots(figsize=(8, 4.5))
            fig.patch.set_facecolor("#f8fafc")
            shap.plots.waterfall(exp, show=False, max_display=10)
            plt.gca().set_facecolor("#f8fafc")
            plt.gca().tick_params(colors="#1f2937")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        except Exception as e:
            st.error(f"Grafik SHAP gagal dimuat: {e}")


# ── TAB 3: DATA INPUT SUMMARY (Updated dengan Bar Horizontal Berskala) ──
with tab3:
    st.markdown("<br>", unsafe_allow_html=True)
    col_a, col_b = st.columns([1.5, 1], gap="large")

    with col_a:
        st.markdown("<div class='section-title'>Visualisasi Input Mayor (Skala Relatif)</div>", unsafe_allow_html=True)
        st.markdown("<p style='color:#64748b; font-size:0.85rem; margin-top:-0.5rem; margin-bottom:1rem;'>Nilai dinormalisasi ke persentase (0-100%) agar adil untuk dibandingkan, mengingat setiap fitur memiliki rentang maksimal yang berbeda-beda.</p>", unsafe_allow_html=True)

        # Mendefinisikan nilai maksimal slider untuk proporsi persentase yang adil
        max_ranges = {
            "anxiety_level": 21,
            "depression": 27,
            "self_esteem": 30,
            "study_load": 5,
            "peer_pressure": 5
        }

        selected_to_plot = {k: v for k, v in raw_input.items() if k in max_ranges.keys()}

        plot_data = []
        for k, v in selected_to_plot.items():
            max_val = max_ranges[k]
            pct = (v / max_val) * 100
            plot_data.append({
                "Faktor": FEATURE_LABELS.get(k, k),
                "Intensitas (%)": pct,
                "Nilai Asli": f"{v} / {max_val}"
            })

        chart_df = pd.DataFrame(plot_data)

        # Menggunakan Altair untuk Bar Chart horizontal dengan label ujung
        bars = alt.Chart(chart_df).mark_bar(color='#3b82f6', cornerRadiusEnd=4, height=22).encode(
            x=alt.X('Intensitas (%):Q', scale=alt.Scale(domain=[0, 100]), title='Skala Relatif Keparahan (%)'),
            y=alt.Y('Faktor:N', sort='-x', title=None, axis=alt.Axis(labelLimit=150, labelFontSize=12)),
            tooltip=['Faktor', 'Nilai Asli', alt.Tooltip('Intensitas (%):Q', format='.1f')]
        )

        # Teks untuk menunjukkan nilai asli di sebelah bar
        text = bars.mark_text(
            align='left',
            baseline='middle',
            dx=5,
            fontSize=11,
            fontWeight=600,
            color='#1e293b'
        ).encode(
            text='Nilai Asli:N'
        )

        st.altair_chart((bars + text).properties(height=280), use_container_width=True)

    with col_b:
        st.markdown("<div class='section-title'>Fitur Kalkulasi Model</div>", unsafe_allow_html=True)
        engineered = {"academic_stress_index": df_input_raw["academic_stress_index"].values[0], "environment_quality_index": df_input_raw["environment_quality_index"].values[0], "social_stress_score": df_input_raw["social_stress_score"].values[0]}
        for feat, val in engineered.items():
            st.metric(label=FEATURE_LABELS.get(feat, feat), value=f"{val:.3f}")
        st.caption("Nilai turunan yang dikalkulasi otomatis sesuai algoritma di tahap Data Preparation.")


# ── TAB 4: BATCH PREDICTION ──
with tab4:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Proses Dataset Sekaligus (Batch Prediction)</div>", unsafe_allow_html=True)
    st.info("Fitur validasi riset: Unggah file CSV data pengujian untuk memprediksi puluhan hingga ratusan responden secara instan.")

    uploaded_file = st.file_uploader("Unggah File CSV Dataset Uji", type=["csv"])

    if uploaded_file is not None:
        try:
            df_batch = pd.read_csv(uploaded_file)
            st.write("Preview Data Uji:", df_batch.head())

            if st.button("Jalankan Prediksi Batch", type="primary"):
                with st.spinner("Model sedang memproses..."):
                    processed_rows = []
                    for _, row in df_batch.iterrows():
                        processed_rows.append(build_input_row(row.to_dict()).iloc[0])

                    df_batch_raw = pd.DataFrame(processed_rows)
                    df_batch_scaled = pd.DataFrame(scaler.transform(df_batch_raw), columns=selected_features)

                    batch_preds = model.predict(df_batch_scaled)
                    df_batch["Hasil_Prediksi_Numerik"] = batch_preds
                    df_batch["Hasil_Prediksi_Label"] = df_batch["Hasil_Prediksi_Numerik"].map(lambda x: STRESS_CONFIG[x]["label"])

                    st.success(f"Berhasil mengklasifikasikan {len(df_batch)} baris data!")
                    st.dataframe(df_batch[["Hasil_Prediksi_Label"] + [c for c in df_batch.columns if c not in ["Hasil_Prediksi_Label", "Hasil_Prediksi_Numerik"]]])

                    csv_batch = df_batch.to_csv(index=False).encode('utf-8')
                    st.download_button(label="📥 Unduh Hasil Uji Batch (CSV)", data=csv_batch, file_name='hasil_prediksi_batch.csv', mime='text/csv')
        except Exception as e:
            st.error(f"Terjadi kesalahan saat memproses file: {e}. Pastikan format kolom sama persis dengan atribut fitur.")

# ── TAB 5: PERFORMA MODEL ──
with tab5:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Evaluasi & Metrik Model Terbaik (XGBoost)</div>", unsafe_allow_html=True)
    st.markdown("""
    <p style='color:#475569; font-size:0.9rem; line-height:1.6; margin-bottom: 1.5rem;'>
        Berdasarkan komparasi algoritma (XGBoost vs Random Forest vs LightGBM) yang telah dievaluasi dengan metode <strong>Cross-Validation 5-Fold</strong> serta dilacak secara riwayat terpusat menggunakan <strong>Weights & Biases (W&B)</strong>, model XGBoost ditetapkan sebagai algoritma terbaik dengan performa sebagai berikut:
    </p>
    """, unsafe_allow_html=True)

    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.markdown("<div class='thesis-metric-box'><div class='thesis-metric-title'>Akurasi Uji</div><div class='thesis-metric-val'>~90%+</div></div>", unsafe_allow_html=True)
    with col_m2:
        st.markdown("<div class='thesis-metric-box'><div class='thesis-metric-title'>Macro F1-Score</div><div class='thesis-metric-val'>Terbaik</div></div>", unsafe_allow_html=True)
    with col_m3:
        st.markdown("<div class='thesis-metric-box'><div class='thesis-metric-title'>ROC-AUC</div><div class='thesis-metric-val'>Terbaik</div></div>", unsafe_allow_html=True)
    with col_m4:
        st.markdown("<div class='thesis-metric-box'><div class='thesis-metric-title'>CV F1 Mean</div><div class='thesis-metric-val'>Stabil</div></div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    **Metodologi Pengembangan (CRISP-DM)**
    1. **Business Understanding**: Merumuskan permasalahan deteksi tingkat stres di kalangan mahasiswa.
    2. **Data Understanding**: Menemukan korelasi fitur psikologis, lingkungan, akademik, dan fisik terhadap stres.
    3. **Data Preparation**: Imputasi data, perhitungan indeks fitur gabungan (*Feature Engineering*), Seleksi Fitur, dan Normalisasi (Standard Scaler).
    4. **Modeling**: Hyperparameter Tuning dan perbandingan algoritma ansambel menggunakan W&B.
    5. **Evaluation**: XGBoost mendominasi kinerja prediksi serta memiliki kompatibilitas penuh dengan SHAP TreeExplainer.
    6. **Deployment**: Aplikasi *Machine Learning* interaktif ini dibangun menggunakan Streamlit.
    """)