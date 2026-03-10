# ============================================================
# app.py — Prediksi Tingkat Stres Mahasiswa
# Framework  : Streamlit
# Model      : XGBoost + SHAP Explainability
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import joblib
import shap
import os
import json
import re
import tempfile
import warnings
warnings.filterwarnings("ignore")

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="StressScope — Prediksi Stres Mahasiswa",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

  :root {
    --bg:      #0d0f1a;
    --surface: #141728;
    --border:  #252840;
    --accent1: #6c63ff;
    --accent2: #ff6b8a;
    --accent3: #43e8d8;
    --text:    #e8eaf6;
    --muted:   #8b8fad;
    --low:     #43e8d8;
    --mid:     #f9c74f;
    --high:    #ff6b8a;
  }

  html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg);
    color: var(--text);
  }

  .hero {
    background: linear-gradient(135deg, #1a1d35 0%, #0d0f1a 60%, #1a1230 100%);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
  }
  .hero::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 220px; height: 220px;
    background: radial-gradient(circle, rgba(108,99,255,0.25) 0%, transparent 70%);
    border-radius: 50%;
  }
  .hero::after {
    content: '';
    position: absolute;
    bottom: -40px; left: 40px;
    width: 140px; height: 140px;
    background: radial-gradient(circle, rgba(255,107,138,0.18) 0%, transparent 70%);
    border-radius: 50%;
  }
  .hero-tag {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.18em;
    color: var(--accent3);
    text-transform: uppercase;
    margin-bottom: 0.5rem;
  }
  .hero h1 {
    font-family: 'Space Mono', monospace;
    font-size: 2.4rem;
    font-weight: 700;
    background: linear-gradient(90deg, #6c63ff, #ff6b8a);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 0.6rem 0;
  }
  .hero p {
    color: var(--muted);
    font-size: 1rem;
    max-width: 640px;
    line-height: 1.65;
    margin: 0;
  }

  .card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1.2rem;
  }
  .card-title {
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 0.12em;
    color: var(--accent1);
    text-transform: uppercase;
    margin-bottom: 0.8rem;
  }

  .badge {
    display: inline-block;
    padding: 0.35rem 1.1rem;
    border-radius: 999px;
    font-family: 'Space Mono', monospace;
    font-weight: 700;
    font-size: 0.95rem;
    letter-spacing: 0.06em;
  }
  .badge-low  { background: rgba(67,232,216,0.18); color: var(--low);  border: 1px solid var(--low);  }
  .badge-mid  { background: rgba(249,199, 79,0.18); color: var(--mid);  border: 1px solid var(--mid);  }
  .badge-high { background: rgba(255,107,138,0.18); color: var(--high); border: 1px solid var(--high); }

  .metric-row { display: flex; gap: 1rem; margin-bottom: 1.2rem; }
  .metric-tile {
    flex: 1;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.1rem 1.3rem;
    text-align: center;
  }
  .metric-tile .val {
    font-family: 'Space Mono', monospace;
    font-size: 1.75rem;
    font-weight: 700;
  }
  .metric-tile .lbl {
    font-size: 0.72rem;
    color: var(--muted);
    margin-top: 0.2rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
  }

  .prob-row { margin-bottom: 0.6rem; }
  .prob-label {
    font-size: 0.82rem;
    color: var(--muted);
    margin-bottom: 0.2rem;
    display: flex;
    justify-content: space-between;
  }
  .prob-bar-bg {
    background: var(--border);
    border-radius: 999px;
    height: 10px;
    overflow: hidden;
  }
  .prob-bar-fill {
    height: 100%;
    border-radius: 999px;
  }

  [data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
  }
  [data-testid="stSidebar"] * { color: var(--text) !important; }

  .stTabs [data-baseweb="tab-list"] {
    gap: 0.5rem;
    background: transparent;
    border-bottom: 1px solid var(--border);
  }
  .stTabs [data-baseweb="tab"] {
    background: transparent !important;
    border-radius: 8px 8px 0 0 !important;
    color: var(--muted) !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.08em;
    border: none !important;
    padding: 0.5rem 1.2rem !important;
  }
  .stTabs [aria-selected="true"] {
    background: var(--surface) !important;
    color: var(--accent1) !important;
    border: 1px solid var(--border) !important;
    border-bottom: 1px solid var(--surface) !important;
  }

  [data-testid="stExpander"] {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
  }

  hr { border-color: var(--border) !important; }
  ::-webkit-scrollbar { width: 6px; }
  ::-webkit-scrollbar-track { background: var(--bg); }
  ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }

  #MainMenu, footer { visibility: hidden; }
  header { visibility: hidden; }

  .stButton>button {
    background: linear-gradient(135deg, var(--accent1), var(--accent2)) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.08em;
    padding: 0.6rem 1.6rem !important;
    width: 100%;
  }

  .info-box {
    background: rgba(108,99,255,0.08);
    border-left: 3px solid var(--accent1);
    border-radius: 0 8px 8px 0;
    padding: 0.8rem 1rem;
    font-size: 0.88rem;
    color: var(--muted);
    margin-bottom: 1rem;
  }
  .warn-box {
    background: rgba(255,107,138,0.08);
    border-left: 3px solid var(--accent2);
    border-radius: 0 8px 8px 0;
    padding: 0.8rem 1rem;
    font-size: 0.88rem;
    color: var(--muted);
    margin-bottom: 1rem;
  }
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
# LOAD MODEL
# Reads selected_features.pkl to know what was ACTUALLY trained,
# then builds sidebar sliders ONLY for those features.
# ════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_artifacts():
    required = ["xgb_model.pkl", "scaler.pkl", "label_encoder.pkl", "selected_features.pkl"]
    missing  = [f for f in required if not os.path.exists(f)]
    if missing:
        return None, None, None, None, missing
    model             = joblib.load("xgb_model.pkl")
    scaler            = joblib.load("scaler.pkl")
    le                = joblib.load("label_encoder.pkl")
    selected_features = joblib.load("selected_features.pkl")
    return model, scaler, le, selected_features, []


# ════════════════════════════════════════════════════════════════
# SHAP FIX
# XGBoost >= 2.x stores base_score as a comma-separated list in
# scientific notation e.g. '[1.93E-2,-2.51E-2,5.83E-3]'.
# float() cannot parse this. Fix: save model to JSON, patch
# base_score to scalar "0.5" (safe XGBoost default), reload,
# then pass the patched booster to TreeExplainer.
# ════════════════════════════════════════════════════════════════
def get_shap_explainer(xgb_sklearn_model):
    import xgboost as xgb

    booster = xgb_sklearn_model.get_booster()

    # Save to temp JSON file
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        tmp_path = tmp.name
    booster.save_model(tmp_path)

    # Patch base_score in the JSON
    with open(tmp_path, "r") as f:
        model_json = json.load(f)
    try:
        model_json["learner"]["learner_model_param"]["base_score"] = "0.5"
    except (KeyError, TypeError):
        pass
    with open(tmp_path, "w") as f:
        json.dump(model_json, f)

    # Reload patched booster
    patched = xgb.Booster()
    patched.load_model(tmp_path)
    os.unlink(tmp_path)

    return shap.TreeExplainer(patched)


# ── Dark matplotlib helper ────────────────────────────────────
def dark_fig(w=9, h=4):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor("#141728")
    ax.set_facecolor("#141728")
    for spine in ax.spines.values():
        spine.set_edgecolor("#252840")
    ax.tick_params(colors="#8b8fad", labelsize=8)
    ax.xaxis.label.set_color("#8b8fad")
    ax.yaxis.label.set_color("#8b8fad")
    ax.title.set_color("#e8eaf6")
    return fig, ax


# ── Feature display names & slider params ────────────────────
DISPLAY_NAMES = {
    "study_load":                   ("📚", "Beban Belajar"),
    "academic_performance":         ("🎓", "Performa Akademik"),
    "future_career_concerns":       ("🔮", "Kekhawatiran Karir"),
    "teacher_student_relationship": ("🤝", "Hub. Guru-Murid"),
    "social_support":               ("👥", "Dukungan Sosial"),
    "sleep_duration":               ("🌙", "Durasi Tidur"),
    "daily_screen_time":            ("📱", "Waktu Layar Harian"),
    "exam_anxiety":                 ("😰", "Kecemasan Ujian"),
    "academic_stress_index":        ("🧮", "Academic Stress Index"),
    "screen_sleep_ratio":           ("⚖️",  "Screen/Sleep Ratio"),
    "anxiety_level":                ("😟", "Tingkat Kecemasan"),
    "self_esteem":                  ("💪", "Self-Esteem"),
    "depression":                   ("💙", "Depresi"),
    "mental_health_history":        ("🧠", "Riwayat Kes. Mental"),
    "headache":                     ("🤕", "Sakit Kepala"),
    "blood_pressure":               ("❤️",  "Tekanan Darah"),
    "sleep_quality":                ("😴", "Kualitas Tidur"),
    "breathing_problem":            ("🫁", "Masalah Pernapasan"),
    "noise_level":                  ("🔊", "Tingkat Kebisingan"),
    "living_conditions":            ("🏠", "Kondisi Tempat Tinggal"),
    "safety":                       ("🛡️",  "Rasa Aman"),
    "basic_needs":                  ("🍽️",  "Kebutuhan Dasar"),
    "peer_pressure":                ("👫", "Tekanan Teman Sebaya"),
    "extracurricular_activities":   ("🎨", "Aktivitas Ekskul"),
    "bullying":                     ("⚠️",  "Bullying"),
}

# (min, max, default, step)
SLIDER_PARAMS = {
    "study_load":                   (0.0, 10.0,  5.0, 0.1),
    "academic_performance":         (0.0, 10.0,  7.0, 0.1),
    "future_career_concerns":       (0.0, 10.0,  5.0, 0.1),
    "teacher_student_relationship": (0.0, 10.0,  5.0, 0.1),
    "social_support":               (0.0, 10.0,  5.0, 0.1),
    "sleep_duration":               (0.0, 12.0,  7.0, 0.5),
    "daily_screen_time":            (0.0, 15.0,  4.0, 0.5),
    "exam_anxiety":                 (0.0, 10.0,  5.0, 0.1),
    "anxiety_level":                (0.0, 21.0, 10.0, 1.0),
    "self_esteem":                  (0.0, 30.0, 15.0, 1.0),
    "depression":                   (0.0, 27.0, 10.0, 1.0),
    "mental_health_history":        (0.0,  1.0,  0.0, 1.0),
    "headache":                     (0.0,  5.0,  2.0, 1.0),
    "blood_pressure":               (1.0,  3.0,  2.0, 1.0),
    "sleep_quality":                (0.0,  5.0,  3.0, 1.0),
    "breathing_problem":            (0.0,  5.0,  2.0, 1.0),
    "noise_level":                  (0.0,  5.0,  2.0, 1.0),
    "living_conditions":            (0.0,  5.0,  3.0, 1.0),
    "safety":                       (0.0,  5.0,  3.0, 1.0),
    "basic_needs":                  (0.0,  5.0,  3.0, 1.0),
    "peer_pressure":                (0.0,  5.0,  2.0, 1.0),
    "extracurricular_activities":   (0.0,  5.0,  2.0, 1.0),
    "bullying":                     (0.0,  5.0,  1.0, 1.0),
}

STRESS_INFO = {
    0: {
        "label": "RENDAH", "badge": "badge-low", "color": "#43e8d8", "emoji": "😊",
        "desc": "Tingkat stres Anda tergolong <strong>rendah</strong>. Kondisi psikologis, "
                "sosial, dan akademik Anda berada pada zona sehat.",
        "tips": ["Pertahankan rutinitas tidur yang baik",
                 "Jaga keseimbangan belajar dan istirahat",
                 "Terus bangun dukungan sosial yang positif"],
    },
    1: {
        "label": "SEDANG", "badge": "badge-mid", "color": "#f9c74f", "emoji": "😐",
        "desc": "Tingkat stres Anda berada di level <strong>sedang</strong>. Ada beberapa "
                "faktor yang mulai menekan. Segera lakukan intervensi ringan.",
        "tips": ["Evaluasi beban studi dan prioritaskan tugas",
                 "Coba teknik relaksasi seperti meditasi",
                 "Bicarakan kekhawatiran kepada konselor"],
    },
    2: {
        "label": "TINGGI", "badge": "badge-high", "color": "#ff6b8a", "emoji": "😟",
        "desc": "Tingkat stres Anda tergolong <strong>tinggi</strong>. Sangat disarankan "
                "untuk segera mencari bantuan profesional atau konselor kampus.",
        "tips": ["Konsultasikan dengan psikolog atau konselor kampus",
                 "Kurangi beban berlebih, delegasikan tugas",
                 "Prioritaskan tidur, olahraga, dan pola makan sehat"],
    },
}

ENGINEERED_SET = {"academic_stress_index", "screen_sleep_ratio"}


# ════════════════════════════════════════════════════════════════
# LOAD ARTIFACTS
# ════════════════════════════════════════════════════════════════
model, scaler, le, selected_features, missing = load_artifacts()


# ════════════════════════════════════════════════════════════════
# SIDEBAR
# Only shows sliders for features the model was actually trained on.
# Engineered features (academic_stress_index, screen_sleep_ratio)
# are computed automatically — no slider needed for those.
# ════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="font-family:'Space Mono',monospace; font-size:1.1rem; font-weight:700;
         background:linear-gradient(90deg,#6c63ff,#ff6b8a);
         -webkit-background-clip:text; -webkit-text-fill-color:transparent;
         margin-bottom:0.3rem;">🧠 StressScope</div>
    <div style="font-size:0.72rem; color:#8b8fad; letter-spacing:0.1em;
         text-transform:uppercase; margin-bottom:1.5rem;">Input Parameter Mahasiswa</div>
    """, unsafe_allow_html=True)

    inputs = {}

    if selected_features:
        slider_feats = [f for f in selected_features if f not in ENGINEERED_SET]
        for feat in slider_feats:
            icon, label = DISPLAY_NAMES.get(feat, ("📌", feat.replace("_", " ").title()))
            lo, hi, default, step = SLIDER_PARAMS.get(feat, (0.0, 10.0, 5.0, 0.1))
            with st.expander(f"{icon}  {label}", expanded=False):
                inputs[feat] = st.slider(
                    label, min_value=lo, max_value=hi,
                    value=default, step=step,
                    label_visibility="collapsed", key=f"s_{feat}"
                )

        # Preview auto-computed engineered features if used
        has_eng = any(f in selected_features for f in ENGINEERED_SET)
        if has_eng:
            st.markdown("""
            <div style="font-size:0.7rem; color:#8b8fad; letter-spacing:0.08em;
                 text-transform:uppercase; margin:0.8rem 0 0.4rem;">
            🛠 Dihitung Otomatis
            </div>""", unsafe_allow_html=True)
            if "academic_stress_index" in selected_features:
                sl  = inputs.get("study_load", 5.0)
                ea  = inputs.get("exam_anxiety", 5.0)
                ap  = inputs.get("academic_performance", 7.0)
                asi = 0.4*sl + 0.4*ea - 0.2*ap
                st.caption(f"🧮 Acad. Stress Index: **{asi:.3f}**")
            if "screen_sleep_ratio" in selected_features:
                sd  = inputs.get("sleep_duration", 7.0)
                dst = inputs.get("daily_screen_time", 4.0)
                ssr = dst / (sd if sd != 0 else 0.1)
                st.caption(f"⚖️  Screen/Sleep Ratio: **{ssr:.3f}**")

    st.markdown("<hr>", unsafe_allow_html=True)
    run_btn = st.button("🔍  Analisis Sekarang", use_container_width=True)


# ── Hero ─────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-tag">Research Tool · CRISP-DM · XGBoost + SHAP</div>
  <h1>StressScope</h1>
  <p>Sistem prediksi tingkat stres mahasiswa berbasis <strong>Machine Learning</strong> dengan
     explainability penuh menggunakan <strong>SHAP</strong>. Masukkan parameter di sidebar,
     klik <em>Analisis Sekarang</em>, dan dapatkan wawasan mendalam tentang faktor-faktor
     yang memengaruhi stres Anda.</p>
</div>
""", unsafe_allow_html=True)

if missing:
    st.markdown(f"""
    <div class="warn-box">
    ⚠️ <strong>File model tidak ditemukan:</strong>
    <code>{'</code>, <code>'.join(missing)}</code><br><br>
    Jalankan notebook pelatihan terlebih dahulu, lalu letakkan semua file <code>.pkl</code>
    di folder yang sama dengan <code>app.py</code>.
    </div>""", unsafe_allow_html=True)
    st.stop()


tab_pred, tab_about, tab_data = st.tabs([
    "🔬  PREDIKSI & SHAP",
    "📖  TENTANG SISTEM",
    "📊  PANDUAN FITUR",
])


# ════════════════════════════════════════════════════════════════
# TAB 1 — PREDIKSI
# ════════════════════════════════════════════════════════════════
with tab_pred:

    if not run_btn:
        st.markdown("""
        <div class="info-box">
        💡 Atur parameter mahasiswa di <strong>sidebar kiri</strong>, lalu klik
        <strong>"Analisis Sekarang"</strong> untuk melihat prediksi dan penjelasan SHAP.
        </div>""", unsafe_allow_html=True)
    else:
        # ── Build full feature dict (including engineered cols) ──
        all_inputs = dict(inputs)

        if "academic_stress_index" in selected_features:
            sl  = all_inputs.get("study_load", 5.0)
            ea  = all_inputs.get("exam_anxiety", 5.0)
            ap  = all_inputs.get("academic_performance", 7.0)
            all_inputs["academic_stress_index"] = 0.4*sl + 0.4*ea - 0.2*ap

        if "screen_sleep_ratio" in selected_features:
            sd  = all_inputs.get("sleep_duration", 7.0)
            dst = all_inputs.get("daily_screen_time", 4.0)
            all_inputs["screen_sleep_ratio"] = dst / (sd if sd != 0 else 0.1)

        # DataFrame ordered exactly as the trained model expects
        df_input = pd.DataFrame([all_inputs])[selected_features]

        # Scale
        df_scaled = pd.DataFrame(
            scaler.transform(df_input),
            columns=selected_features
        )

        # Predict
        pred_class = int(model.predict(df_scaled)[0])
        pred_proba = model.predict_proba(df_scaled)[0]
        info       = STRESS_INFO[pred_class]

        # ── A: Hasil Prediksi ────────────────────────────────────
        st.markdown("### Hasil Prediksi")
        col_res, col_prob = st.columns([1, 1.4])

        with col_res:
            st.markdown(f"""
            <div class="card" style="text-align:center; padding:2rem;">
              <div style="font-size:3.5rem;">{info['emoji']}</div>
              <div style="font-family:'Space Mono',monospace; color:#8b8fad;
                   font-size:0.7rem; letter-spacing:0.15em; margin:0.6rem 0 0.4rem;">
                TINGKAT STRES
              </div>
              <span class="badge {info['badge']}">{info['label']}</span>
              <div style="margin-top:1.2rem; font-size:0.84rem; color:#8b8fad; line-height:1.6;">
                {info['desc']}
              </div>
            </div>""", unsafe_allow_html=True)

        with col_prob:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">Distribusi Probabilitas</div>', unsafe_allow_html=True)
            labels_map = {0: ("RENDAH","#43e8d8"), 1: ("SEDANG","#f9c74f"), 2: ("TINGGI","#ff6b8a")}
            for i, prob in enumerate(pred_proba):
                lbl, clr = labels_map[i]
                pct = prob * 100
                st.markdown(f"""
                <div class="prob-row">
                  <div class="prob-label">
                    <span>{lbl}</span>
                    <span style="color:{clr};font-weight:600;">{pct:.1f}%</span>
                  </div>
                  <div class="prob-bar-bg">
                    <div class="prob-bar-fill" style="width:{pct:.1f}%;background:{clr};"></div>
                  </div>
                </div>""", unsafe_allow_html=True)

            fig_pie, ax_pie = plt.subplots(figsize=(4, 3))
            fig_pie.patch.set_facecolor("#141728")
            ax_pie.set_facecolor("#141728")
            _, _, autotexts = ax_pie.pie(
                pred_proba, labels=["Rendah","Sedang","Tinggi"],
                colors=["#43e8d8","#f9c74f","#ff6b8a"], autopct="%1.1f%%",
                startangle=90,
                wedgeprops=dict(width=0.55, edgecolor="#141728", linewidth=2),
                textprops={"color":"#8b8fad","fontsize":8},
            )
            for at in autotexts:
                at.set_color("#e8eaf6"); at.set_fontsize(8)
            ax_pie.set_title("Confidence Distribution", color="#e8eaf6", fontsize=9, pad=8)
            st.pyplot(fig_pie, use_container_width=False)
            plt.close(fig_pie)
            st.markdown("</div>", unsafe_allow_html=True)

        # ── Tips ────────────────────────────────────────────────
        st.markdown("#### 💡 Rekomendasi Tindakan")
        tip_cols = st.columns(3)
        emojis = ["🌿","🧘","📋"]
        for i, (col, tip) in enumerate(zip(tip_cols, info["tips"])):
            col.markdown(f"""
            <div class="card" style="text-align:center;">
              <div style="font-size:1.5rem; margin-bottom:0.5rem;">{emojis[i]}</div>
              <div style="font-size:0.82rem; color:#8b8fad; line-height:1.5;">{tip}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<hr>", unsafe_allow_html=True)

        # ── B: Ringkasan Fitur ───────────────────────────────────
        st.markdown("### Input Anda — Ringkasan Fitur")
        summary_rows = []
        for feat in selected_features:
            icon, label = DISPLAY_NAMES.get(feat, ("📌", feat.replace("_"," ").title()))
            raw_val    = float(df_input[feat].iloc[0])
            scaled_val = float(df_scaled[feat].iloc[0])
            tipe       = "🛠 Engineered" if feat in ENGINEERED_SET else "📥 Input Langsung"
            summary_rows.append({
                "": icon,
                "Fitur":        label,
                "Tipe":         tipe,
                "Nilai Input":  round(raw_val, 4),
                "Nilai Scaled": round(scaled_val, 4),
            })
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

        st.markdown("<hr>", unsafe_allow_html=True)

        # ── C: SHAP ──────────────────────────────────────────────
        st.markdown("### 🔍 Penjelasan SHAP (Explainable AI)")
        st.markdown("""
        <div class="info-box">
        SHAP (<em>SHapley Additive exPlanations</em>) mengukur kontribusi <strong>setiap fitur</strong>
        terhadap prediksi. Nilai <span style="color:#ff6b8a;">positif</span> mendorong stres
        lebih tinggi; nilai <span style="color:#43e8d8;">negatif</span> mendorong lebih rendah.
        </div>""", unsafe_allow_html=True)

        with st.spinner("Menghitung SHAP values..."):
            try:
                explainer   = get_shap_explainer(model)
                shap_values = explainer.shap_values(df_scaled)

                # ── Normalise shap_values to (n_samples, n_features, n_classes) ──
                if isinstance(shap_values, list):
                    # list of arrays shape (n_samples, n_features) — one per class
                    sv_matrix = np.stack(shap_values, axis=-1)
                    ev_array  = np.array(explainer.expected_value)
                elif isinstance(shap_values, np.ndarray):
                    if shap_values.ndim == 3:
                        sv_matrix = shap_values
                        ev_array  = np.array(explainer.expected_value)
                    elif shap_values.ndim == 2:
                        sv_matrix = shap_values[:, :, np.newaxis]
                        ev_val    = explainer.expected_value
                        ev_array  = np.array([ev_val] if np.isscalar(ev_val) else ev_val)
                    else:
                        raise ValueError(f"Unexpected shap_values shape: {shap_values.shape}")
                else:
                    raise TypeError(f"Unexpected shap_values type: {type(shap_values)}")

                n_classes  = sv_matrix.shape[2]
                safe_class = min(pred_class, n_classes - 1)

                shap_sample    = sv_matrix[0, :, safe_class]
                ev_val         = explainer.expected_value
                if hasattr(ev_val, "__len__"):
                    expected_value = float(ev_val[safe_class])
                else:
                    expected_value = float(ev_val)

                feat_names    = selected_features
                sorted_idx    = np.argsort(np.abs(shap_sample))[::-1]
                sorted_vals   = shap_sample[sorted_idx]
                sorted_labels = [DISPLAY_NAMES.get(feat_names[i], ("",""))[1]
                                 for i in sorted_idx]
                colors_bar    = ["#ff6b8a" if v > 0 else "#43e8d8" for v in sorted_vals]

                # ── C1: Horizontal bar chart ──────────────────────
                st.markdown("#### Kontribusi Fitur — Sample Ini")
                bar_h = max(3.5, len(feat_names) * 0.55)
                fig_bar, ax_bar = dark_fig(w=9, h=bar_h)
                bars = ax_bar.barh(sorted_labels[::-1], sorted_vals[::-1],
                                   color=colors_bar[::-1], height=0.6,
                                   edgecolor="#141728", linewidth=0.5)
                ax_bar.axvline(0, color="#252840", linewidth=1)
                ax_bar.set_xlabel("SHAP Value", color="#8b8fad", fontsize=8)
                ax_bar.set_title(f"SHAP — Kelas Prediksi: {info['label']}",
                                 color="#e8eaf6", fontsize=10)
                for bar, val in zip(bars, sorted_vals[::-1]):
                    ax_bar.text(
                        val + (0.003 if val >= 0 else -0.003),
                        bar.get_y() + bar.get_height()/2,
                        f"{val:+.4f}", va="center",
                        ha="left" if val >= 0 else "right",
                        color="#e8eaf6", fontsize=7,
                    )
                plt.tight_layout()
                st.pyplot(fig_bar, use_container_width=True)
                plt.close(fig_bar)

                # ── C2: Waterfall ─────────────────────────────────
                st.markdown("#### Waterfall — Bagaimana Prediksi Terbentuk")
                fig_wf, ax_wf = dark_fig(w=10, h=4.5)
                bottoms, heights, bar_colors = [], [], []
                cur = expected_value
                for v in shap_sample[sorted_idx]:
                    if v >= 0:
                        bottoms.append(cur); heights.append(v)
                        bar_colors.append("#ff6b8a")
                    else:
                        bottoms.append(cur+v); heights.append(-v)
                        bar_colors.append("#43e8d8")
                    cur += v

                x_pos = np.arange(len(heights))
                ax_wf.bar(x_pos, heights, bottom=bottoms, color=bar_colors,
                          width=0.55, edgecolor="#141728", linewidth=0.5)
                ax_wf.axhline(expected_value, color="#f9c74f", linewidth=1,
                              linestyle="--", alpha=0.6)
                ax_wf.set_xticks(x_pos)
                ax_wf.set_xticklabels(sorted_labels, rotation=30, ha="right",
                                      fontsize=7, color="#8b8fad")
                ax_wf.set_ylabel("Model Output (log-odds)", color="#8b8fad", fontsize=8)
                ax_wf.set_title("Waterfall: Kumulatif Kontribusi SHAP",
                                color="#e8eaf6", fontsize=10)
                ax_wf.legend(
                    handles=[
                        mpatches.Patch(color="#ff6b8a", label="Mendorong STRES NAIK"),
                        mpatches.Patch(color="#43e8d8", label="Mendorong STRES TURUN"),
                        plt.Line2D([0],[0], color="#f9c74f", linestyle="--",
                                   label="Baseline E[f(X)]"),
                    ],
                    facecolor="#141728", edgecolor="#252840",
                    labelcolor="#8b8fad", fontsize=7,
                )
                plt.tight_layout()
                st.pyplot(fig_wf, use_container_width=True)
                plt.close(fig_wf)

                # ── C3: SHAP Table ────────────────────────────────
                st.markdown("#### Tabel Detail SHAP Values")
                pct33, pct66 = np.percentile(np.abs(shap_sample), [33, 66])
                shap_rows = []
                for i in sorted_idx:
                    icon, label = DISPLAY_NAMES.get(feat_names[i],
                                                    ("📌", feat_names[i]))
                    sv     = shap_sample[i]
                    dampak = ("Tinggi" if abs(sv) > pct66 else
                              "Sedang" if abs(sv) > pct33 else "Rendah")
                    shap_rows.append({
                        "": icon,
                        "Fitur":       label,
                        "Nilai Input": round(float(df_input[feat_names[i]].iloc[0]), 4),
                        "SHAP Value":  round(float(sv), 5),
                        "Arah":        "🔴 Naik" if sv > 0 else "🔵 Turun",
                        "Dampak":      dampak,
                    })
                st.dataframe(pd.DataFrame(shap_rows),
                             use_container_width=True, hide_index=True)

                # ── C4: Narasi otomatis ───────────────────────────
                top_pos = [DISPLAY_NAMES.get(feat_names[sorted_idx[i]], ("",""))[1]
                           for i in range(len(sorted_idx))
                           if shap_sample[sorted_idx[i]] > 0][:2]
                top_neg = [DISPLAY_NAMES.get(feat_names[sorted_idx[i]], ("",""))[1]
                           for i in range(len(sorted_idx))
                           if shap_sample[sorted_idx[i]] < 0][:2]
                st.markdown(f"""
                <div class="info-box">
                📝 <strong>Interpretasi Otomatis:</strong><br>
                Faktor yang paling <span style="color:#ff6b8a;">meningkatkan</span> risiko stres:
                <strong>{", ".join(top_pos) if top_pos else "–"}</strong>.<br>
                Faktor yang paling <span style="color:#43e8d8;">menurunkan</span> risiko stres:
                <strong>{", ".join(top_neg) if top_neg else "–"}</strong>.
                </div>""", unsafe_allow_html=True)

            except Exception as e:
                st.error(f"SHAP error: {e}")
                st.exception(e)


# ════════════════════════════════════════════════════════════════
# TAB 2 — TENTANG SISTEM
# ════════════════════════════════════════════════════════════════
with tab_about:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div class="card">
          <div class="card-title">Tentang Sistem</div>
          <p style="font-size:0.88rem; color:#8b8fad; line-height:1.7;">
            <strong style="color:#e8eaf6;">StressScope</strong> adalah aplikasi prediksi tingkat stres
            mahasiswa yang dibangun menggunakan kerangka kerja <strong style="color:#6c63ff;">CRISP-DM</strong>
            dan tiga algoritma: <em>Random Forest, XGBoost, LightGBM</em>.
          </p>
          <p style="font-size:0.88rem; color:#8b8fad; line-height:1.7;">
            Model aktif: <strong style="color:#6c63ff;">XGBoost</strong> — dipilih berdasarkan
            Accuracy, Macro F1, dan ROC-AUC pada evaluasi 5-fold cross-validation.
            Explainability via <strong style="color:#6c63ff;">SHAP TreeExplainer</strong>.
          </p>
        </div>
        <div class="card">
          <div class="card-title">Pipeline CRISP-DM</div>
          <div style="font-size:0.83rem; color:#8b8fad; line-height:2.2;">
            1️⃣ <strong style="color:#e8eaf6;">Business Understanding</strong><br>
            2️⃣ <strong style="color:#e8eaf6;">Data Understanding</strong> — EDA &amp; distribusi<br>
            3️⃣ <strong style="color:#e8eaf6;">Data Preparation</strong> — Imputasi &amp; scaling<br>
            4️⃣ <strong style="color:#e8eaf6;">Modeling</strong> — RF · XGBoost · LightGBM<br>
            5️⃣ <strong style="color:#e8eaf6;">Evaluation</strong> — Accuracy · F1 · ROC-AUC<br>
            6️⃣ <strong style="color:#e8eaf6;">Deployment</strong> — Streamlit + SHAP
          </div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        n_feats = len(selected_features) if selected_features else "?"
        feat_badges = "".join(
            f'<code style="background:#0d0f1a;padding:0.15rem 0.45rem;'
            f'border-radius:4px;font-size:0.73rem;margin:2px;display:inline-block;">{f}</code> '
            for f in (selected_features or [])
        )
        st.markdown(f"""
        <div class="card">
          <div class="card-title">Fitur Model Aktif ({n_feats} fitur)</div>
          <p style="font-size:0.84rem; color:#8b8fad; line-height:1.6; margin-bottom:0.6rem;">
            Sidebar hanya menampilkan slider untuk fitur yang benar-benar ada di dataset dan
            digunakan saat pelatihan:
          </p>
          <div>{feat_badges}</div>
        </div>
        <div class="card">
          <div class="card-title">Label Kelas</div>
          <div style="font-size:0.85rem; line-height:2.4;">
            <span class="badge badge-low">0 — RENDAH</span>&nbsp;Stres minimal<br>
            <span class="badge badge-mid">1 — SEDANG</span>&nbsp;Mulai menekan<br>
            <span class="badge badge-high">2 — TINGGI</span>&nbsp;Perlu intervensi
          </div>
        </div>
        <div class="card">
          <div class="card-title">Disclaimer</div>
          <p style="font-size:0.82rem; color:#8b8fad; line-height:1.6;">
            ⚠️ Aplikasi ini bersifat <strong style="color:#f9c74f;">akademik</strong>.
            Hasil bukan diagnosis resmi. Untuk kondisi berat, konsultasikan dengan
            <strong style="color:#e8eaf6;">profesional kesehatan mental</strong>.
          </p>
        </div>
        """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
# TAB 3 — PANDUAN FITUR
# ════════════════════════════════════════════════════════════════
with tab_data:
    st.markdown("### 📋 Fitur yang Digunakan Model")
    if selected_features:
        guide_rows = []
        for feat in selected_features:
            icon, label = DISPLAY_NAMES.get(feat, ("📌", feat.replace("_"," ").title()))
            lo, hi, default, _ = SLIDER_PARAMS.get(feat, (0.0, 10.0, 5.0, 0.1))
            guide_rows.append({
                "": icon, "Fitur": label, "Kode Kolom": feat,
                "Min": lo, "Max": hi, "Default": default,
                "Tipe": "🛠 Engineered" if feat in ENGINEERED_SET else "📥 Input Langsung",
            })
        st.dataframe(pd.DataFrame(guide_rows), use_container_width=True, hide_index=True)
    else:
        st.info("Muat model terlebih dahulu untuk melihat panduan fitur.")

    st.markdown("### 📈 Eksplorasi Dataset")
    csv_path = "StressLevelDataset.csv"
    if os.path.exists(csv_path):
        df_raw = pd.read_csv(csv_path)
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(f"""
            <div class="metric-row">
              <div class="metric-tile">
                <div class="val" style="color:#6c63ff;">{len(df_raw):,}</div>
                <div class="lbl">Sampel</div>
              </div>
              <div class="metric-tile">
                <div class="val" style="color:#43e8d8;">{df_raw.shape[1]}</div>
                <div class="lbl">Kolom</div>
              </div>
              <div class="metric-tile">
                <div class="val" style="color:#ff6b8a;">{df_raw['stress_level'].nunique()}</div>
                <div class="lbl">Kelas</div>
              </div>
            </div>""", unsafe_allow_html=True)

            dist = df_raw["stress_level"].value_counts().sort_index()
            fig_d, ax_d = dark_fig(5, 3)
            ax_d.bar(["Rendah","Sedang","Tinggi"], dist.values,
                     color=["#43e8d8","#f9c74f","#ff6b8a"],
                     width=0.5, edgecolor="#141728")
            ax_d.set_title("Distribusi Kelas Stres", color="#e8eaf6", fontsize=9)
            ax_d.set_ylabel("Jumlah", color="#8b8fad", fontsize=8)
            st.pyplot(fig_d, use_container_width=True)
            plt.close(fig_d)

        with col_b:
            num_cols = df_raw.select_dtypes(include=np.number).columns.tolist()
            subset   = [c for c in
                        ["anxiety_level","depression","sleep_quality",
                         "study_load","social_support","stress_level"]
                        if c in num_cols][:6]
            if subset:
                fig_h, ax_h = plt.subplots(figsize=(5, 4))
                fig_h.patch.set_facecolor("#141728")
                ax_h.set_facecolor("#141728")
                sns.heatmap(df_raw[subset].corr(), ax=ax_h, cmap="coolwarm", center=0,
                            annot=True, fmt=".2f", annot_kws={"size":7},
                            linewidths=0.5, linecolor="#252840",
                            cbar_kws={"shrink":0.75})
                ax_h.tick_params(colors="#8b8fad", labelsize=7)
                ax_h.set_title("Heatmap Korelasi (Subset)", color="#e8eaf6", fontsize=9)
                st.pyplot(fig_h, use_container_width=True)
                plt.close(fig_h)
    else:
        st.markdown("""
        <div class="info-box">
        Letakkan <code>StressLevelDataset.csv</code> di folder yang sama dengan <code>app.py</code>
        untuk melihat eksplorasi dataset.
        </div>""", unsafe_allow_html=True)


# ── Footer ────────────────────────────────────────────────────
st.markdown("""<br>
<div style="text-align:center;padding:1.5rem;border-top:1px solid #252840;margin-top:2rem;">
  <span style="font-family:'Space Mono',monospace;font-size:0.7rem;color:#3d4070;
       letter-spacing:0.12em;">
    STRESSSCOPE · MACHINE LEARNING · CRISP-DM · SHAP EXPLAINABILITY
  </span>
</div>""", unsafe_allow_html=True)