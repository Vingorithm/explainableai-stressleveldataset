# ============================================================
# app.py — Prediksi Tingkat Stres Mahasiswa
# Framework  : Streamlit  |  Model : XGBoost + SHAP
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
    --bg:      #0d0f1a;  --surface: #141728;  --border:  #252840;
    --accent1: #6c63ff;  --accent2: #ff6b8a;  --accent3: #43e8d8;
    --text:    #e8eaf6;  --muted:   #8b8fad;
    --low:     #43e8d8;  --mid:     #f9c74f;   --high:    #ff6b8a;
  }
  html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg);
    color: var(--text);
  }
  .hero {
    background: linear-gradient(135deg, #1a1d35 0%, #0d0f1a 60%, #1a1230 100%);
    border: 1px solid var(--border); border-radius: 16px;
    padding: 2.5rem 3rem; margin-bottom: 2rem;
    position: relative; overflow: hidden;
  }
  .hero::before {
    content:''; position:absolute; top:-60px; right:-60px;
    width:220px; height:220px;
    background:radial-gradient(circle,rgba(108,99,255,.25) 0%,transparent 70%);
    border-radius:50%;
  }
  .hero::after {
    content:''; position:absolute; bottom:-40px; left:40px;
    width:140px; height:140px;
    background:radial-gradient(circle,rgba(255,107,138,.18) 0%,transparent 70%);
    border-radius:50%;
  }
  .hero-tag {
    font-family:'Space Mono',monospace; font-size:.7rem;
    letter-spacing:.18em; color:var(--accent3);
    text-transform:uppercase; margin-bottom:.5rem;
  }
  .hero h1 {
    font-family:'Space Mono',monospace; font-size:2.4rem; font-weight:700;
    background:linear-gradient(90deg,#6c63ff,#ff6b8a);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
    margin:0 0 .6rem 0;
  }
  .hero p { color:var(--muted); font-size:1rem; max-width:640px; line-height:1.65; margin:0; }

  .card {
    background:var(--surface); border:1px solid var(--border);
    border-radius:14px; padding:1.4rem 1.6rem; margin-bottom:1.2rem;
  }
  .card-title {
    font-family:'Space Mono',monospace; font-size:.75rem;
    letter-spacing:.12em; color:var(--accent1);
    text-transform:uppercase; margin-bottom:.8rem;
  }
  .badge {
    display:inline-block; padding:.35rem 1.1rem; border-radius:999px;
    font-family:'Space Mono',monospace; font-weight:700;
    font-size:.95rem; letter-spacing:.06em;
  }
  .badge-low  { background:rgba(67,232,216,.18); color:var(--low);  border:1px solid var(--low);  }
  .badge-mid  { background:rgba(249,199,79,.18);  color:var(--mid);  border:1px solid var(--mid);  }
  .badge-high { background:rgba(255,107,138,.18); color:var(--high); border:1px solid var(--high); }

  .metric-row { display:flex; gap:1rem; margin-bottom:1.2rem; }
  .metric-tile {
    flex:1; background:var(--surface); border:1px solid var(--border);
    border-radius:12px; padding:1.1rem 1.3rem; text-align:center;
  }
  .metric-tile .val { font-family:'Space Mono',monospace; font-size:1.75rem; font-weight:700; }
  .metric-tile .lbl { font-size:.72rem; color:var(--muted); margin-top:.2rem; text-transform:uppercase; }

  .prob-row { margin-bottom:.6rem; }
  .prob-label { font-size:.82rem; color:var(--muted); margin-bottom:.2rem; display:flex; justify-content:space-between; }
  .prob-bar-bg { background:var(--border); border-radius:999px; height:10px; overflow:hidden; }
  .prob-bar-fill { height:100%; border-radius:999px; }

  [data-testid="stSidebar"] { background:var(--surface)!important; border-right:1px solid var(--border)!important; }
  [data-testid="stSidebar"] * { color:var(--text)!important; }
  [data-testid="stExpander"] { background:var(--surface); border:1px solid var(--border); border-radius:10px; }

  .stTabs [data-baseweb="tab-list"] { gap:.5rem; background:transparent; border-bottom:1px solid var(--border); }
  .stTabs [data-baseweb="tab"] {
    background:transparent!important; border-radius:8px 8px 0 0!important;
    color:var(--muted)!important; font-family:'Space Mono',monospace!important;
    font-size:.78rem!important; border:none!important; padding:.5rem 1.2rem!important;
  }
  .stTabs [aria-selected="true"] {
    background:var(--surface)!important; color:var(--accent1)!important;
    border:1px solid var(--border)!important; border-bottom:1px solid var(--surface)!important;
  }

  hr { border-color:var(--border)!important; }
  ::-webkit-scrollbar { width:6px; }
  ::-webkit-scrollbar-track { background:var(--bg); }
  ::-webkit-scrollbar-thumb { background:var(--border); border-radius:3px; }
  #MainMenu, footer, header { visibility:hidden; }

  .stButton>button {
    background:linear-gradient(135deg,var(--accent1),var(--accent2))!important;
    color:white!important; border:none!important; border-radius:10px!important;
    font-family:'Space Mono',monospace!important; font-size:.85rem!important;
    padding:.6rem 1.6rem!important; width:100%;
  }
  .info-box {
    background:rgba(108,99,255,.08); border-left:3px solid var(--accent1);
    border-radius:0 8px 8px 0; padding:.8rem 1rem;
    font-size:.88rem; color:var(--muted); margin-bottom:1rem;
  }
  .warn-box {
    background:rgba(255,107,138,.08); border-left:3px solid var(--accent2);
    border-radius:0 8px 8px 0; padding:.8rem 1rem;
    font-size:.88rem; color:var(--muted); margin-bottom:1rem;
  }
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
# SHAP + XGBoost 2.x COMPATIBILITY
#
# XGBoost >= 2.x stores base_score as a per-class list in
# save_config() output, e.g. "[1.93E-2,-2.51E-2,5.83E-3]".
# SHAP 0.44.x calls float() on that string and crashes.
#
# Fix: patch save_config() on the BOOSTER INSTANCE right before
# passing it to TreeExplainer. No class-level patching (which
# causes infinite recursion). No file round-trips.
# ════════════════════════════════════════════════════════════════
def _fix_booster_base_score(booster):
    """
    Replace booster.save_config with a version that converts a
    bracketed list base_score to a plain scalar string.
    Returns the same booster with the method replaced in-place.
    """
    _orig_sc = booster.save_config          # capture the real method

    def _patched_sc():
        raw = _orig_sc()                    # call the real method (no recursion)
        try:
            cfg = json.loads(raw)
            lmp = cfg.get("learner", {}).get("learner_model_param", {})
            bs  = str(lmp.get("base_score", "0.5"))
            if "[" in bs:
                tokens = [t for t in re.split(r"[,\[\]\s]+", bs) if t]
                scalar = float(tokens[0]) if tokens else 0.5
                cfg["learner"]["learner_model_param"]["base_score"] = str(scalar)
            return json.dumps(cfg)
        except Exception:
            return raw                      # fallback: return original
    
    booster.save_config = _patched_sc       # instance-level replacement only
    return booster


# ════════════════════════════════════════════════════════════════
# LOAD ARTIFACTS
# Reads selected_features.pkl to know what the model was trained
# on, then builds sidebar sliders ONLY for those features.
# ════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_artifacts():
    required = ["xgb_model.pkl", "scaler.pkl", "label_encoder.pkl", "selected_features.pkl"]
    missing  = [f for f in required if not os.path.exists(f)]
    if missing:
        return None, None, None, None, missing
    return (
        joblib.load("xgb_model.pkl"),
        joblib.load("scaler.pkl"),
        joblib.load("label_encoder.pkl"),
        joblib.load("selected_features.pkl"),
        [],
    )


# ── Dark matplotlib helper ────────────────────────────────────
def dark_fig(w=9, h=4):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor("#141728")
    ax.set_facecolor("#141728")
    for sp in ax.spines.values():
        sp.set_edgecolor("#252840")
    ax.tick_params(colors="#8b8fad", labelsize=8)
    ax.xaxis.label.set_color("#8b8fad")
    ax.yaxis.label.set_color("#8b8fad")
    ax.title.set_color("#e8eaf6")
    return fig, ax


# ════════════════════════════════════════════════════════════════
# FEATURE METADATA
# Matches the new notebook's selected_features exactly.
# Engineered features are computed automatically from base inputs.
# ════════════════════════════════════════════════════════════════

# (icon, display label)
DISPLAY_NAMES = {
    # Daily Activities
    "sleep_quality":                ("😴", "Kualitas Tidur"),
    "headache":                     ("🤕", "Sakit Kepala"),
    "blood_pressure":               ("❤️",  "Tekanan Darah"),
    "breathing_problem":            ("🫁", "Masalah Pernapasan"),
    "extracurricular_activities":   ("🎨", "Aktivitas Ekskul"),
    "basic_needs":                  ("🍽️",  "Kebutuhan Dasar"),
    "noise_level":                  ("🔊", "Tingkat Kebisingan"),
    "living_conditions":            ("🏠", "Kondisi Tempat Tinggal"),
    # Academic Factors
    "study_load":                   ("📚", "Beban Belajar"),
    "academic_performance":         ("🎓", "Performa Akademik"),
    "teacher_student_relationship": ("🤝", "Hub. Guru–Murid"),
    "future_career_concerns":       ("🔮", "Kekhawatiran Karir"),
    "peer_pressure":                ("👫", "Tekanan Teman Sebaya"),
    # Engineered
    "academic_stress_index":        ("🧮", "Academic Stress Index"),
    "daily_life_stress_index":      ("📊", "Daily Life Stress Index"),
    # Extra (in case model includes them)
    "anxiety_level":                ("😟", "Tingkat Kecemasan"),
    "self_esteem":                  ("💪", "Self-Esteem"),
    "depression":                   ("💙", "Depresi"),
    "social_support":               ("👥", "Dukungan Sosial"),
    "mental_health_history":        ("🧠", "Riwayat Kes. Mental"),
    "bullying":                     ("⚠️",  "Bullying"),
    "safety":                       ("🛡️",  "Rasa Aman"),
}

# (min, max, default, step)
SLIDER_PARAMS = {
    "sleep_quality":                (0.0,  5.0, 3.0, 1.0),
    "headache":                     (0.0,  5.0, 2.0, 1.0),
    "blood_pressure":               (1.0,  3.0, 2.0, 1.0),
    "breathing_problem":            (0.0,  5.0, 2.0, 1.0),
    "extracurricular_activities":   (0.0,  5.0, 2.0, 1.0),
    "basic_needs":                  (0.0,  5.0, 3.0, 1.0),
    "noise_level":                  (0.0,  5.0, 2.0, 1.0),
    "living_conditions":            (0.0,  5.0, 3.0, 1.0),
    "study_load":                   (0.0, 10.0, 5.0, 1.0),
    "academic_performance":         (0.0, 10.0, 7.0, 1.0),
    "teacher_student_relationship": (0.0, 10.0, 5.0, 1.0),
    "future_career_concerns":       (0.0, 10.0, 5.0, 1.0),
    "peer_pressure":                (0.0,  5.0, 2.0, 1.0),
    "anxiety_level":                (0.0, 21.0,10.0, 1.0),
    "self_esteem":                  (0.0, 30.0,15.0, 1.0),
    "depression":                   (0.0, 27.0,10.0, 1.0),
    "social_support":               (0.0, 10.0, 5.0, 1.0),
    "mental_health_history":        (0.0,  1.0, 0.0, 1.0),
    "bullying":                     (0.0,  5.0, 1.0, 1.0),
    "safety":                       (0.0,  5.0, 3.0, 1.0),
}

# Features computed automatically — no slider needed
ENGINEERED_SET = {"academic_stress_index", "daily_life_stress_index"}

# Group labels for sidebar sections
GROUPS = {
    "🌙 Aktivitas Harian": [
        "sleep_quality", "headache", "blood_pressure", "breathing_problem",
        "extracurricular_activities", "basic_needs", "noise_level", "living_conditions",
    ],
    "📚 Faktor Akademik": [
        "study_load", "academic_performance", "teacher_student_relationship",
        "future_career_concerns", "peer_pressure",
    ],
}

STRESS_INFO = {
    0: {
        "label": "RENDAH", "badge": "badge-low", "color": "#43e8d8", "emoji": "😊",
        "desc": "Tingkat stres Anda tergolong <strong>rendah</strong>. Kondisi psikologis, "
                "sosial, dan akademik Anda berada pada zona sehat. Pertahankan kebiasaan baik ini.",
        "tips": ["Pertahankan kualitas tidur yang baik setiap malam",
                 "Jaga keseimbangan antara belajar dan istirahat",
                 "Terus bangun dukungan sosial yang positif"],
    },
    1: {
        "label": "SEDANG", "badge": "badge-mid", "color": "#f9c74f", "emoji": "😐",
        "desc": "Tingkat stres Anda berada di level <strong>sedang</strong>. Beberapa faktor "
                "mulai menekan kondisi Anda. Segera lakukan intervensi ringan.",
        "tips": ["Evaluasi beban studi dan buat prioritas tugas",
                 "Coba teknik relaksasi seperti meditasi atau journaling",
                 "Bicarakan kekhawatiran kepada teman atau konselor"],
    },
    2: {
        "label": "TINGGI", "badge": "badge-high", "color": "#ff6b8a", "emoji": "😟",
        "desc": "Tingkat stres Anda tergolong <strong>tinggi</strong>. Kondisi ini dapat "
                "berdampak negatif pada kesehatan. Sangat disarankan mencari bantuan profesional.",
        "tips": ["Segera konsultasikan dengan psikolog atau konselor kampus",
                 "Kurangi beban berlebih dan delegasikan tugas bila memungkinkan",
                 "Prioritaskan tidur, olahraga ringan, dan pola makan sehat"],
    },
}


# ════════════════════════════════════════════════════════════════
# LOAD MODEL
# ════════════════════════════════════════════════════════════════
model, scaler, le, selected_features, missing = load_artifacts()


# ════════════════════════════════════════════════════════════════
# SIDEBAR — built dynamically from selected_features.pkl
# ════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="font-family:'Space Mono',monospace;font-size:1.1rem;font-weight:700;
         background:linear-gradient(90deg,#6c63ff,#ff6b8a);
         -webkit-background-clip:text;-webkit-text-fill-color:transparent;
         margin-bottom:.3rem;">🧠 StressScope</div>
    <div style="font-size:.72rem;color:#8b8fad;letter-spacing:.1em;
         text-transform:uppercase;margin-bottom:1.2rem;">Input Parameter Mahasiswa</div>
    """, unsafe_allow_html=True)

    inputs = {}

    if selected_features:
        slider_feats = [f for f in selected_features if f not in ENGINEERED_SET]

        # Group features by category for cleaner sidebar
        rendered = set()
        for group_label, group_feats in GROUPS.items():
            group_in_model = [f for f in group_feats if f in slider_feats]
            if not group_in_model:
                continue
            st.markdown(f"""
            <div style="font-size:.68rem;color:#8b8fad;letter-spacing:.1em;
                 text-transform:uppercase;margin:.8rem 0 .3rem;">
            {group_label}
            </div>""", unsafe_allow_html=True)
            for feat in group_in_model:
                icon, label = DISPLAY_NAMES.get(feat, ("📌", feat.replace("_"," ").title()))
                lo, hi, default, step = SLIDER_PARAMS.get(feat, (0.0, 10.0, 5.0, 1.0))
                with st.expander(f"{icon}  {label}", expanded=False):
                    inputs[feat] = st.slider(
                        label, min_value=lo, max_value=hi,
                        value=default, step=step,
                        label_visibility="collapsed", key=f"s_{feat}"
                    )
                rendered.add(feat)

        # Any remaining features not in groups
        remaining = [f for f in slider_feats if f not in rendered]
        if remaining:
            st.markdown("""
            <div style="font-size:.68rem;color:#8b8fad;letter-spacing:.1em;
                 text-transform:uppercase;margin:.8rem 0 .3rem;">
            📌 Lainnya
            </div>""", unsafe_allow_html=True)
            for feat in remaining:
                icon, label = DISPLAY_NAMES.get(feat, ("📌", feat.replace("_"," ").title()))
                lo, hi, default, step = SLIDER_PARAMS.get(feat, (0.0, 10.0, 5.0, 1.0))
                with st.expander(f"{icon}  {label}", expanded=False):
                    inputs[feat] = st.slider(
                        label, min_value=lo, max_value=hi,
                        value=default, step=step,
                        label_visibility="collapsed", key=f"s_{feat}"
                    )

        # Preview auto-computed engineered features
        if any(f in selected_features for f in ENGINEERED_SET):
            st.markdown("""
            <div style="font-size:.68rem;color:#8b8fad;letter-spacing:.1em;
                 text-transform:uppercase;margin:.8rem 0 .3rem;">
            🛠 Dihitung Otomatis
            </div>""", unsafe_allow_html=True)
            if "academic_stress_index" in selected_features:
                sl  = inputs.get("study_load", 5.0)
                fc  = inputs.get("future_career_concerns", 5.0)
                pp  = inputs.get("peer_pressure", 2.0)
                asi = (sl + fc + pp) / 3.0
                st.caption(f"🧮 Academic Stress Index: **{asi:.3f}**")
            if "daily_life_stress_index" in selected_features:
                nl  = inputs.get("noise_level", 2.0)
                lc  = inputs.get("living_conditions", 3.0)
                bn  = inputs.get("basic_needs", 3.0)
                dlsi = (nl + lc + bn) / 3.0
                st.caption(f"📊 Daily Life Stress Index: **{dlsi:.3f}**")

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
        # ── Compute engineered features (same formula as notebook) ──
        all_inputs = dict(inputs)

        if "academic_stress_index" in selected_features:
            sl  = all_inputs.get("study_load", 5.0)
            fc  = all_inputs.get("future_career_concerns", 5.0)
            pp  = all_inputs.get("peer_pressure", 2.0)
            all_inputs["academic_stress_index"] = (sl + fc + pp) / 3.0

        if "daily_life_stress_index" in selected_features:
            nl  = all_inputs.get("noise_level", 2.0)
            lc  = all_inputs.get("living_conditions", 3.0)
            bn  = all_inputs.get("basic_needs", 3.0)
            all_inputs["daily_life_stress_index"] = (nl + lc + bn) / 3.0

        # Ordered exactly as model expects
        df_input  = pd.DataFrame([all_inputs])[selected_features]
        df_scaled = pd.DataFrame(scaler.transform(df_input), columns=selected_features)

        pred_class = int(model.predict(df_scaled)[0])
        pred_proba = model.predict_proba(df_scaled)[0]
        info       = STRESS_INFO[pred_class]

        # ── A: Hasil Prediksi ────────────────────────────────────
        st.markdown("### Hasil Prediksi")
        col_res, col_prob = st.columns([1, 1.4])

        with col_res:
            st.markdown(f"""
            <div class="card" style="text-align:center;padding:2rem;">
              <div style="font-size:3.5rem;">{info['emoji']}</div>
              <div style="font-family:'Space Mono',monospace;color:#8b8fad;
                   font-size:.7rem;letter-spacing:.15em;margin:.6rem 0 .4rem;">
                TINGKAT STRES
              </div>
              <span class="badge {info['badge']}">{info['label']}</span>
              <div style="margin-top:1.2rem;font-size:.84rem;color:#8b8fad;line-height:1.6;">
                {info['desc']}
              </div>
            </div>""", unsafe_allow_html=True)

        with col_prob:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">Distribusi Probabilitas</div>',
                        unsafe_allow_html=True)
            prob_meta = {0: ("RENDAH","#43e8d8"), 1: ("SEDANG","#f9c74f"), 2: ("TINGGI","#ff6b8a")}
            for i, prob in enumerate(pred_proba):
                lbl, clr = prob_meta[i]
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
            fig_pie.patch.set_facecolor("#141728"); ax_pie.set_facecolor("#141728")
            _, _, autotexts = ax_pie.pie(
                pred_proba, labels=["Rendah","Sedang","Tinggi"],
                colors=["#43e8d8","#f9c74f","#ff6b8a"], autopct="%1.1f%%", startangle=90,
                wedgeprops=dict(width=0.55, edgecolor="#141728", linewidth=2),
                textprops={"color":"#8b8fad","fontsize":8},
            )
            for at in autotexts: at.set_color("#e8eaf6"); at.set_fontsize(8)
            ax_pie.set_title("Confidence Distribution", color="#e8eaf6", fontsize=9, pad=8)
            st.pyplot(fig_pie, use_container_width=False); plt.close(fig_pie)
            st.markdown("</div>", unsafe_allow_html=True)

        # ── Tips ────────────────────────────────────────────────
        st.markdown("#### 💡 Rekomendasi Tindakan")
        tip_cols = st.columns(3)
        for i, (col, tip) in enumerate(zip(tip_cols, info["tips"])):
            col.markdown(f"""
            <div class="card" style="text-align:center;">
              <div style="font-size:1.5rem;margin-bottom:.5rem;">{'🌿🧘📋'[i]}</div>
              <div style="font-size:.82rem;color:#8b8fad;line-height:1.5;">{tip}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<hr>", unsafe_allow_html=True)

        # ── B: Ringkasan Fitur ───────────────────────────────────
        st.markdown("### Input Anda — Ringkasan Fitur")
        rows = []
        for feat in selected_features:
            icon, label = DISPLAY_NAMES.get(feat, ("📌", feat.replace("_"," ").title()))
            rows.append({
                "": icon,
                "Fitur":        label,
                "Tipe":         "🛠 Engineered" if feat in ENGINEERED_SET else "📥 Input",
                "Nilai Input":  round(float(df_input[feat].iloc[0]), 4),
                "Nilai Scaled": round(float(df_scaled[feat].iloc[0]), 4),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        st.markdown("<hr>", unsafe_allow_html=True)

        # ── C: SHAP ──────────────────────────────────────────────
        st.markdown("### 🔍 Penjelasan SHAP (Explainable AI)")
        st.markdown("""
        <div class="info-box">
        SHAP (<em>SHapley Additive exPlanations</em>) mengukur kontribusi <strong>setiap fitur</strong>
        terhadap prediksi. Nilai <span style="color:#ff6b8a;">positif (merah)</span> mendorong stres
        lebih tinggi; nilai <span style="color:#43e8d8;">negatif (biru)</span> mendorong lebih rendah.
        </div>""", unsafe_allow_html=True)

        with st.spinner("Menghitung SHAP values..."):
            try:
                # Patch save_config() on the booster instance directly.
                # This fixes XGBoost 2.x storing base_score as a bracketed
                # list string that SHAP cannot parse with float().
                booster   = _fix_booster_base_score(model.get_booster())
                explainer = shap.TreeExplainer(booster)
                shap_raw  = explainer.shap_values(df_scaled)

                # ── Normalise to (n_samples, n_features, n_classes) ──
                if isinstance(shap_raw, list):
                    sv = np.stack(shap_raw, axis=-1)     # list of (n,f) → (n,f,c)
                    ev = np.array(explainer.expected_value)
                elif isinstance(shap_raw, np.ndarray):
                    if shap_raw.ndim == 3:
                        sv = shap_raw
                        ev = np.array(explainer.expected_value)
                    elif shap_raw.ndim == 2:
                        sv = shap_raw[:, :, np.newaxis]
                        ev = np.array([explainer.expected_value])
                    else:
                        raise ValueError(f"Unexpected shap ndim={shap_raw.ndim}")
                else:
                    raise TypeError(f"Unexpected shap type {type(shap_raw)}")

                n_cls      = sv.shape[2]
                safe_cls   = min(pred_class, n_cls - 1)
                shap_sample = sv[0, :, safe_cls]
                exp_val     = float(ev[safe_cls]) if ev.ndim > 0 else float(ev)

                feat_names    = selected_features
                sorted_idx    = np.argsort(np.abs(shap_sample))[::-1]
                sorted_sv     = shap_sample[sorted_idx]
                sorted_labels = [DISPLAY_NAMES.get(feat_names[i], ("",""))[1]
                                 for i in sorted_idx]
                colors_bar    = ["#ff6b8a" if v > 0 else "#43e8d8" for v in sorted_sv]

                # ── C1: Horizontal bar ────────────────────────────
                st.markdown("#### Kontribusi Fitur — Sample Ini")
                bar_h = max(3.5, len(feat_names) * 0.55)
                fig_bar, ax_bar = dark_fig(w=9, h=bar_h)
                bars = ax_bar.barh(
                    sorted_labels[::-1], sorted_sv[::-1],
                    color=colors_bar[::-1], height=0.6,
                    edgecolor="#141728", linewidth=0.5,
                )
                ax_bar.axvline(0, color="#252840", linewidth=1)
                ax_bar.set_xlabel("SHAP Value (kontribusi terhadap output model)",
                                  color="#8b8fad", fontsize=8)
                ax_bar.set_title(f"SHAP — Kelas Prediksi: {info['label']}",
                                 color="#e8eaf6", fontsize=10)
                for bar, val in zip(bars, sorted_sv[::-1]):
                    offset = 0.003 if val >= 0 else -0.003
                    ax_bar.text(
                        val + offset, bar.get_y() + bar.get_height()/2,
                        f"{val:+.4f}", va="center",
                        ha="left" if val >= 0 else "right",
                        color="#e8eaf6", fontsize=7,
                    )
                plt.tight_layout()
                st.pyplot(fig_bar, use_container_width=True); plt.close(fig_bar)

                # ── C2: Waterfall ─────────────────────────────────
                st.markdown("#### Waterfall — Bagaimana Prediksi Terbentuk")
                fig_wf, ax_wf = dark_fig(w=10, h=4.5)
                bottoms, heights, wf_colors = [], [], []
                cur = exp_val
                for v in sorted_sv:
                    if v >= 0:
                        bottoms.append(cur); heights.append(v)
                        wf_colors.append("#ff6b8a")
                    else:
                        bottoms.append(cur+v); heights.append(-v)
                        wf_colors.append("#43e8d8")
                    cur += v
                x_pos = np.arange(len(heights))
                ax_wf.bar(x_pos, heights, bottom=bottoms, color=wf_colors,
                          width=0.55, edgecolor="#141728", linewidth=0.5)
                ax_wf.axhline(exp_val, color="#f9c74f", linewidth=1,
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
                                   label=f"Baseline E[f(X)] = {exp_val:.3f}"),
                    ],
                    facecolor="#141728", edgecolor="#252840",
                    labelcolor="#8b8fad", fontsize=7,
                )
                plt.tight_layout()
                st.pyplot(fig_wf, use_container_width=True); plt.close(fig_wf)

                # ── C3: SHAP Table ────────────────────────────────
                st.markdown("#### Tabel Detail SHAP Values")
                pct33, pct66 = np.percentile(np.abs(shap_sample), [33, 66])
                shap_rows = []
                for i in sorted_idx:
                    icon, label = DISPLAY_NAMES.get(feat_names[i],
                                                    ("📌", feat_names[i]))
                    sv_i   = shap_sample[i]
                    dampak = ("🔥 Tinggi" if abs(sv_i) > pct66
                              else "⚡ Sedang" if abs(sv_i) > pct33
                              else "💤 Rendah")
                    shap_rows.append({
                        "": icon,
                        "Fitur":       label,
                        "Tipe":        "🛠 Eng." if feat_names[i] in ENGINEERED_SET else "📥 Input",
                        "Nilai Input": round(float(df_input[feat_names[i]].iloc[0]), 3),
                        "SHAP Value":  round(float(sv_i), 5),
                        "Arah":        "🔴 Naik" if sv_i > 0 else "🔵 Turun",
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
                Faktor yang paling <span style="color:#ff6b8a;">meningkatkan</span> risiko stres Anda:
                <strong>{", ".join(top_pos) if top_pos else "–"}</strong>.<br>
                Faktor yang paling <span style="color:#43e8d8;">menurunkan</span> risiko stres Anda:
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
          <p style="font-size:.88rem;color:#8b8fad;line-height:1.7;">
            <strong style="color:#e8eaf6;">StressScope</strong> adalah aplikasi prediksi tingkat stres
            mahasiswa berbasis kerangka kerja <strong style="color:#6c63ff;">CRISP-DM</strong> dengan
            tiga algoritma: <em>Random Forest, XGBoost, LightGBM</em>.
          </p>
          <p style="font-size:.88rem;color:#8b8fad;line-height:1.7;">
            Model aktif: <strong style="color:#6c63ff;">XGBoost</strong> — dipilih berdasarkan
            Accuracy, Macro F1, dan ROC-AUC pada 5-fold cross-validation.
            Explainability menggunakan <strong style="color:#6c63ff;">SHAP TreeExplainer</strong>.
          </p>
        </div>
        <div class="card">
          <div class="card-title">Pipeline CRISP-DM</div>
          <div style="font-size:.83rem;color:#8b8fad;line-height:2.2;">
            1️⃣ <strong style="color:#e8eaf6;">Business Understanding</strong><br>
            2️⃣ <strong style="color:#e8eaf6;">Data Understanding</strong> — EDA &amp; distribusi<br>
            3️⃣ <strong style="color:#e8eaf6;">Data Preparation</strong> — Imputasi, engineering, scaling<br>
            4️⃣ <strong style="color:#e8eaf6;">Modeling</strong> — RF · XGBoost · LightGBM + CV<br>
            5️⃣ <strong style="color:#e8eaf6;">Evaluation</strong> — Accuracy · F1 · ROC-AUC<br>
            6️⃣ <strong style="color:#e8eaf6;">Deployment</strong> — Streamlit + SHAP
          </div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        n_feats = len(selected_features) if selected_features else "?"
        feat_badges = "".join(
            f'<code style="background:#0d0f1a;padding:.15rem .45rem;border-radius:4px;'
            f'font-size:.73rem;margin:2px;display:inline-block;">{f}</code>'
            for f in (selected_features or [])
        )
        st.markdown(f"""
        <div class="card">
          <div class="card-title">Feature Engineering</div>
          <div style="background:#0d0f1a;border-radius:8px;padding:1rem;
               font-family:'Space Mono',monospace;font-size:.75rem;
               color:#43e8d8;line-height:1.9;">
            academic_stress_index =<br>
            &nbsp;&nbsp;(study_load + future_career_concerns + peer_pressure) / 3<br><br>
            daily_life_stress_index =<br>
            &nbsp;&nbsp;(noise_level + living_conditions + basic_needs) / 3
          </div>
        </div>
        <div class="card">
          <div class="card-title">Fitur Model Aktif ({n_feats} fitur)</div>
          <div style="line-height:2;">{feat_badges}</div>
        </div>
        <div class="card">
          <div class="card-title">Label Kelas</div>
          <div style="font-size:.85rem;line-height:2.4;">
            <span class="badge badge-low">0 — RENDAH</span>&nbsp;Stres minimal<br>
            <span class="badge badge-mid">1 — SEDANG</span>&nbsp;Mulai menekan<br>
            <span class="badge badge-high">2 — TINGGI</span>&nbsp;Perlu intervensi
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="warn-box">
    ⚠️ <strong>Disclaimer:</strong> Aplikasi ini bersifat <strong>akademik dan informatif</strong>.
    Hasil prediksi bukan diagnosis medis atau psikologis resmi. Jika mengalami tekanan berat,
    konsultasikan dengan <strong>profesional kesehatan mental</strong> atau konselor kampus.
    </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
# TAB 3 — PANDUAN FITUR
# ════════════════════════════════════════════════════════════════
with tab_data:
    st.markdown("### 📋 Fitur yang Digunakan Model")
    if selected_features:
        guide = []
        for feat in selected_features:
            icon, label = DISPLAY_NAMES.get(feat, ("📌", feat.replace("_"," ").title()))
            lo, hi, default, _ = SLIDER_PARAMS.get(feat, (0.0, 10.0, 5.0, 1.0))
            guide.append({
                "": icon, "Fitur": label, "Kode Kolom": feat,
                "Min": lo, "Max": hi, "Default": default,
                "Tipe": "🛠 Engineered" if feat in ENGINEERED_SET else "📥 Input",
            })
        st.dataframe(pd.DataFrame(guide), use_container_width=True, hide_index=True)

    st.markdown("### 🧮 Rumus Feature Engineering")
    st.markdown("""
    <div class="card">
      <div class="card-title">Academic Stress Index</div>
      <p style="font-size:.85rem;color:#8b8fad;line-height:1.6;">
        Rata-rata dari <strong>Beban Belajar</strong>, <strong>Kekhawatiran Karir</strong>,
        dan <strong>Tekanan Teman Sebaya</strong>. Semakin tinggi ketiga faktor ini,
        semakin tinggi indeks stres akademik.
      </p>
      <div style="background:#0d0f1a;border-radius:8px;padding:.8rem 1rem;
           font-family:'Space Mono',monospace;font-size:.75rem;color:#43e8d8;">
        academic_stress_index = (study_load + future_career_concerns + peer_pressure) / 3
      </div>
    </div>
    <div class="card">
      <div class="card-title">Daily Life Stress Index</div>
      <p style="font-size:.85rem;color:#8b8fad;line-height:1.6;">
        Rata-rata dari <strong>Tingkat Kebisingan</strong>, <strong>Kondisi Tempat Tinggal</strong>,
        dan <strong>Kebutuhan Dasar</strong>. Mengukur tekanan dari lingkungan fisik sehari-hari.
      </p>
      <div style="background:#0d0f1a;border-radius:8px;padding:.8rem 1rem;
           font-family:'Space Mono',monospace;font-size:.75rem;color:#43e8d8;">
        daily_life_stress_index = (noise_level + living_conditions + basic_needs) / 3
      </div>
    </div>
    """, unsafe_allow_html=True)

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
                     color=["#43e8d8","#f9c74f","#ff6b8a"], width=0.5, edgecolor="#141728")
            ax_d.set_title("Distribusi Kelas Stres", color="#e8eaf6", fontsize=9)
            ax_d.set_ylabel("Jumlah Sampel", color="#8b8fad", fontsize=8)
            st.pyplot(fig_d, use_container_width=True); plt.close(fig_d)

        with col_b:
            num_cols = df_raw.select_dtypes(include=np.number).columns.tolist()
            hm_cols  = [c for c in [
                "study_load","future_career_concerns","peer_pressure",
                "noise_level","living_conditions","basic_needs","stress_level"
            ] if c in num_cols][:7]
            if hm_cols:
                fig_h, ax_h = plt.subplots(figsize=(5.5, 4.5))
                fig_h.patch.set_facecolor("#141728"); ax_h.set_facecolor("#141728")
                sns.heatmap(df_raw[hm_cols].corr(), ax=ax_h, cmap="coolwarm", center=0,
                            annot=True, fmt=".2f", annot_kws={"size":7},
                            linewidths=0.5, linecolor="#252840",
                            cbar_kws={"shrink":0.75})
                ax_h.tick_params(colors="#8b8fad", labelsize=7)
                ax_h.set_title("Korelasi Fitur Utama", color="#e8eaf6", fontsize=9)
                st.pyplot(fig_h, use_container_width=True); plt.close(fig_h)
    else:
        st.markdown("""
        <div class="info-box">
        Letakkan <code>StressLevelDataset.csv</code> di folder yang sama dengan <code>app.py</code>
        untuk melihat eksplorasi dataset.
        </div>""", unsafe_allow_html=True)


# ── Footer ────────────────────────────────────────────────────
st.markdown("""<br>
<div style="text-align:center;padding:1.5rem;border-top:1px solid #252840;margin-top:2rem;">
  <span style="font-family:'Space Mono',monospace;font-size:.7rem;
       color:#3d4070;letter-spacing:.12em;">
    STRESSSCOPE · MACHINE LEARNING · CRISP-DM · SHAP EXPLAINABILITY
  </span>
</div>""", unsafe_allow_html=True)