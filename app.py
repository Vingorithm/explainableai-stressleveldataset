# ============================================================
# app.py — Prediksi Tingkat Stres Mahasiswa
# Framework : Streamlit | Model : XGBoost + SHAP
# ============================================================
#
# WHY THIS FIX IS NECESSARY
# ─────────────────────────
# SHAP 0.44.1 switched XGBoost model parsing from binary to ubjson
# format (PR #3345). In the ubjson code path, SHAP calls:
#
#   xgb_model.save_raw("ubj")  → bytes
#   ubjson.decode(bytes)       → dict with native Python types
#   float(data["learner"]["learner_model_param"]["base_score"])
#
# XGBoost >= 2.x stores base_score as a per-class list (e.g.
# [0.019, -0.025, 0.006] for 3-class), so float(list) → TypeError.
# The value appears as '[1.93E-2,-2.51E-2,5.83E-3]' in the error
# because ubjson sometimes returns it as a string too.
#
# Previous attempts to patch save_config() (JSON) didn't work because
# the ubjson path never calls save_config() at all.
# Class-level __init__ patches caused infinite recursion because SHAP's
# internal fallback/retry logic re-instantiates XGBTreeModelLoader.
#
# THE DEFINITIVE FIX: edit shap/explainers/_tree.py on disk at startup,
# replacing the single float() call with safe code that handles both
# list and string forms of base_score. This is a one-time idempotent
# edit applied before any SHAP class is instantiated.
# ============================================================

import os
import re
import sys
import importlib.util

# ════════════════════════════════════════════════════════════════
# STEP 1 — Patch shap/_tree.py on disk before importing shap
# This is idempotent: the sentinel string ensures it only runs once.
# ════════════════════════════════════════════════════════════════
def _patch_shap_tree_file():
    """
    Find the installed shap/_tree.py and replace the broken
    `float(base_score)` line with code that handles list/string forms.
    Safe to call multiple times — a sentinel ensures one edit only.
    """
    spec = importlib.util.find_spec("shap")
    if spec is None:
        return  # shap not installed yet; will fail later with a clear error

    tree_path = os.path.join(os.path.dirname(spec.origin),
                             "explainers", "_tree.py")
    if not os.path.exists(tree_path):
        return

    with open(tree_path, "r", encoding="utf-8") as f:
        src = f.read()

    # Sentinel: if we've already patched, do nothing
    if "_STRESSSCOPE_PATCHED_" in src:
        return

    # The exact broken line in SHAP 0.44.1 (line ~1760)
    OLD = ('                self.base_score = float('
           'xgb_params["learner_model_param"]["base_score"])')

    # Replacement: handle list, bracketed string, or plain string
    NEW = (
        '                # _STRESSSCOPE_PATCHED_ — XGBoost 2.x/3.x compat\n'
        '                _bs = xgb_params["learner_model_param"]["base_score"]\n'
        '                if isinstance(_bs, list):\n'
        '                    self.base_score = float(_bs[0])\n'
        '                elif isinstance(_bs, str) and "[" in _bs:\n'
        '                    import re as _re\n'
        '                    _toks = [t for t in _re.split(r\'[,\\[\\]\\s]+\', _bs) if t]\n'
        '                    self.base_score = float(_toks[0]) if _toks else 0.5\n'
        '                else:\n'
        '                    self.base_score = float(_bs)'
    )

    if OLD not in src:
        # Already fixed by a newer SHAP version, or line changed — skip
        return

    patched = src.replace(OLD, NEW, 1)
    try:
        with open(tree_path, "w", encoding="utf-8") as f:
            f.write(patched)
    except OSError:
        pass  # Read-only filesystem — patch won't apply but error shown below


_patch_shap_tree_file()

# ════════════════════════════════════════════════════════════════
# STEP 2 — Now import shap (with the patched file)
# ════════════════════════════════════════════════════════════════
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import joblib
import shap
import json
import warnings
warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="StressScope — Prediksi Stres Mahasiswa",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');
:root {
  --bg:#0d0f1a; --surface:#141728; --border:#252840;
  --a1:#6c63ff; --a2:#ff6b8a; --a3:#43e8d8;
  --tx:#e8eaf6; --mu:#8b8fad;
  --lo:#43e8d8; --mi:#f9c74f; --hi:#ff6b8a;
}
html,body,[class*="css"]{font-family:'DM Sans',sans-serif;background:var(--bg);color:var(--tx);}
.hero{background:linear-gradient(135deg,#1a1d35,#0d0f1a 60%,#1a1230);
  border:1px solid var(--border);border-radius:16px;padding:2.5rem 3rem;
  margin-bottom:2rem;position:relative;overflow:hidden;}
.hero::before{content:'';position:absolute;top:-60px;right:-60px;width:220px;height:220px;
  background:radial-gradient(circle,rgba(108,99,255,.25),transparent 70%);border-radius:50%;}
.hero::after{content:'';position:absolute;bottom:-40px;left:40px;width:140px;height:140px;
  background:radial-gradient(circle,rgba(255,107,138,.18),transparent 70%);border-radius:50%;}
.hero-tag{font-family:'Space Mono',monospace;font-size:.7rem;letter-spacing:.18em;
  color:var(--a3);text-transform:uppercase;margin-bottom:.5rem;}
.hero h1{font-family:'Space Mono',monospace;font-size:2.4rem;font-weight:700;
  background:linear-gradient(90deg,#6c63ff,#ff6b8a);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;margin:0 0 .6rem;}
.hero p{color:var(--mu);font-size:1rem;max-width:640px;line-height:1.65;margin:0;}
.card{background:var(--surface);border:1px solid var(--border);border-radius:14px;
  padding:1.4rem 1.6rem;margin-bottom:1.2rem;}
.ctitle{font-family:'Space Mono',monospace;font-size:.75rem;letter-spacing:.12em;
  color:var(--a1);text-transform:uppercase;margin-bottom:.8rem;}
.badge{display:inline-block;padding:.35rem 1.1rem;border-radius:999px;
  font-family:'Space Mono',monospace;font-weight:700;font-size:.95rem;}
.bl{background:rgba(67,232,216,.18);color:var(--lo);border:1px solid var(--lo);}
.bm{background:rgba(249,199,79,.18);color:var(--mi);border:1px solid var(--mi);}
.bh{background:rgba(255,107,138,.18);color:var(--hi);border:1px solid var(--hi);}
.prob-row{margin-bottom:.6rem;}
.prob-label{font-size:.82rem;color:var(--mu);margin-bottom:.2rem;display:flex;justify-content:space-between;}
.prob-bg{background:var(--border);border-radius:999px;height:10px;overflow:hidden;}
.prob-fill{height:100%;border-radius:999px;}
[data-testid="stSidebar"]{background:var(--surface)!important;border-right:1px solid var(--border)!important;}
[data-testid="stSidebar"] *{color:var(--tx)!important;}
[data-testid="stExpander"]{background:var(--surface);border:1px solid var(--border);border-radius:10px;}
.stTabs [data-baseweb="tab-list"]{gap:.5rem;background:transparent;border-bottom:1px solid var(--border);}
.stTabs [data-baseweb="tab"]{background:transparent!important;border-radius:8px 8px 0 0!important;
  color:var(--mu)!important;font-family:'Space Mono',monospace!important;
  font-size:.78rem!important;border:none!important;padding:.5rem 1.2rem!important;}
.stTabs [aria-selected="true"]{background:var(--surface)!important;color:var(--a1)!important;
  border:1px solid var(--border)!important;border-bottom:1px solid var(--surface)!important;}
hr{border-color:var(--border)!important;}
::-webkit-scrollbar{width:6px;}
::-webkit-scrollbar-track{background:var(--bg);}
::-webkit-scrollbar-thumb{background:var(--border);border-radius:3px;}
#MainMenu,footer,header{visibility:hidden;}
.stButton>button{background:linear-gradient(135deg,var(--a1),var(--a2))!important;
  color:#fff!important;border:none!important;border-radius:10px!important;
  font-family:'Space Mono',monospace!important;font-size:.85rem!important;
  padding:.6rem 1.6rem!important;width:100%;}
.ibox{background:rgba(108,99,255,.08);border-left:3px solid var(--a1);
  border-radius:0 8px 8px 0;padding:.8rem 1rem;font-size:.88rem;color:var(--mu);margin-bottom:1rem;}
.wbox{background:rgba(255,107,138,.08);border-left:3px solid var(--a2);
  border-radius:0 8px 8px 0;padding:.8rem 1rem;font-size:.88rem;color:var(--mu);margin-bottom:1rem;}
.metric-row{display:flex;gap:1rem;margin-bottom:1.2rem;}
.metric-tile{flex:1;background:var(--surface);border:1px solid var(--border);
  border-radius:12px;padding:1.1rem 1.3rem;text-align:center;}
.metric-tile .val{font-family:'Space Mono',monospace;font-size:1.75rem;font-weight:700;}
.metric-tile .lbl{font-size:.72rem;color:var(--mu);margin-top:.2rem;text-transform:uppercase;}
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
# LOAD ARTIFACTS
# ════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_artifacts():
    needed = ["xgb_model.pkl", "scaler.pkl", "label_encoder.pkl", "selected_features.pkl"]
    missing = [f for f in needed if not os.path.exists(f)]
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
# FEATURE METADATA  (15 features from notebook)
# ════════════════════════════════════════════════════════════════
DISPLAY = {
    "sleep_quality":                ("😴", "Kualitas Tidur"),
    "headache":                     ("🤕", "Sakit Kepala"),
    "blood_pressure":               ("❤️",  "Tekanan Darah"),
    "breathing_problem":            ("🫁", "Masalah Pernapasan"),
    "extracurricular_activities":   ("🎨", "Aktivitas Ekskul"),
    "basic_needs":                  ("🍽️",  "Kebutuhan Dasar"),
    "noise_level":                  ("🔊", "Tingkat Kebisingan"),
    "living_conditions":            ("🏠", "Kondisi Tempat Tinggal"),
    "study_load":                   ("📚", "Beban Belajar"),
    "academic_performance":         ("🎓", "Performa Akademik"),
    "teacher_student_relationship": ("🤝", "Hub. Guru–Murid"),
    "future_career_concerns":       ("🔮", "Kekhawatiran Karir"),
    "peer_pressure":                ("👫", "Tekanan Teman Sebaya"),
    "academic_stress_index":        ("🧮", "Academic Stress Index"),
    "daily_life_stress_index":      ("📊", "Daily Life Stress Index"),
}

SLIDERS = {
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
}

ENGINEERED = {"academic_stress_index", "daily_life_stress_index"}

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
    0: dict(
        label="RENDAH", badge="bl", color="#43e8d8", emoji="😊",
        desc="Tingkat stres Anda tergolong <strong>rendah</strong>. Kondisi fisik, "
             "akademik, dan lingkungan Anda berada pada zona sehat.",
        tips=["Pertahankan kualitas tidur yang baik setiap malam",
              "Jaga keseimbangan antara belajar dan istirahat",
              "Terus bangun dukungan sosial yang positif"],
    ),
    1: dict(
        label="SEDANG", badge="bm", color="#f9c74f", emoji="😐",
        desc="Tingkat stres Anda berada di level <strong>sedang</strong>. Beberapa "
             "faktor mulai menekan. Segera lakukan intervensi ringan.",
        tips=["Evaluasi beban studi dan buat prioritas tugas",
              "Coba teknik relaksasi seperti meditasi atau journaling",
              "Bicarakan kekhawatiran kepada konselor"],
    ),
    2: dict(
        label="TINGGI", badge="bh", color="#ff6b8a", emoji="😟",
        desc="Tingkat stres Anda tergolong <strong>tinggi</strong>. Sangat disarankan "
             "untuk segera mencari bantuan profesional atau konselor kampus.",
        tips=["Konsultasikan dengan psikolog atau konselor kampus",
              "Kurangi beban berlebih dan delegasikan tugas",
              "Prioritaskan tidur, olahraga ringan, dan pola makan sehat"],
    ),
}


# ════════════════════════════════════════════════════════════════
# LOAD MODEL
# ════════════════════════════════════════════════════════════════
model, scaler, le, selected_features, missing = load_artifacts()


# ════════════════════════════════════════════════════════════════
# SIDEBAR
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
        slider_feats = [f for f in selected_features if f not in ENGINEERED]
        rendered = set()

        for grp_label, grp_feats in GROUPS.items():
            in_model = [f for f in grp_feats if f in slider_feats]
            if not in_model:
                continue
            st.markdown(
                f'<div style="font-size:.68rem;color:#8b8fad;letter-spacing:.1em;'
                f'text-transform:uppercase;margin:.8rem 0 .3rem;">{grp_label}</div>',
                unsafe_allow_html=True)
            for feat in in_model:
                icon, label = DISPLAY.get(feat, ("📌", feat.replace("_", " ").title()))
                lo, hi, dflt, step = SLIDERS.get(feat, (0.0, 10.0, 5.0, 1.0))
                with st.expander(f"{icon}  {label}", expanded=False):
                    inputs[feat] = st.slider(label, lo, hi, dflt, step,
                                             label_visibility="collapsed", key=f"s_{feat}")
                rendered.add(feat)

        for feat in [f for f in slider_feats if f not in rendered]:
            icon, label = DISPLAY.get(feat, ("📌", feat.replace("_", " ").title()))
            lo, hi, dflt, step = SLIDERS.get(feat, (0.0, 10.0, 5.0, 1.0))
            with st.expander(f"{icon}  {label}", expanded=False):
                inputs[feat] = st.slider(label, lo, hi, dflt, step,
                                         label_visibility="collapsed", key=f"s_{feat}")

        if any(f in selected_features for f in ENGINEERED):
            st.markdown(
                '<div style="font-size:.68rem;color:#8b8fad;letter-spacing:.1em;'
                'text-transform:uppercase;margin:.8rem 0 .3rem;">🛠 Dihitung Otomatis</div>',
                unsafe_allow_html=True)
            if "academic_stress_index" in selected_features:
                v = (inputs.get("study_load", 5) + inputs.get("future_career_concerns", 5)
                     + inputs.get("peer_pressure", 2)) / 3
                st.caption(f"🧮 Academic Stress Index: **{v:.3f}**")
            if "daily_life_stress_index" in selected_features:
                v = (inputs.get("noise_level", 2) + inputs.get("living_conditions", 3)
                     + inputs.get("basic_needs", 3)) / 3
                st.caption(f"📊 Daily Life Stress Index: **{v:.3f}**")

    st.markdown("<hr>", unsafe_allow_html=True)
    run_btn = st.button("🔍  Analisis Sekarang", use_container_width=True)


# ── Hero ──────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-tag">Research Tool · CRISP-DM · XGBoost + SHAP</div>
  <h1>StressScope</h1>
  <p>Sistem prediksi tingkat stres mahasiswa berbasis <strong>Machine Learning</strong>
     dengan explainability penuh menggunakan <strong>SHAP</strong>. Masukkan parameter
     di sidebar, klik <em>Analisis Sekarang</em>.</p>
</div>""", unsafe_allow_html=True)

if missing:
    st.markdown(
        f'<div class="wbox">⚠️ <strong>File tidak ditemukan:</strong> '
        f'<code>{"</code>, <code>".join(missing)}</code><br>'
        f'Letakkan semua file <code>.pkl</code> di folder yang sama dengan <code>app.py</code>.'
        f'</div>', unsafe_allow_html=True)
    st.stop()


# ── Tabs ──────────────────────────────────────────────────────
t1, t2, t3 = st.tabs(["🔬  PREDIKSI & SHAP", "📖  TENTANG SISTEM", "📊  PANDUAN FITUR"])


# ════════════════════════════════════════════════════════════════
# TAB 1 — PREDIKSI
# ════════════════════════════════════════════════════════════════
with t1:
    if not run_btn:
        st.markdown(
            '<div class="ibox">💡 Atur parameter di <strong>sidebar kiri</strong>, '
            'lalu klik <strong>"Analisis Sekarang"</strong>.</div>',
            unsafe_allow_html=True)
    else:
        # ── Compute engineered features ─────────────────────────
        all_inp = dict(inputs)
        if "academic_stress_index" in selected_features:
            all_inp["academic_stress_index"] = (
                all_inp.get("study_load", 5)
                + all_inp.get("future_career_concerns", 5)
                + all_inp.get("peer_pressure", 2)
            ) / 3.0
        if "daily_life_stress_index" in selected_features:
            all_inp["daily_life_stress_index"] = (
                all_inp.get("noise_level", 2)
                + all_inp.get("living_conditions", 3)
                + all_inp.get("basic_needs", 3)
            ) / 3.0

        df_in  = pd.DataFrame([all_inp])[selected_features]
        df_sc  = pd.DataFrame(scaler.transform(df_in), columns=selected_features)
        pred   = int(model.predict(df_sc)[0])
        proba  = model.predict_proba(df_sc)[0]
        info   = STRESS_INFO[pred]

        # ── A: Result card + probability ───────────────────────
        st.markdown("### Hasil Prediksi")
        ca, cb = st.columns([1, 1.4])
        with ca:
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

        with cb:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="ctitle">Distribusi Probabilitas</div>',
                        unsafe_allow_html=True)
            pm = {0: ("RENDAH","#43e8d8"), 1: ("SEDANG","#f9c74f"), 2: ("TINGGI","#ff6b8a")}
            for i, p in enumerate(proba):
                lbl, clr = pm[i]
                pct = p * 100
                st.markdown(f"""
                <div class="prob-row">
                  <div class="prob-label"><span>{lbl}</span>
                    <span style="color:{clr};font-weight:600;">{pct:.1f}%</span></div>
                  <div class="prob-bg">
                    <div class="prob-fill" style="width:{pct:.1f}%;background:{clr};"></div>
                  </div>
                </div>""", unsafe_allow_html=True)
            fig_pie, ax_pie = plt.subplots(figsize=(4, 3))
            fig_pie.patch.set_facecolor("#141728"); ax_pie.set_facecolor("#141728")
            _, _, ats = ax_pie.pie(
                proba, labels=["Rendah","Sedang","Tinggi"],
                colors=["#43e8d8","#f9c74f","#ff6b8a"], autopct="%1.1f%%", startangle=90,
                wedgeprops=dict(width=.55, edgecolor="#141728", linewidth=2),
                textprops={"color":"#8b8fad","fontsize":8})
            for a in ats:
                a.set_color("#e8eaf6"); a.set_fontsize(8)
            ax_pie.set_title("Confidence", color="#e8eaf6", fontsize=9, pad=8)
            st.pyplot(fig_pie, use_container_width=False); plt.close(fig_pie)
            st.markdown("</div>", unsafe_allow_html=True)

        # ── Tips ────────────────────────────────────────────────
        st.markdown("#### 💡 Rekomendasi Tindakan")
        tc = st.columns(3)
        for i, (col, tip) in enumerate(zip(tc, info["tips"])):
            col.markdown(f"""
            <div class="card" style="text-align:center;">
              <div style="font-size:1.5rem;margin-bottom:.5rem;">{'🌿🧘📋'[i]}</div>
              <div style="font-size:.82rem;color:#8b8fad;line-height:1.5;">{tip}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<hr>", unsafe_allow_html=True)

        # ── B: Feature table ────────────────────────────────────
        st.markdown("### Input Anda — Ringkasan Fitur")
        rows = []
        for feat in selected_features:
            icon, label = DISPLAY.get(feat, ("📌", feat.replace("_", " ").title()))
            rows.append({
                "": icon,
                "Fitur":        label,
                "Tipe":         "🛠 Engineered" if feat in ENGINEERED else "📥 Input",
                "Nilai Input":  round(float(df_in[feat].iloc[0]), 4),
                "Nilai Scaled": round(float(df_sc[feat].iloc[0]), 4),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        st.markdown("<hr>", unsafe_allow_html=True)

        # ── C: SHAP ──────────────────────────────────────────────
        st.markdown("### 🔍 Penjelasan SHAP (Explainable AI)")
        st.markdown("""<div class="ibox">
        SHAP (<em>SHapley Additive exPlanations</em>) mengukur kontribusi setiap fitur.
        Nilai <span style="color:#ff6b8a;">positif (merah)</span> mendorong stres lebih tinggi;
        nilai <span style="color:#43e8d8;">negatif (biru)</span> mendorong lebih rendah.
        </div>""", unsafe_allow_html=True)

        with st.spinner("Menghitung SHAP values..."):
            try:
                # Pass the sklearn model directly — SHAP handles get_booster() internally.
                # The _tree.py patch above has already fixed the base_score issue.
                explainer = shap.TreeExplainer(model)
                shap_raw  = explainer.shap_values(df_sc)

                # Normalise to (n_samples, n_features, n_classes)
                if isinstance(shap_raw, list):
                    sv = np.stack(shap_raw, axis=-1)
                    ev = np.array(explainer.expected_value)
                elif isinstance(shap_raw, np.ndarray) and shap_raw.ndim == 3:
                    sv = shap_raw
                    ev = np.array(explainer.expected_value)
                elif isinstance(shap_raw, np.ndarray) and shap_raw.ndim == 2:
                    sv = shap_raw[:, :, np.newaxis]
                    ev = np.array([explainer.expected_value])
                else:
                    raise ValueError(f"Unexpected shap_raw shape: {np.array(shap_raw).shape}")

                n_cls   = sv.shape[2]
                cls_idx = min(pred, n_cls - 1)
                sv_sam  = sv[0, :, cls_idx]
                ev_val  = float(ev[cls_idx]) if hasattr(ev, "__len__") else float(ev)

                feat_names = selected_features
                sidx       = np.argsort(np.abs(sv_sam))[::-1]
                sv_sorted  = sv_sam[sidx]
                lbl_sorted = [DISPLAY.get(feat_names[i], ("",""))[1] for i in sidx]
                col_sorted = ["#ff6b8a" if v > 0 else "#43e8d8" for v in sv_sorted]

                # C1: Bar chart
                st.markdown("#### Kontribusi Fitur — Sample Ini")
                fig_bar, ax_bar = dark_fig(9, max(3.5, len(feat_names) * 0.55))
                bars = ax_bar.barh(lbl_sorted[::-1], sv_sorted[::-1],
                                   color=col_sorted[::-1], height=0.6,
                                   edgecolor="#141728", linewidth=0.5)
                ax_bar.axvline(0, color="#252840", linewidth=1)
                ax_bar.set_xlabel("SHAP Value", color="#8b8fad", fontsize=8)
                ax_bar.set_title(f"SHAP — Kelas: {info['label']}",
                                 color="#e8eaf6", fontsize=10)
                for bar, val in zip(bars, sv_sorted[::-1]):
                    off = 0.003 if val >= 0 else -0.003
                    ax_bar.text(val + off, bar.get_y() + bar.get_height() / 2,
                                f"{val:+.4f}", va="center",
                                ha="left" if val >= 0 else "right",
                                color="#e8eaf6", fontsize=7)
                plt.tight_layout()
                st.pyplot(fig_bar, use_container_width=True); plt.close(fig_bar)

                # C2: Waterfall chart
                st.markdown("#### Waterfall — Bagaimana Prediksi Terbentuk")
                fig_wf, ax_wf = dark_fig(10, 4.5)
                bots, hgts, wcols = [], [], []
                cur = ev_val
                for v in sv_sorted:
                    if v >= 0:
                        bots.append(cur); hgts.append(v); wcols.append("#ff6b8a")
                    else:
                        bots.append(cur + v); hgts.append(-v); wcols.append("#43e8d8")
                    cur += v
                xp = np.arange(len(hgts))
                ax_wf.bar(xp, hgts, bottom=bots, color=wcols,
                          width=.55, edgecolor="#141728", linewidth=.5)
                ax_wf.axhline(ev_val, color="#f9c74f", lw=1, ls="--", alpha=.6)
                ax_wf.set_xticks(xp)
                ax_wf.set_xticklabels(lbl_sorted, rotation=30, ha="right",
                                      fontsize=7, color="#8b8fad")
                ax_wf.set_ylabel("Model Output (log-odds)", color="#8b8fad", fontsize=8)
                ax_wf.set_title("Waterfall: Kumulatif Kontribusi SHAP",
                                color="#e8eaf6", fontsize=10)
                ax_wf.legend(handles=[
                    mpatches.Patch(color="#ff6b8a", label="Mendorong STRES NAIK"),
                    mpatches.Patch(color="#43e8d8", label="Mendorong STRES TURUN"),
                    plt.Line2D([0], [0], color="#f9c74f", ls="--",
                               label=f"Baseline E[f(X)]={ev_val:.3f}"),
                ], facecolor="#141728", edgecolor="#252840",
                   labelcolor="#8b8fad", fontsize=7)
                plt.tight_layout()
                st.pyplot(fig_wf, use_container_width=True); plt.close(fig_wf)

                # C3: Detail table
                st.markdown("#### Tabel Detail SHAP Values")
                p33, p66 = np.percentile(np.abs(sv_sam), [33, 66])
                shap_rows = []
                for i in sidx:
                    icon, label = DISPLAY.get(feat_names[i], ("📌", feat_names[i]))
                    sv_i = sv_sam[i]
                    shap_rows.append({
                        "": icon, "Fitur": label,
                        "Tipe": "🛠" if feat_names[i] in ENGINEERED else "📥",
                        "Nilai Input": round(float(df_in[feat_names[i]].iloc[0]), 3),
                        "SHAP Value":  round(float(sv_i), 5),
                        "Arah":        "🔴 Naik" if sv_i > 0 else "🔵 Turun",
                        "Dampak":      ("🔥 Tinggi" if abs(sv_i) > p66
                                        else "⚡ Sedang" if abs(sv_i) > p33
                                        else "💤 Rendah"),
                    })
                st.dataframe(pd.DataFrame(shap_rows),
                             use_container_width=True, hide_index=True)

                # C4: Auto narrative
                pos_f = [DISPLAY.get(feat_names[sidx[i]], ("",""))[1]
                         for i in range(len(sidx)) if sv_sam[sidx[i]] > 0][:2]
                neg_f = [DISPLAY.get(feat_names[sidx[i]], ("",""))[1]
                         for i in range(len(sidx)) if sv_sam[sidx[i]] < 0][:2]
                st.markdown(f"""<div class="ibox">
                📝 <strong>Interpretasi Otomatis:</strong><br>
                Faktor yang paling <span style="color:#ff6b8a;">meningkatkan</span> risiko stres:
                <strong>{", ".join(pos_f) or "–"}</strong>.<br>
                Faktor yang paling <span style="color:#43e8d8;">menurunkan</span> risiko stres:
                <strong>{", ".join(neg_f) or "–"}</strong>.
                </div>""", unsafe_allow_html=True)

            except Exception as e:
                st.error(f"SHAP error: {e}")
                st.exception(e)


# ════════════════════════════════════════════════════════════════
# TAB 2 — TENTANG SISTEM
# ════════════════════════════════════════════════════════════════
with t2:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div class="card">
          <div class="ctitle">Tentang Sistem</div>
          <p style="font-size:.88rem;color:#8b8fad;line-height:1.7;">
            <strong style="color:#e8eaf6;">StressScope</strong> memprediksi tingkat stres
            mahasiswa menggunakan kerangka <strong style="color:#6c63ff;">CRISP-DM</strong>
            dengan tiga algoritma: Random Forest, XGBoost, LightGBM.
          </p>
          <p style="font-size:.88rem;color:#8b8fad;line-height:1.7;">
            Model aktif: <strong style="color:#6c63ff;">XGBoost</strong> — dipilih berdasarkan
            Accuracy, Macro F1, dan ROC-AUC (5-fold CV).
            Explainability: <strong style="color:#6c63ff;">SHAP TreeExplainer</strong>.
          </p>
        </div>
        <div class="card">
          <div class="ctitle">Pipeline CRISP-DM</div>
          <div style="font-size:.83rem;color:#8b8fad;line-height:2.2;">
            1️⃣ <strong style="color:#e8eaf6;">Business Understanding</strong><br>
            2️⃣ <strong style="color:#e8eaf6;">Data Understanding</strong> — EDA<br>
            3️⃣ <strong style="color:#e8eaf6;">Data Preparation</strong> — Imputasi, engineering, scaling<br>
            4️⃣ <strong style="color:#e8eaf6;">Modeling</strong> — RF · XGBoost · LightGBM<br>
            5️⃣ <strong style="color:#e8eaf6;">Evaluation</strong> — Accuracy · F1 · ROC-AUC<br>
            6️⃣ <strong style="color:#e8eaf6;">Deployment</strong> — Streamlit + SHAP
          </div>
        </div>""", unsafe_allow_html=True)

    with c2:
        n = len(selected_features) if selected_features else "?"
        badges = "".join(
            f'<code style="background:#0d0f1a;padding:.15rem .4rem;border-radius:4px;'
            f'font-size:.73rem;margin:2px;display:inline-block;">{f}</code>'
            for f in (selected_features or [])
        )
        st.markdown(f"""
        <div class="card">
          <div class="ctitle">Feature Engineering</div>
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
          <div class="ctitle">Fitur Model ({n} total)</div>
          <div style="line-height:2.2;">{badges}</div>
        </div>
        <div class="card">
          <div class="ctitle">Label Kelas</div>
          <div style="font-size:.85rem;line-height:2.4;">
            <span class="badge bl">0 — RENDAH</span>&nbsp;Stres minimal<br>
            <span class="badge bm">1 — SEDANG</span>&nbsp;Mulai menekan<br>
            <span class="badge bh">2 — TINGGI</span>&nbsp;Perlu intervensi
          </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("""<div class="wbox">⚠️ <strong>Disclaimer:</strong>
    Aplikasi ini bersifat <strong>akademik</strong>. Hasil bukan diagnosis resmi.
    Jika mengalami tekanan berat, konsultasikan dengan profesional kesehatan mental.
    </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
# TAB 3 — PANDUAN FITUR
# ════════════════════════════════════════════════════════════════
with t3:
    st.markdown("### 📋 Fitur yang Digunakan Model")
    if selected_features:
        guide = []
        for feat in selected_features:
            icon, label = DISPLAY.get(feat, ("📌", feat.replace("_", " ").title()))
            lo, hi, dflt, _ = SLIDERS.get(feat, (0.0, 10.0, 5.0, 1.0))
            guide.append({"": icon, "Fitur": label, "Kode": feat,
                           "Min": lo, "Max": hi, "Default": dflt,
                           "Tipe": "🛠 Engineered" if feat in ENGINEERED else "📥 Input"})
        st.dataframe(pd.DataFrame(guide), use_container_width=True, hide_index=True)

    st.markdown("### 🧮 Rumus Feature Engineering")
    st.markdown("""
    <div class="card">
      <div class="ctitle">Academic Stress Index</div>
      <div style="background:#0d0f1a;border-radius:8px;padding:.8rem;
           font-family:'Space Mono',monospace;font-size:.75rem;color:#43e8d8;">
        academic_stress_index = (study_load + future_career_concerns + peer_pressure) / 3
      </div>
    </div>
    <div class="card">
      <div class="ctitle">Daily Life Stress Index</div>
      <div style="background:#0d0f1a;border-radius:8px;padding:.8rem;
           font-family:'Space Mono',monospace;font-size:.75rem;color:#43e8d8;">
        daily_life_stress_index = (noise_level + living_conditions + basic_needs) / 3
      </div>
    </div>""", unsafe_allow_html=True)

    st.markdown("### 📈 Eksplorasi Dataset")
    csv_path = "StressLevelDataset.csv"
    if os.path.exists(csv_path):
        df_raw = pd.read_csv(csv_path)
        ca, cb = st.columns(2)
        with ca:
            st.markdown(f"""<div class="metric-row">
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
                     color=["#43e8d8","#f9c74f","#ff6b8a"], width=.5, edgecolor="#141728")
            ax_d.set_title("Distribusi Kelas Stres", color="#e8eaf6", fontsize=9)
            ax_d.set_ylabel("Jumlah", color="#8b8fad", fontsize=8)
            st.pyplot(fig_d, use_container_width=True); plt.close(fig_d)
        with cb:
            num_cols = df_raw.select_dtypes(include=np.number).columns.tolist()
            hm = [c for c in ["study_load","future_career_concerns","peer_pressure",
                               "noise_level","living_conditions","basic_needs","stress_level"]
                  if c in num_cols][:7]
            if hm:
                fig_h, ax_h = plt.subplots(figsize=(5.5, 4.5))
                fig_h.patch.set_facecolor("#141728"); ax_h.set_facecolor("#141728")
                sns.heatmap(df_raw[hm].corr(), ax=ax_h, cmap="coolwarm", center=0,
                            annot=True, fmt=".2f", annot_kws={"size":7},
                            linewidths=.5, linecolor="#252840", cbar_kws={"shrink":.75})
                ax_h.tick_params(colors="#8b8fad", labelsize=7)
                ax_h.set_title("Korelasi Fitur Utama", color="#e8eaf6", fontsize=9)
                st.pyplot(fig_h, use_container_width=True); plt.close(fig_h)
    else:
        st.markdown(
            '<div class="ibox">Letakkan <code>StressLevelDataset.csv</code> '
            'di folder yang sama dengan <code>app.py</code>.</div>',
            unsafe_allow_html=True)


# ── Footer ────────────────────────────────────────────────────
st.markdown("""<br>
<div style="text-align:center;padding:1.5rem;border-top:1px solid #252840;margin-top:2rem;">
  <span style="font-family:'Space Mono',monospace;font-size:.7rem;color:#3d4070;
       letter-spacing:.12em;">STRESSSCOPE · MACHINE LEARNING · CRISP-DM · SHAP</span>
</div>""", unsafe_allow_html=True)