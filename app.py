# ============================================================
# app.py — Prediksi Tingkat Stres Mahasiswa
# Framework  : Streamlit
# Model      : XGBoost + SHAP Explainability
# Author     : Stress Level Prediction Project
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

  /* Root palette */
  :root {
    --bg:        #0d0f1a;
    --surface:   #141728;
    --border:    #252840;
    --accent1:   #6c63ff;
    --accent2:   #ff6b8a;
    --accent3:   #43e8d8;
    --text:      #e8eaf6;
    --muted:     #8b8fad;
    --low:       #43e8d8;
    --mid:       #f9c74f;
    --high:      #ff6b8a;
  }

  html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg);
    color: var(--text);
  }

  /* Hero header */
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

  /* Cards */
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

  /* Stress badge */
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

  /* Metric tiles */
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

  /* Probability bar */
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
    transition: width 0.6s ease;
  }

  /* Sidebar */
  [data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
  }
  [data-testid="stSidebar"] * { color: var(--text) !important; }

  /* Tabs */
  .stTabs [data-baseweb="tab-list"] {
    gap: 0.5rem;
    background: transparent;
    border-bottom: 1px solid var(--border);
    padding-bottom: 0;
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

  /* Expander */
  [data-testid="stExpander"] {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
  }

  /* Divider */
  hr { border-color: var(--border) !important; }

  /* Scrollbar */
  ::-webkit-scrollbar { width: 6px; }
  ::-webkit-scrollbar-track { background: var(--bg); }
  ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }

  /* Hide Streamlit default branding */
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
    transition: opacity 0.2s;
  }
  .stButton>button:hover { opacity: 0.85; }

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


# ── Helper: load model ────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_artifacts():
    """Load XGBoost model + preprocessing artefacts from disk."""
    missing = [f for f in ["xgb_model.pkl","scaler.pkl","label_encoder.pkl","selected_features.pkl"]
               if not os.path.exists(f)]
    if missing:
        return None, None, None, None, missing
    model           = joblib.load("xgb_model.pkl")
    scaler          = joblib.load("scaler.pkl")
    le              = joblib.load("label_encoder.pkl")
    selected_features = joblib.load("selected_features.pkl")
    return model, scaler, le, selected_features, []


# ── Helper: matplotlib dark style ────────────────────────────
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


# ── Feature meta ─────────────────────────────────────────────
FEATURE_META = {
    "sleep_duration":             ("🌙", "Durasi Tidur",             "jam/hari",   0.0, 12.0, 7.0),
    "daily_screen_time":          ("📱", "Waktu Layar Harian",       "jam/hari",   0.0, 15.0, 4.0),
    "study_load":                 ("📚", "Beban Belajar",             "0–10",       0.0, 10.0, 5.0),
    "exam_anxiety":               ("😰", "Kecemasan Ujian",          "0–10",       0.0, 10.0, 5.0),
    "academic_performance":       ("🎓", "Performa Akademik",         "0–10",       0.0, 10.0, 7.0),
    "future_career_concerns":     ("🔮", "Kekhawatiran Karir",        "0–10",       0.0, 10.0, 6.0),
    "teacher_student_relationship":("🤝","Hub. Guru–Murid",          "0–10",       0.0, 10.0, 8.0),
    "social_support":             ("👥", "Dukungan Sosial",           "0–10",       0.0, 10.0, 7.0),
}

STRESS_INFO = {
    0: {
        "label": "RENDAH",
        "badge": "badge-low",
        "color": "#43e8d8",
        "emoji": "😊",
        "desc":  "Tingkat stres Anda tergolong rendah. Kondisi psikologis, sosial, dan akademik Anda berada pada zona sehat. Pertahankan pola tidur, aktivitas sosial, dan manajemen waktu yang baik.",
        "tips":  ["Pertahankan rutinitas tidur 7–8 jam/malam",
                  "Jaga keseimbangan antara belajar dan istirahat",
                  "Terus bangun dukungan sosial yang positif"],
    },
    1: {
        "label": "SEDANG",
        "badge": "badge-mid",
        "color": "#f9c74f",
        "emoji": "😐",
        "desc":  "Tingkat stres Anda berada di level sedang. Ada beberapa faktor yang mulai menekan Anda. Segera lakukan intervensi ringan agar tidak meningkat ke level tinggi.",
        "tips":  ["Evaluasi beban studi dan prioritaskan tugas",
                  "Coba teknik relaksasi seperti meditasi atau journaling",
                  "Bicarakan kekhawatiran kepada teman atau konselor"],
    },
    2: {
        "label": "TINGGI",
        "badge": "badge-high",
        "color": "#ff6b8a",
        "emoji": "😟",
        "desc":  "Tingkat stres Anda tergolong tinggi. Hal ini dapat berdampak negatif pada kesehatan fisik dan mental. Sangat disarankan untuk segera mencari bantuan profesional atau konselor kampus.",
        "tips":  ["Konsultasikan kondisi Anda dengan konselor atau psikolog kampus",
                  "Kurangi beban berlebih dan delegasikan tugas bila memungkinkan",
                  "Prioritaskan tidur, olahraga, dan pola makan sehat"],
    },
}


# ── Sidebar: input ────────────────────────────────────────────
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
    for key, (icon, label, unit, lo, hi, default) in FEATURE_META.items():
        with st.expander(f"{icon} {label}", expanded=False):
            st.caption(f"Satuan: **{unit}**")
            inputs[key] = st.slider(
                label, min_value=lo, max_value=hi,
                value=default, step=0.5 if hi > 10 else 0.1,
                label_visibility="collapsed", key=key
            )

    st.markdown("<hr>", unsafe_allow_html=True)
    run_btn = st.button("🔍  Analisis Sekarang", use_container_width=True)


# ── Hero ──────────────────────────────────────────────────────
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


# ── Load model ────────────────────────────────────────────────
model, scaler, le, selected_features, missing = load_artifacts()

if missing:
    st.markdown(f"""
    <div class="warn-box">
    ⚠️ <strong>Model belum ditemukan.</strong> File berikut tidak ada di direktori yang sama dengan <code>app.py</code>:<br>
    <code>{'</code>, <code>'.join(missing)}</code><br><br>
    Silakan jalankan notebook pelatihan terlebih dahulu untuk menghasilkan file <code>.pkl</code>,
    lalu letakkan semua file di folder yang sama dengan <code>app.py</code> sebelum menjalankan Streamlit.
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ── About tab (always visible) ────────────────────────────────
tab_pred, tab_about, tab_data = st.tabs([
    "🔬  PREDIKSI & SHAP",
    "📖  TENTANG SISTEM",
    "📊  PANDUAN FITUR",
])


# ═══════════════════════════════════════════════════════════════
# TAB 1 — PREDIKSI
# ═══════════════════════════════════════════════════════════════
with tab_pred:

    if not run_btn:
        st.markdown("""
        <div class="info-box">
        💡 Atur parameter mahasiswa di <strong>sidebar kiri</strong>, lalu klik tombol
        <strong>"Analisis Sekarang"</strong> untuk melihat prediksi dan penjelasan SHAP.
        </div>
        """, unsafe_allow_html=True)

    else:
        # ── Feature engineering (sama dengan notebook) ──────────
        sleep_safe = inputs["sleep_duration"] if inputs["sleep_duration"] != 0 else 0.1

        engineered = {
            **inputs,
            "academic_stress_index": (
                0.4 * inputs["study_load"] +
                0.4 * inputs["exam_anxiety"] -
                0.2 * inputs["academic_performance"]
            ),
            "screen_sleep_ratio": inputs["daily_screen_time"] / sleep_safe,
        }

        df_input = pd.DataFrame([engineered])
        df_input = df_input[selected_features]

        # ── Scale ────────────────────────────────────────────────
        df_scaled = pd.DataFrame(
            scaler.transform(df_input),
            columns=selected_features
        )

        # ── Predict ──────────────────────────────────────────────
        pred_class = int(model.predict(df_scaled)[0])
        pred_proba = model.predict_proba(df_scaled)[0]
        info       = STRESS_INFO[pred_class]

        # ── Section A: Stress Level Result ───────────────────────
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
            </div>
            """, unsafe_allow_html=True)

        with col_prob:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">Distribusi Probabilitas</div>', unsafe_allow_html=True)

            labels_map = {0: ("RENDAH",  "#43e8d8"),
                          1: ("SEDANG",  "#f9c74f"),
                          2: ("TINGGI",  "#ff6b8a")}

            for i, prob in enumerate(pred_proba):
                lbl, clr = labels_map[i]
                pct = prob * 100
                st.markdown(f"""
                <div class="prob-row">
                  <div class="prob-label">
                    <span>{lbl}</span><span style="color:{clr};font-weight:600;">{pct:.1f}%</span>
                  </div>
                  <div class="prob-bar-bg">
                    <div class="prob-bar-fill"
                         style="width:{pct:.1f}%; background:{clr};"></div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

            # Probability pie
            fig_pie, ax_pie = plt.subplots(figsize=(4, 3))
            fig_pie.patch.set_facecolor("#141728")
            ax_pie.set_facecolor("#141728")
            wedge_colors = ["#43e8d8","#f9c74f","#ff6b8a"]
            wedges, texts, autotexts = ax_pie.pie(
                pred_proba,
                labels=["Rendah","Sedang","Tinggi"],
                colors=wedge_colors,
                autopct="%1.1f%%",
                startangle=90,
                wedgeprops=dict(width=0.55, edgecolor="#141728", linewidth=2),
                textprops={"color":"#8b8fad","fontsize":8},
            )
            for at in autotexts:
                at.set_color("#e8eaf6")
                at.set_fontsize(8)
            ax_pie.set_title("Confidence Distribution", color="#e8eaf6", fontsize=9, pad=8)
            st.pyplot(fig_pie, use_container_width=False)
            plt.close(fig_pie)

            st.markdown("</div>", unsafe_allow_html=True)

        # ── Section B: Tips ─────────────────────────────────────
        st.markdown("#### 💡 Rekomendasi Tindakan")
        tip_cols = st.columns(len(info["tips"]))
        for i, (col, tip) in enumerate(zip(tip_cols, info["tips"])):
            col.markdown(f"""
            <div class="card" style="text-align:center; height:100%;">
              <div style="font-size:1.5rem; margin-bottom:0.5rem;">{'🌿🧘📋'[i]}</div>
              <div style="font-size:0.82rem; color:#8b8fad; line-height:1.5;">{tip}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<hr>", unsafe_allow_html=True)

        # ── Section C: Feature Summary ───────────────────────────
        st.markdown("### Input Anda — Ringkasan Fitur")

        # Show engineered features too
        summary_data = {
            "Fitur": [],
            "Nilai Input": [],
            "Nilai Scaled": [],
        }
        for feat in selected_features:
            summary_data["Fitur"].append(feat.replace("_"," ").title())
            summary_data["Nilai Input"].append(round(engineered.get(feat, 0), 3))
            summary_data["Nilai Scaled"].append(round(df_scaled[feat].iloc[0], 3))

        df_summary = pd.DataFrame(summary_data)
        st.dataframe(df_summary, use_container_width=True, hide_index=True)

        st.markdown("<hr>", unsafe_allow_html=True)

        # ── Section D: SHAP ──────────────────────────────────────
        st.markdown("### 🔍 Penjelasan SHAP (Explainable AI)")
        st.markdown("""
        <div class="info-box">
        SHAP (<em>SHapley Additive exPlanations</em>) mengukur kontribusi <strong>setiap fitur</strong>
        terhadap prediksi. Nilai positif mendorong stres lebih tinggi; nilai negatif mendorong lebih rendah.
        </div>
        """, unsafe_allow_html=True)

        with st.spinner("Menghitung SHAP values…"):
            try:
                explainer   = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(df_scaled)

                # shap_values shape: (n_samples, n_features, n_classes)  for XGBoost multi-class
                # OR list of arrays — handle both
                if isinstance(shap_values, list):
                    sv_matrix = np.stack(shap_values, axis=-1)   # → (1, n_feat, n_class)
                    ev_array  = np.array(explainer.expected_value)
                else:
                    # numpy array (1, n_feat, n_class)
                    sv_matrix = shap_values
                    ev_array  = np.array(explainer.expected_value)

                shap_sample    = sv_matrix[0, :, pred_class]   # (n_feat,)
                expected_value = float(ev_array[pred_class])

                # ── D1: Bar chart of SHAP ─────────────────────────
                st.markdown("#### Kontribusi Fitur — Sample Ini")
                feat_names  = selected_features
                sorted_idx  = np.argsort(np.abs(shap_sample))[::-1]
                sorted_vals = shap_sample[sorted_idx]
                sorted_feat = [feat_names[i].replace("_"," ").title() for i in sorted_idx]
                colors_bar  = ["#ff6b8a" if v > 0 else "#43e8d8" for v in sorted_vals]

                fig_bar, ax_bar = dark_fig(w=9, h=4)
                bars = ax_bar.barh(sorted_feat[::-1], sorted_vals[::-1],
                                   color=colors_bar[::-1], height=0.6,
                                   edgecolor="#141728", linewidth=0.5)
                ax_bar.axvline(0, color="#252840", linewidth=1)
                ax_bar.set_xlabel("SHAP Value (impact on model output)", color="#8b8fad", fontsize=8)
                ax_bar.set_title(f"SHAP — Kelas Prediksi: {info['label']}", color="#e8eaf6", fontsize=10)

                # Value labels
                for bar, val in zip(bars, sorted_vals[::-1]):
                    ax_bar.text(
                        val + (0.01 if val >= 0 else -0.01),
                        bar.get_y() + bar.get_height()/2,
                        f"{val:+.3f}",
                        va="center",
                        ha="left" if val >= 0 else "right",
                        color="#e8eaf6", fontsize=7,
                    )
                plt.tight_layout()
                st.pyplot(fig_bar, use_container_width=True)
                plt.close(fig_bar)

                # ── D2: Force-plot style waterfall ───────────────
                st.markdown("#### Waterfall — Bagaimana Prediksi Terbentuk")

                fig_wf, ax_wf = dark_fig(w=10, h=4.5)
                running = expected_value
                positions = [running]
                for v in shap_sample[sorted_idx]:
                    running += v
                    positions.append(running)

                x_labels = ["E[f(X)]"] + [feat_names[i].replace("_"," ").title() for i in sorted_idx] + ["f(X)"]
                bottoms  = []
                heights  = []
                bar_colors = []

                cur = expected_value
                for v in shap_sample[sorted_idx]:
                    if v >= 0:
                        bottoms.append(cur)
                        heights.append(v)
                        bar_colors.append("#ff6b8a")
                    else:
                        bottoms.append(cur + v)
                        heights.append(-v)
                        bar_colors.append("#43e8d8")
                    cur += v

                x_pos = np.arange(len(heights))
                ax_wf.bar(x_pos, heights, bottom=bottoms, color=bar_colors, width=0.55,
                          edgecolor="#141728", linewidth=0.5)

                # Baseline dot
                ax_wf.axhline(expected_value, color="#f9c74f", linewidth=1, linestyle="--", alpha=0.6)
                ax_wf.set_xticks(x_pos)
                short_labels = [feat_names[i].replace("_"," ").replace(" ","\\n").title() for i in sorted_idx]
                ax_wf.set_xticklabels([l.replace("_"," ").title() for l in
                                        [feat_names[i] for i in sorted_idx]],
                                       rotation=30, ha="right", fontsize=7, color="#8b8fad")
                ax_wf.set_ylabel("Model Output (log-odds)", color="#8b8fad", fontsize=8)
                ax_wf.set_title("Waterfall: Kumulatif Kontribusi SHAP", color="#e8eaf6", fontsize=10)

                red_patch  = mpatches.Patch(color="#ff6b8a", label="Mendorong STRES NAIK")
                blue_patch = mpatches.Patch(color="#43e8d8", label="Mendorong STRES TURUN")
                yel_line   = plt.Line2D([0],[0], color="#f9c74f", linestyle="--", label="Baseline E[f(X)]")
                ax_wf.legend(handles=[red_patch, blue_patch, yel_line],
                             facecolor="#141728", edgecolor="#252840",
                             labelcolor="#8b8fad", fontsize=7)
                plt.tight_layout()
                st.pyplot(fig_wf, use_container_width=True)
                plt.close(fig_wf)

                # ── D3: SHAP table ────────────────────────────────
                st.markdown("#### Tabel Detail SHAP Values")
                shap_df = pd.DataFrame({
                    "Fitur": [feat_names[i].replace("_"," ").title() for i in sorted_idx],
                    "Nilai Input":  [round(engineered.get(feat_names[i], 0), 3) for i in sorted_idx],
                    "SHAP Value":   [round(shap_sample[i], 4)   for i in sorted_idx],
                    "Arah":         ["🔴 Naik" if shap_sample[i] > 0 else "🔵 Turun"
                                     for i in sorted_idx],
                    "Dampak":       ["Tinggi" if abs(shap_sample[i]) > np.percentile(np.abs(shap_sample),66)
                                     else ("Sedang" if abs(shap_sample[i]) > np.percentile(np.abs(shap_sample),33)
                                           else "Rendah")
                                     for i in sorted_idx],
                })
                st.dataframe(shap_df, use_container_width=True, hide_index=True)

                # ── D4: Narasi ────────────────────────────────────
                top_pos = [feat_names[sorted_idx[i]].replace("_"," ") for i in range(len(sorted_idx))
                           if shap_sample[sorted_idx[i]] > 0][:2]
                top_neg = [feat_names[sorted_idx[i]].replace("_"," ") for i in range(len(sorted_idx))
                           if shap_sample[sorted_idx[i]] < 0][:2]

                st.markdown(f"""
                <div class="info-box">
                📝 <strong>Interpretasi Otomatis:</strong><br>
                Faktor yang paling <span style="color:#ff6b8a;">meningkatkan</span> prediksi stres Anda:
                <strong>{", ".join(top_pos) if top_pos else "–"}</strong>.<br>
                Faktor yang paling <span style="color:#43e8d8;">menurunkan</span> prediksi stres Anda:
                <strong>{", ".join(top_neg) if top_neg else "–"}</strong>.
                </div>
                """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"SHAP error: {e}")
                st.exception(e)


# ═══════════════════════════════════════════════════════════════
# TAB 2 — ABOUT
# ═══════════════════════════════════════════════════════════════
with tab_about:
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("""
        <div class="card">
          <div class="card-title">Tentang Sistem</div>
          <p style="font-size:0.88rem; color:#8b8fad; line-height:1.7;">
            <strong style="color:#e8eaf6;">StressScope</strong> adalah aplikasi prediksi tingkat stres
            mahasiswa yang dibangun menggunakan kerangka kerja <strong style="color:#6c63ff;">CRISP-DM</strong>
            dan tiga algoritma machine learning: <em>Random Forest, XGBoost, dan LightGBM</em>.
          </p>
          <p style="font-size:0.88rem; color:#8b8fad; line-height:1.7;">
            Model yang digunakan untuk prediksi real-time adalah <strong style="color:#6c63ff;">XGBoost</strong>,
            dipilih berdasarkan perbandingan metrik Accuracy, Macro F1-Score, dan ROC-AUC pada
            evaluasi 5-fold cross-validation.
          </p>
          <p style="font-size:0.88rem; color:#8b8fad; line-height:1.7;">
            Explainability disediakan oleh <strong style="color:#6c63ff;">SHAP (TreeExplainer)</strong>,
            yang memungkinkan interpretasi lokal (per sampel) maupun global (seluruh dataset).
          </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="card">
          <div class="card-title">Pipeline CRISP-DM</div>
          <div style="font-size:0.83rem; color:#8b8fad; line-height:2;">
            1️⃣ &nbsp;<strong style="color:#e8eaf6;">Business Understanding</strong> — Definisi tujuan prediksi<br>
            2️⃣ &nbsp;<strong style="color:#e8eaf6;">Data Understanding</strong> — EDA & visualisasi distribusi<br>
            3️⃣ &nbsp;<strong style="color:#e8eaf6;">Data Preparation</strong> — Imputasi, engineering, scaling<br>
            4️⃣ &nbsp;<strong style="color:#e8eaf6;">Modeling</strong> — RF · XGBoost · LightGBM + CV<br>
            5️⃣ &nbsp;<strong style="color:#e8eaf6;">Evaluation</strong> — Accuracy · Macro F1 · ROC-AUC<br>
            6️⃣ &nbsp;<strong style="color:#e8eaf6;">Deployment</strong> — Streamlit + SHAP Explainability
          </div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="card">
          <div class="card-title">Feature Engineering</div>
          <p style="font-size:0.85rem; color:#8b8fad; line-height:1.6; margin-bottom:0.8rem;">
            Dua fitur baru dibuat dari fitur asli:
          </p>
          <div style="background:#0d0f1a; border-radius:8px; padding:1rem;
               font-family:'Space Mono',monospace; font-size:0.75rem; color:#43e8d8;
               line-height:1.8;">
            academic_stress_index =<br>
            &nbsp;&nbsp;0.4 × study_load<br>
            &nbsp;&nbsp;+ 0.4 × exam_anxiety<br>
            &nbsp;&nbsp;− 0.2 × academic_performance<br><br>
            screen_sleep_ratio =<br>
            &nbsp;&nbsp;daily_screen_time / sleep_duration
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="card">
          <div class="card-title">Label Kelas Target</div>
          <div style="font-size:0.85rem; line-height:2.2;">
            <span class="badge badge-low">0 — RENDAH</span>
            &nbsp;Stres minimal, kondisi sehat<br>
            <span class="badge badge-mid">1 — SEDANG</span>
            &nbsp;Tekanan mulai terasa, perlu perhatian<br>
            <span class="badge badge-high">2 — TINGGI</span>
            &nbsp;Stres berat, intervensi direkomendasikan
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="card">
          <div class="card-title">Disclaimer</div>
          <p style="font-size:0.82rem; color:#8b8fad; line-height:1.6;">
            ⚠️ Aplikasi ini bersifat <strong style="color:#f9c74f;">akademik dan informatif</strong>.
            Hasil prediksi <strong>bukan</strong> diagnosis medis atau psikologis resmi.
            Jika Anda mengalami tekanan berat, segera konsultasikan dengan
            <strong style="color:#e8eaf6;">profesional kesehatan mental</strong> atau konselor kampus.
          </p>
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# TAB 3 — DATA GUIDE
# ═══════════════════════════════════════════════════════════════
with tab_data:
    st.markdown("### 📋 Panduan Fitur Input")

    guide_data = []
    for key, (icon, label, unit, lo, hi, default) in FEATURE_META.items():
        guide_data.append({
            "Ikon": icon,
            "Nama Fitur": label,
            "Kode Kolom":  key,
            "Satuan / Skala": unit,
            "Min": lo,
            "Max": hi,
            "Default": default,
        })
    st.dataframe(pd.DataFrame(guide_data), use_container_width=True, hide_index=True)

    st.markdown("### 🛠️ Fitur Rekayasa (Engineered)")
    st.markdown("""
    <div class="card">
      <div class="card-title">academic_stress_index</div>
      <p style="font-size:0.85rem; color:#8b8fad; line-height:1.6;">
        Indeks komposit yang menggabungkan beban belajar, kecemasan ujian, dan performa akademik.
        Semakin tinggi beban dan kecemasan — semakin tinggi indeks ini. Performa akademik yang baik
        menurunkan indeks. Dihitung otomatis dari input Anda.
      </p>
    </div>
    <div class="card">
      <div class="card-title">screen_sleep_ratio</div>
      <p style="font-size:0.85rem; color:#8b8fad; line-height:1.6;">
        Rasio waktu layar terhadap durasi tidur. Nilai > 1 menunjukkan bahwa seseorang menghabiskan
        lebih banyak waktu di depan layar daripada tidur — kondisi yang umumnya berkorelasi dengan
        stres lebih tinggi. Dihitung otomatis dari input Anda.
      </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Quick EDA from CSV (if available) ────────────────────
    csv_path = "StressLevelDataset.csv"
    if os.path.exists(csv_path):
        st.markdown("### 📈 Eksplorasi Dataset")
        df_raw = pd.read_csv(csv_path)

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(f"""
            <div class="metric-row">
              <div class="metric-tile">
                <div class="val" style="color:#6c63ff;">{len(df_raw):,}</div>
                <div class="lbl">Total Sampel</div>
              </div>
              <div class="metric-tile">
                <div class="val" style="color:#43e8d8;">{df_raw.shape[1]}</div>
                <div class="lbl">Fitur Asli</div>
              </div>
              <div class="metric-tile">
                <div class="val" style="color:#ff6b8a;">{df_raw['stress_level'].nunique()}</div>
                <div class="lbl">Kelas Target</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            # Distribution bar
            dist = df_raw["stress_level"].value_counts().sort_index()
            fig_dist, ax_dist = dark_fig(w=5, h=3)
            ax_dist.bar(["Rendah","Sedang","Tinggi"], dist.values,
                        color=["#43e8d8","#f9c74f","#ff6b8a"],
                        width=0.5, edgecolor="#141728")
            ax_dist.set_title("Distribusi Kelas Stres", color="#e8eaf6", fontsize=9)
            ax_dist.set_ylabel("Jumlah", color="#8b8fad", fontsize=8)
            st.pyplot(fig_dist, use_container_width=True)
            plt.close(fig_dist)

        with col_b:
            # Correlation heatmap (subset)
            subset_cols = ["anxiety_level","depression","sleep_quality",
                           "study_load","social_support","stress_level"]
            subset_cols = [c for c in subset_cols if c in df_raw.columns]
            corr_sub = df_raw[subset_cols].corr()
            fig_hm, ax_hm = plt.subplots(figsize=(5, 4))
            fig_hm.patch.set_facecolor("#141728")
            ax_hm.set_facecolor("#141728")
            sns.heatmap(corr_sub, ax=ax_hm, cmap="coolwarm", center=0,
                        annot=True, fmt=".2f", annot_kws={"size":7},
                        linewidths=0.5, linecolor="#252840",
                        cbar_kws={"shrink":0.75})
            ax_hm.tick_params(colors="#8b8fad", labelsize=7)
            ax_hm.set_title("Heatmap Korelasi (Subset)", color="#e8eaf6", fontsize=9)
            st.pyplot(fig_hm, use_container_width=True)
            plt.close(fig_hm)
    else:
        st.markdown("""
        <div class="info-box">
        Letakkan <code>StressLevelDataset.csv</code> di folder yang sama dengan <code>app.py</code>
        untuk melihat eksplorasi dataset di sini.
        </div>
        """, unsafe_allow_html=True)


# ── Footer ────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center; padding:1.5rem;
     border-top:1px solid #252840; margin-top:2rem;">
  <span style="font-family:'Space Mono',monospace; font-size:0.7rem;
       color:#3d4070; letter-spacing:0.12em;">
    STRESSSCOPE · MACHINE LEARNING · CRISP-DM · SHAP EXPLAINABILITY
  </span>
</div>
""", unsafe_allow_html=True)