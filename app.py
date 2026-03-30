# app.py  –  Aplikasi Prediksi Tingkat Stres Mahasiswa
#
# ═══════════════════════════════════════════════════════════
# RINGKASAN TIGA BUG YANG DIPERBAIKI:
#
# BUG 1 – base_score array (root cause awal)
#   XGBoost >=2.0 menyimpan base_score sebagai array per kelas
#   ("[1.93E-2,-2.51E-2,5.84E-3]") untuk multiclass.
#   SHAP mencoba float(base_score) → ValueError.
#   FIX: Patch via JSON save/load (bukan load_config() yang diabaikan XGBoost).
#
# BUG 2 – patch_xgb_for_shap() versi lama tidak efektif
#   Versi sebelumnya memakai booster.load_config(json.dumps(cfg)).
#   Ternyata XGBoost 3.x mengabaikan perubahan base_score via load_config()
#   karena dibaca ulang dari format UBJ binary (save_raw), bukan dari config.
#   FIX: Simpan booster ke file JSON → patch → load_model() ulang.
#
# BUG 3 – SHAP 0.46.0 crash saat import di Python 3.14
#   shap/plots/colors/_colorconv.py menggunakan np.floating yang
#   dihapus di NumPy 2.x, sehingga crash saat import shap.
#   FIX: Upgrade ke shap>=0.47.0 (sudah diperbaiki mulai versi itu).
# ═══════════════════════════════════════════════════════════

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import ast
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────
# LOAD MODEL & PREPROCESSORS
# ──────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model             = joblib.load("xgb_model.pkl")
    scaler            = joblib.load("scaler.pkl")
    label_encoder     = joblib.load("label_encoder.pkl")
    selected_features = joblib.load("selected_features.pkl")
    return model, scaler, label_encoder, selected_features

model, scaler, le, selected_features = load_artifacts()

STRESS_LABEL = {0: "🟢 Rendah", 1: "🟡 Sedang", 2: "🔴 Tinggi"}


# ──────────────────────────────────────────────
# [FIX BUG 1 & 2] Patch XGBoost base_score via JSON save/load
# ──────────────────────────────────────────────
@st.cache_resource
def get_patched_model_and_explainer(_model):
    """
    Memperbaiki inkompatibilitas XGBoost >=2.0 dengan SHAP.

    Root cause:
      XGBoost >=2.0 pada multiclass menyimpan base_score sebagai
      array per kelas, misal '[1.93E-2,-2.51E-2,5.84E-3]'.
      SHAP membacanya dari format UBJ binary (save_raw) lalu
      melakukan float(base_score) → ValueError.

    Mengapa load_config() tidak bisa fix ini:
      XGBoost 3.x mengabaikan perubahan base_score yang ditulis
      ulang via load_config(), karena nilai aslinya tetap terbaca
      dari binary UBJ saat SHAP memanggil save_raw().

    Solusi: simpan ke JSON (format teks) → patch → load_model().
    """
    import xgboost as xgb
    import shap

    booster = _model.get_booster()

    # Periksa apakah base_score perlu dipatch
    booster.save_model("/tmp/_xgb_check.json")
    with open("/tmp/_xgb_check.json") as f:
        jmodel = json.load(f)

    raw_bs = jmodel["learner"]["learner_model_param"]["base_score"]
    try:
        float(raw_bs)  # sudah scalar → tidak perlu patch
        model_for_shap = _model
    except (ValueError, TypeError):
        # Array string → patch dengan scalar netral 0.5
        jmodel["learner"]["learner_model_param"]["base_score"] = "0.5"
        with open("/tmp/_xgb_patched.json", "w") as f:
            json.dump(jmodel, f)
        model_for_shap = xgb.XGBClassifier()
        model_for_shap.load_model("/tmp/_xgb_patched.json")

    explainer = shap.TreeExplainer(model_for_shap)
    return model_for_shap, explainer


# ──────────────────────────────────────────────
# FUNGSI NORMALISASI (sama persis dengan training)
# ──────────────────────────────────────────────
MINMAX_RANGE = {
    "study_load":             (0, 5),
    "future_career_concerns": (0, 5),
    "academic_performance":   (0, 5),
    "peer_pressure":          (0, 5),
    "bullying":               (0, 5),
    "social_support":         (0, 5),
}

def minmax_norm_single(value: float, feat: str) -> float:
    lo, hi = MINMAX_RANGE[feat]
    return (value - lo) / (hi - lo) if hi != lo else 0.0


def build_input_row(raw: dict) -> pd.DataFrame:
    w_study  = 0.6342
    w_career = 0.7426
    w_acad   = 0.7209
    total_w  = w_study + w_career + w_acad

    academic_stress_index = (
        (w_study  / total_w) * minmax_norm_single(raw["study_load"],             "study_load")
        + (w_career / total_w) * minmax_norm_single(raw["future_career_concerns"], "future_career_concerns")
        + (w_acad   / total_w) * (1 - minmax_norm_single(raw["academic_performance"], "academic_performance"))
    )
    MAX_LIVING, MAX_SAFETY, MAX_BASIC = 5, 5, 5
    environment_quality_index = (
        raw["noise_level"]
        + (MAX_LIVING - raw["living_conditions"])
        + (MAX_SAFETY - raw["safety"])
        + (MAX_BASIC  - raw["basic_needs"])
    )
    social_stress_score = (
        minmax_norm_single(raw["peer_pressure"], "peer_pressure")
        + minmax_norm_single(raw["bullying"],      "bullying")
        + (1 - minmax_norm_single(raw["social_support"], "social_support"))
    )
    full = {**raw,
            "academic_stress_index":     academic_stress_index,
            "environment_quality_index": environment_quality_index,
            "social_stress_score":       social_stress_score}
    return pd.DataFrame([full])[selected_features]


# ──────────────────────────────────────────────
# UI – HEADER
# ──────────────────────────────────────────────
st.title("🎓 Prediksi Tingkat Stres Mahasiswa")
st.write(
    "Aplikasi ini memprediksi tingkat stres mahasiswa berdasarkan "
    "20 faktor psikologis, fisik, akademik, dan sosial, "
    "serta memberikan penjelasan SHAP atas prediksi yang dihasilkan."
)

# ──────────────────────────────────────────────
# UI – SIDEBAR INPUT
# ──────────────────────────────────────────────
st.sidebar.header("📋 Input Data Mahasiswa")

with st.sidebar:
    st.subheader("🧠 Faktor Psikologis")
    anxiety_level         = st.slider("Tingkat Kecemasan",         0, 21, 10)
    self_esteem           = st.slider("Harga Diri (Self-Esteem)",  0, 30, 15)
    mental_health_history = st.selectbox("Riwayat Masalah Mental", [0, 1],
                                          format_func=lambda x: "Tidak" if x == 0 else "Ya")
    depression            = st.slider("Tingkat Depresi",            0, 27, 10)

    st.subheader("🏥 Kesehatan Fisik")
    headache          = st.slider("Frekuensi Sakit Kepala",    0, 5, 2)
    blood_pressure    = st.slider("Tekanan Darah",             1, 3, 2)
    sleep_quality     = st.slider("Kualitas Tidur",            1, 5, 3)
    breathing_problem = st.slider("Masalah Pernapasan",        0, 5, 1)

    st.subheader("🏠 Lingkungan")
    noise_level       = st.slider("Tingkat Kebisingan",        0, 5, 2)
    living_conditions = st.slider("Kondisi Tempat Tinggal",    1, 5, 3)
    safety            = st.slider("Tingkat Keamanan",          1, 5, 3)
    basic_needs       = st.slider("Pemenuhan Kebutuhan Dasar", 1, 5, 3)

    st.subheader("📚 Akademik")
    academic_performance         = st.slider("Performa Akademik",    1, 5, 3)
    study_load                   = st.slider("Beban Belajar",        1, 5, 3)
    teacher_student_relationship = st.slider("Hub. Dosen-Mahasiswa", 1, 5, 3)
    future_career_concerns       = st.slider("Kekhawatiran Karir",   1, 5, 3)

    st.subheader("👥 Sosial")
    social_support             = st.slider("Dukungan Sosial",           1, 5, 3)
    peer_pressure              = st.slider("Tekanan Teman Sebaya",      1, 5, 3)
    extracurricular_activities = st.slider("Aktivitas Ekstrakurikuler", 0, 5, 2)
    bullying                   = st.slider("Tingkat Perundungan",       0, 5, 1)

# ──────────────────────────────────────────────
# PROSES PREDIKSI
# ──────────────────────────────────────────────
raw_input = {
    "anxiety_level":                anxiety_level,
    "self_esteem":                  self_esteem,
    "mental_health_history":        mental_health_history,
    "depression":                   depression,
    "headache":                     headache,
    "blood_pressure":               blood_pressure,
    "sleep_quality":                sleep_quality,
    "breathing_problem":            breathing_problem,
    "noise_level":                  noise_level,
    "living_conditions":            living_conditions,
    "safety":                       safety,
    "basic_needs":                  basic_needs,
    "academic_performance":         academic_performance,
    "study_load":                   study_load,
    "teacher_student_relationship": teacher_student_relationship,
    "future_career_concerns":       future_career_concerns,
    "social_support":               social_support,
    "peer_pressure":                peer_pressure,
    "extracurricular_activities":   extracurricular_activities,
    "bullying":                     bullying,
}

df_input_raw    = build_input_row(raw_input)
df_input_scaled = pd.DataFrame(
    scaler.transform(df_input_raw),
    columns=selected_features,
)

prediction       = model.predict(df_input_scaled)
prediction_proba = model.predict_proba(df_input_scaled)

# ──────────────────────────────────────────────
# UI – HASIL PREDIKSI
# ──────────────────────────────────────────────
st.subheader("📊 Ringkasan Input")
col1, col2 = st.columns(2)
with col1:
    st.write("**20 Fitur Asli:**")
    st.dataframe(
        pd.DataFrame(raw_input.items(), columns=["Fitur", "Nilai"]),
        use_container_width=True,
    )
with col2:
    st.write("**3 Fitur Turunan (Engineered):**")
    eng_df = pd.DataFrame({
        "Fitur": ["academic_stress_index", "environment_quality_index", "social_stress_score"],
        "Nilai": [
            round(df_input_raw["academic_stress_index"].values[0], 4),
            round(df_input_raw["environment_quality_index"].values[0], 4),
            round(df_input_raw["social_stress_score"].values[0], 4),
        ],
    })
    st.dataframe(eng_df, use_container_width=True)

st.divider()
st.subheader("🎯 Hasil Prediksi Tingkat Stres")
predicted_label = STRESS_LABEL[prediction[0]]
st.metric("Tingkat Stres", predicted_label)

st.write("**Probabilitas per Kelas:**")
prob_df = pd.DataFrame({
    "Kelas":        [STRESS_LABEL[i] for i in range(3)],
    "Probabilitas": [f"{p:.2%}" for p in prediction_proba[0]],
})
st.dataframe(prob_df, use_container_width=True)

# ──────────────────────────────────────────────
# UI – SHAP EXPLANATION
# ──────────────────────────────────────────────
st.divider()
st.subheader("🔍 Penjelasan Prediksi (SHAP)")

if hasattr(model, "get_booster"):
    try:
        import shap

        # [FIX BUG 1 & 2] Model dengan base_score sudah dipatch via JSON
        model_for_shap, shap_explainer = get_patched_model_and_explainer(model)
        shap_vals = shap_explainer.shap_values(df_input_scaled)

        # Handle format SHAP lama (list) dan baru (array 3D)
        pred_class = int(prediction[0])
        if isinstance(shap_vals, list):
            sv = shap_vals[pred_class][0]
            ev = shap_explainer.expected_value[pred_class]
        else:
            # SHAP >=0.47: 3D array [sample, feature, class]
            sv = shap_vals[0, :, pred_class]
            ev = (shap_explainer.expected_value[pred_class]
                  if hasattr(shap_explainer.expected_value, "__len__")
                  else shap_explainer.expected_value)

        st.write(
            "Grafik berikut menunjukkan kontribusi setiap fitur terhadap prediksi. "
            "**Merah** = mendorong stres lebih tinggi · **Biru** = mendorong stres lebih rendah."
        )

        # ── Waterfall Plot ──────────────────────────────────────────
        exp = shap.Explanation(
            values=sv,
            base_values=ev,
            data=df_input_scaled.iloc[0].values,
            feature_names=selected_features,
        )
        fig1, _ = plt.subplots(figsize=(10, 7))
        shap.plots.waterfall(exp, show=False)
        st.pyplot(fig1)
        plt.close(fig1)

        # ── Bar Chart Kontribusi SHAP ───────────────────────────────
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        shap_series = pd.Series(sv, index=selected_features).sort_values(key=abs, ascending=True)
        colors = ["#E24B4A" if v > 0 else "#378ADD" for v in shap_series]
        shap_series.plot(kind="barh", color=colors, ax=ax2)
        ax2.axvline(0, color="gray", linewidth=0.8)
        ax2.set_title("Kontribusi Fitur (SHAP) terhadap Prediksi Ini")
        ax2.set_xlabel("SHAP Value")
        st.pyplot(fig2)
        plt.close(fig2)

    except Exception as e:
        st.error(f"⚠️ SHAP gagal dijalankan: {e}")
        st.info(
            "Pastikan `requirements.txt` menggunakan `shap>=0.47.0` "
            "agar kompatibel dengan Python 3.14 dan XGBoost 3.x."
        )
else:
    st.warning("SHAP TreeExplainer hanya tersedia untuk model berbasis pohon.")

st.caption(
    "Model: XGBoost · Interpretabilitas: SHAP TreeExplainer · "
    "Framework: CRISP-DM"
)