import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="Prediksi Tingkat Stres Mahasiswa", layout="wide")

# =====================================================
# LOAD MODEL DAN PREPROCESSOR
# =====================================================

@st.cache_resource
def load_artifacts():
    model = joblib.load("xgb_model.pkl")
    scaler = joblib.load("scaler.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    selected_features = joblib.load("selected_features.pkl")
    return model, scaler, label_encoder, selected_features

model, scaler, le, selected_features = load_artifacts()

# =====================================================
# TITLE
# =====================================================

st.title("Prediksi Tingkat Stres Mahasiswa")
st.write(
"Aplikasi ini memprediksi tingkat stres mahasiswa berdasarkan beberapa faktor "
"dan menjelaskan hasil prediksi menggunakan Explainable AI (SHAP)."
)

# =====================================================
# SIDEBAR INPUT
# =====================================================

st.sidebar.header("Input Fitur Mahasiswa")

sleep_duration = st.sidebar.slider("Durasi Tidur (jam)", 0.0, 10.0, 7.0)
daily_screen_time = st.sidebar.slider("Waktu Layar Harian (jam)", 0.0, 15.0, 4.0)
study_load = st.sidebar.slider("Beban Belajar", 0.0, 10.0, 5.0)
exam_anxiety = st.sidebar.slider("Kecemasan Ujian", 0.0, 10.0, 5.0)
academic_performance = st.sidebar.slider("Performa Akademik", 0.0, 10.0, 7.0)
future_career_concerns = st.sidebar.slider("Kekhawatiran Karir", 0.0, 10.0, 6.0)
teacher_student_relationship = st.sidebar.slider("Hubungan Guru-Murid", 0.0, 10.0, 8.0)
social_support = st.sidebar.slider("Dukungan Sosial", 0.0, 10.0, 7.0)

# =====================================================
# FEATURE ENGINEERING (SAMA DENGAN TRAINING)
# =====================================================

academic_stress_index = (
    0.4 * study_load +
    0.4 * exam_anxiety -
    0.2 * academic_performance
)

sleep_duration_safe = sleep_duration if sleep_duration != 0 else 0.1

screen_sleep_ratio = daily_screen_time / sleep_duration_safe

input_data = pd.DataFrame({

    "sleep_duration":[sleep_duration],
    "daily_screen_time":[daily_screen_time],
    "study_load":[study_load],
    "exam_anxiety":[exam_anxiety],
    "academic_performance":[academic_performance],
    "future_career_concerns":[future_career_concerns],
    "teacher_student_relationship":[teacher_student_relationship],
    "social_support":[social_support],
    "academic_stress_index":[academic_stress_index],
    "screen_sleep_ratio":[screen_sleep_ratio]

})

st.subheader("Input Pengguna")
st.dataframe(input_data)

# =====================================================
# PREPROCESSING
# =====================================================

input_scaled = scaler.transform(input_data[selected_features])

input_scaled_df = pd.DataFrame(
    input_scaled,
    columns=selected_features
)

# =====================================================
# PREDICTION
# =====================================================

prediction = model.predict(input_scaled_df)
prediction_proba = model.predict_proba(input_scaled_df)

stress_map = {
    0:"Rendah",
    1:"Sedang",
    2:"Tinggi"
}

predicted_label = stress_map[prediction[0]]

st.subheader("Hasil Prediksi")

st.success(f"Tingkat Stres Diprediksi: **{predicted_label}**")

st.write("Probabilitas Prediksi:")

for i, prob in enumerate(prediction_proba[0]):
    st.write(f"{stress_map[i]} : {prob:.2f}")

# =====================================================
# SHAP EXPLANATION (SAMA DENGAN NOTEBOOK)
# =====================================================

st.subheader("Penjelasan Prediksi (SHAP Explainability)")

try:

    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(input_scaled_df)

    sample_index = 0

    predicted_class = prediction[0]

    shap_sample = shap_values[sample_index, :, predicted_class]

    expected_value = explainer.expected_value[predicted_class]

    fig, ax = plt.subplots(figsize=(10,6))

    shap.force_plot(
        expected_value,
        shap_sample,
        input_scaled_df.iloc[sample_index],
        matplotlib=True,
        show=False
    )

    st.pyplot(fig)

    st.write(
    "Grafik di atas menunjukkan kontribusi setiap fitur terhadap hasil prediksi. "
    "Fitur berwarna merah meningkatkan tingkat stres, sedangkan biru menurunkannya."
    )

except Exception as e:

    st.warning("Visualisasi SHAP tidak dapat ditampilkan.")
    st.write(str(e))