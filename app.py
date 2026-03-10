
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# Load the saved model and preprocessing tools
@st.cache_resource
def load_model_and_preprocessors():
    model = joblib.load('xgb_model.pkl')
    scaler = joblib.load('scaler.pkl')
    label_encoder = joblib.load('label_encoder.pkl')
    selected_features = joblib.load('selected_features.pkl')
    return model, scaler, label_encoder, selected_features

model, scaler, le, selected_features = load_model_and_preprocessors()

# Streamlit App Title
st.title('Prediksi Tingkat Stres Mahasiswa')
st.write('Aplikasi ini memprediksi tingkat stres mahasiswa berdasarkan beberapa faktor, serta memberikan penjelasan mengapa prediksi tersebut dihasilkan menggunakan SHAP.')

# Sidebar for user input
st.sidebar.header('Input Fitur Mahasiswa')

def user_input_features():
    sleep_duration = st.sidebar.slider('Durasi Tidur (jam)', 0.0, 10.0, 7.0)
    daily_screen_time = st.sidebar.slider('Waktu Layar Harian (jam)', 0.0, 15.0, 4.0)
    study_load = st.sidebar.slider('Beban Belajar (skala 0-10)', 0.0, 10.0, 5.0)
    exam_anxiety = st.sidebar.slider('Kecemasan Ujian (skala 0-10)', 0.0, 10.0, 5.0)
    academic_performance = st.sidebar.slider('Performa Akademik (skala 0-10)', 0.0, 10.0, 7.0)
    future_career_concerns = st.sidebar.slider('Kekhawatiran Karir Masa Depan (skala 0-10)', 0.0, 10.0, 6.0)
    teacher_student_relationship = st.sidebar.slider('Hubungan Guru-Murid (skala 0-10)', 0.0, 10.0, 8.0)
    social_support = st.sidebar.slider('Dukungan Sosial (skala 0-10)', 0.0, 10.0, 7.0)

    # Recalculate academic_stress_index and screen_sleep_ratio based on user input
    academic_stress_index = (0.4 * study_load + 0.4 * exam_anxiety - 0.2 * academic_performance)

    # Handle division by zero for sleep_duration
    sleep_duration_safe = sleep_duration if sleep_duration != 0 else 0.1
    screen_sleep_ratio = daily_screen_time / sleep_duration_safe

    data = {
        'sleep_duration': sleep_duration,
        'daily_screen_time': daily_screen_time,
        'study_load': study_load,
        'exam_anxiety': exam_anxiety,
        'academic_performance': academic_performance,
        'future_career_concerns': future_career_concerns,
        'teacher_student_relationship': teacher_student_relationship,
        'social_support': social_support,
        'academic_stress_index': academic_stress_index,
        'screen_sleep_ratio': screen_sleep_ratio
    }
    features = pd.DataFrame(data, index=[0])
    return features

df_input = user_input_features()

st.subheader('Input Pengguna')
st.write(df_input)

# Scale the input features
df_input_scaled = scaler.transform(df_input[selected_features])
df_input_scaled_df = pd.DataFrame(df_input_scaled, columns=selected_features)

# Make prediction
prediction = model.predict(df_input_scaled_df)
prediction_proba = model.predict_proba(df_input_scaled_df)

st.subheader('Hasil Prediksi Tingkat Stres')
stress_level_map = {0: 'Rendah', 1: 'Sedang', 2: 'Tinggi'}
predicted_stress_level = stress_level_map[prediction[0]]

st.write(f'Tingkat Stres yang Diprediksi: **{predicted_stress_level}**')
st.write('Probabilitas:')
for i, prob in enumerate(prediction_proba[0]):
    st.write(f'- {stress_level_map[i]}: {prob:.2f}')

# SHAP Explanation
st.subheader('Penjelasan Prediksi (SHAP Values)')

# Only compute SHAP values if the model is a tree-based model compatible with TreeExplainer
if hasattr(model, 'get_booster'): # For XGBoost
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(df_input_scaled_df)

    # Ensure shap_values is a list of arrays for multi-output models
    if isinstance(shap_values, list):
        # For multi-class classification, shap_values will be a list where each element is an array for a class
        # We need to select the shap values for the predicted class
        predicted_class_idx = prediction[0]
        shap_values_for_predicted_class = shap_values[predicted_class_idx][0]
        expected_value_for_predicted_class = explainer.expected_value[predicted_class_idx]
    else:
        # For binary classification or if TreeExplainer returns a single array
        shap_values_for_predicted_class = shap_values[0]
        expected_value_for_predicted_class = explainer.expected_value

    fig, ax = plt.subplots(figsize=(10, 6))
    shap.force_plot(expected_value_for_predicted_class, shap_values_for_predicted_class, df_input_scaled_df.iloc[0], matplotlib=True, show=False)
    st.pyplot(fig)

    st.write('Grafik di atas menunjukkan bagaimana setiap fitur berkontribusi pada prediksi tingkat stres. Warna merah menunjukkan fitur yang mendorong prediksi ke tingkat stres lebih tinggi, sedangkan warna biru mendorong ke tingkat stres lebih rendah.')
else:
    st.write('SHAP Explanation tidak tersedia untuk model ini dengan TreeExplainer.')
