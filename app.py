
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler # Ensure StandardScaler is imported if used

# --- 1. Muat Model dan Scaler yang Tersimpan ---
model_filename = 'gradient_boosting_model.pkl'
scaler_filename = 'standard_scaler.pkl'

@st.cache_resource # Cache the model loading for better performance
def load_model_and_scaler():
    with open(model_filename, 'rb') as file:
        loaded_model = pickle.load(file)
    with open(scaler_filename, 'rb') as file:
        loaded_scaler = pickle.load(file)
    return loaded_model, loaded_scaler

loaded_model, loaded_scaler = load_model_and_scaler()

# --- 2. Fungsi Preprocessing (Harus sama dengan saat training) ---
def preprocess_input(input_df, scaler):
    processed_new_df = input_df.copy()

    # Rekonstruksi `list_label_enc` dan `list_one_hot` dari training
    list_label_enc = ['Pendidikan', 'Jurusan']
    list_one_hot = ['Jenis_Kelamin', 'Status_Bekerja']

    # Mapping untuk Pendidikan dan Jurusan (sesuai urutan LabelEncoder saat training)
    pendidikan_mapping = {'D3': 0, 'S1': 1, 'SMA': 2, 'SMK': 3}
    jurusan_mapping = {'Administrasi': 0, 'Desain Grafis': 1, 'Otomotif': 2, 'Teknik Las': 3, 'Teknik Listrik': 4}

    # Apply Label Encoding
    processed_new_df['Pendidikan'] = processed_new_df['Pendidikan'].map(pendidikan_mapping)
    processed_new_df['Jurusan'] = processed_new_df['Jurusan'].map(jurusan_mapping)
    
    # Handle potential missing values from mapping for robustness (e.g., if new category appears)
    # For this specific dataset, we assume input will match known categories.
    # In a real app, you might want to handle `NaN` values here or reject invalid inputs.
    
    # One-Hot Encoding
    df_onehot_new = pd.get_dummies(processed_new_df[list_one_hot], prefix=list_one_hot)
    df_onehot_new = df_onehot_new.astype(int)

    # Ensure all expected one-hot columns are present, fill with 0 if not
    expected_onehot_cols = [
        'Jenis_Kelamin_Laki-laki',
        'Jenis_Kelamin_Wanita',
        'Status_Bekerja_Belum Bekerja',
        'Status_Bekerja_Sudah Bekerja'
    ]
    for col in expected_onehot_cols:
        if col not in df_onehot_new.columns:
            df_onehot_new[col] = 0
    df_onehot_new = df_onehot_new[expected_onehot_cols]
    
    # Combine all features
    kolom_numerik = ['Usia', 'Durasi_Jam', 'Nilai_Ujian'] # from df_bersih
    processed_features = pd.concat([
        processed_new_df[kolom_numerik],
        processed_new_df[list_label_enc], # Add label encoded columns back
        df_onehot_new
    ], axis=1)
    
    # Define feature_cols for consistent column order and scaling
    feature_cols = ['Usia', 'Durasi_Jam', 'Nilai_Ujian', 'Pendidikan', 'Jurusan',
                    'Jenis_Kelamin_Laki-laki', 'Jenis_Kelamin_Wanita',
                    'Status_Bekerja_Belum Bekerja', 'Status_Bekerja_Sudah Bekerja']
    
    processed_features = processed_features[feature_cols]

    # Scaling features
    scaled_features = scaler.transform(processed_features)
    scaled_features_df = pd.DataFrame(scaled_features, columns=feature_cols)

    return scaled_features_df

# --- 3. Streamlit App Interface ---
st.title('Prediksi Gaji Pertama Peserta Vokasi')
st.write('Aplikasi ini memprediksi gaji pertama (dalam juta Rupiah) berdasarkan data peserta pelatihan vokasi.')

# Input form
with st.form('prediction_form'):
    st.header('Input Data Peserta')

    usia = st.slider('Usia', 18, 60, 25)
    durasi_jam = st.slider('Durasi Pelatihan (Jam)', 20, 100, 60)
    nilai_ujian = st.slider('Nilai Ujian', 50.0, 100.0, 75.0, 0.1)
    
    pendidikan = st.selectbox('Pendidikan', ['D3', 'S1', 'SMA', 'SMK'])
    jurusan = st.selectbox('Jurusan', ['Administrasi', 'Desain Grafis', 'Otomotif', 'Teknik Las', 'Teknik Listrik'])
    jenis_kelamin = st.selectbox('Jenis Kelamin', ['Laki-laki', 'Wanita', 'Pria'])
    status_bekerja = st.selectbox('Status Bekerja', ['Sudah Bekerja', 'Belum Bekerja'])

    submitted = st.form_submit_button('Prediksi Gaji')

    if submitted:
        # Create DataFrame from inputs
        new_data = {
            'Usia': [usia],
            'Durasi_Jam': [durasi_jam],
            'Nilai_Ujian': [nilai_ujian],
            'Pendidikan': [pendidikan],
            'Jurusan': [jurusan],
            'Jenis_Kelamin': [jenis_kelamin],
            'Status_Bekerja': [status_bekerja]
        }
        input_df = pd.DataFrame(new_data)

        # Preprocess input data
        processed_input = preprocess_input(input_df, loaded_scaler)

        # Make prediction
        prediction = loaded_model.predict(processed_input)
        
        st.success(f'Prediksi Gaji Pertama: **{prediction[0]:.2f} Juta Rupiah**')
