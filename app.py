
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# --- 1. Muat Model dan Scaler yang Tersimpan ---
model_filename = 'gradient_boosting_model.pkl'
scaler_filename = 'standard_scaler.pkl'

@st.cache_resource
def load_model_and_scaler():
    try:
        with open(model_filename, 'rb') as file:
            loaded_model = pickle.load(file)
        with open(scaler_filename, 'rb') as file:
            loaded_scaler = pickle.load(file)
        return loaded_model, loaded_scaler
    except FileNotFoundError:
        st.error("Model atau Scaler tidak ditemukan. Pastikan file 'gradient_boosting_model.pkl' dan 'standard_scaler.pkl' ada di direktori yang sama.")
        st.stop()
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat model atau scaler: {e}")
        st.stop()

loaded_model, loaded_scaler = load_model_and_scaler()

# --- 2. Definisi Mappings dan Kolom ---
# Mappings untuk Label Encoding (harus konsisten dengan saat training)
pendidikan_mapping = {'D3': 0, 'S1': 1, 'SMA': 2, 'SMK': 3}
jurusan_mapping = {'Administrasi': 0, 'Desain Grafis': 1, 'Otomotif': 2, 'Teknik Las': 3, 'Teknik Listrik': 4}

# Mapping untuk Jenis_Kelamin sebelum One-Hot Encoding
gender_replace_mapping = {'Pria': 'Laki-laki', 'L': 'Laki-laki', 'Perempuan': 'Wanita', 'P': 'Wanita'}

# List kolom kategorikal yang di-label encode
list_label_enc = ['Pendidikan', 'Jurusan']
# List kolom kategorikal yang di-one-hot encode
list_one_hot = ['Jenis_Kelamin', 'Status_Bekerja']

# Semua kolom fitur yang diharapkan oleh model (urutan harus sama dengan x_train)
feature_cols = ['Usia', 'Durasi_Jam', 'Nilai_Ujian', 'Pendidikan', 'Jurusan',
                'Jenis_Kelamin_Laki-laki', 'Jenis_Kelamin_Wanita',
                'Status_Bekerja_Belum Bekerja', 'Status_Bekerja_Sudah Bekerja']

# --- 3. Fungsi Preprocessing Data Input Baru ---
def preprocess_input(input_df_raw):
    processed_df = input_df_raw.copy()

    # 1. Apply gender replacement mapping (consistency with notebook's df.replace())
    if 'Jenis_Kelamin' in processed_df.columns:
        processed_df['Jenis_Kelamin'] = processed_df['Jenis_Kelamin'].replace(gender_replace_mapping)

    # 2. Label Encoding
    for col in list_label_enc:
        if col == 'Pendidikan':
            processed_df[col] = processed_df[col].map(pendidikan_mapping)
        elif col == 'Jurusan':
            processed_df[col] = processed_df[col].map(jurusan_mapping)
        if processed_df[col].isnull().any():
            st.warning(f"Kategori tidak dikenal di '{col}'. Silakan periksa input.")
            return None # Indicate preprocessing failure

    # 3. One-Hot Encoding
    df_onehot_processed = pd.get_dummies(processed_df[list_one_hot], prefix=list_one_hot)
    df_onehot_processed = df_onehot_processed.astype(int)

    # Ensure all expected one-hot columns are present (add with 0 if missing)
    expected_onehot_columns_from_training = [
        'Jenis_Kelamin_Laki-laki', 'Jenis_Kelamin_Wanita',
        'Status_Bekerja_Belum Bekerja', 'Status_Bekerja_Sudah Bekerja'
    ]
    for col in expected_onehot_columns_from_training:
        if col not in df_onehot_processed.columns:
            df_onehot_processed[col] = 0

    # Drop original categorical columns
    processed_df = processed_df.drop(columns=list_one_hot)

    # Concatenate all features
    final_features = pd.concat([processed_df, df_onehot_processed], axis=1)

    # Ensure feature columns match the order and names from training
    final_features = final_features[feature_cols]

    # 4. Scaling numerical features
    scaled_features = loaded_scaler.transform(final_features)
    scaled_features_df = pd.DataFrame(scaled_features, columns=feature_cols)

    return scaled_features_df

# --- 4. Antarmuka Streamlit ---
st.title('💰 Prediksi Gaji Awal Lulusan Pelatihan Vokasi')
st.write('Aplikasi ini memprediksi estimasi gaji awal lulusan berdasarkan data pelatihan vokasi.')

# Input fields dari user
with st.form("prediction_form"):
    st.header("Informasi Peserta")
    col1, col2 = st.columns(2)
    with col1:
        usia = st.slider('Usia', 18, 60, 25)
        durasi_jam = st.slider('Durasi Pelatihan (Jam)', 20, 100, 50)
        pendidikan = st.selectbox('Pendidikan', options=list(pendidikan_mapping.keys()))
        jenis_kelamin = st.selectbox('Jenis Kelamin', options=['Laki-laki', 'Wanita']) # Opsi 'Pria' dihapus di sini
    with col2:
        nilai_ujian = st.slider('Nilai Ujian', 0.0, 100.0, 75.0)
        jurusan = st.selectbox('Jurusan', options=list(jurusan_mapping.keys()))
        status_bekerja = st.selectbox('Status Bekerja', options=['Sudah Bekerja', 'Belum Bekerja'])

    submitted = st.form_submit_button("Prediksi Gaji")

    if submitted:
        # Create DataFrame from input
        input_data_raw = pd.DataFrame([[usia, durasi_jam, nilai_ujian, pendidikan, jurusan, jenis_kelamin, status_bekerja]],
                                      columns=['Usia', 'Durasi_Jam', 'Nilai_Ujian', 'Pendidikan', 'Jurusan', 'Jenis_Kelamin', 'Status_Bekerja'])

        # Preprocess the raw input data
        final_input_for_prediction = preprocess_input(input_data_raw)

        if final_input_for_prediction is not None:
            # Make prediction
            prediction = loaded_model.predict(final_input_for_prediction)[0]
            st.success(f'### Prediksi Gaji Awal: Rp {prediction * 1_000_000:,.2f}') # Convert to actual Rupiah value
            st.balloons()
