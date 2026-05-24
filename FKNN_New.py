import streamlit as st
import numpy as np
import pandas as pd
import operator
import pickle
from sklearn.base import BaseEstimator, ClassifierMixin

# ============================================================
# DEFINISI CLASS AGAR PICKLE BISA MEMBACA MODEL
# ============================================================

class FuzzyKNN(BaseEstimator, ClassifierMixin):
    def __init__(self, k=7, m=1.5):
        self.k = k
        self.m = m

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.n = len(y)
        self.classes = np.unique(y)
        self.memberships = self._compute_memberships()
        return self

    def _find_k_nearest_neighbors(self, x):
        distances = np.linalg.norm(self.X - x, axis=1)
        idx = np.argsort(distances)
        return idx[:self.k]

    def _compute_memberships(self):
        memberships = []
        for i in range(self.n):
            x = self.X[i]
            y = self.y[i]
            neighbor_idx = self._find_k_nearest_neighbors(x)
            neighbor_classes = self.y[neighbor_idx]
            membership = {}
            for c in self.classes:
                count = np.sum(neighbor_classes == c)
                uci = 0.49 * (count / self.k)
                if c == y:
                    uci += 0.51
                membership[c] = uci
            memberships.append(membership)
        return memberships

    def predict(self, X):
        predictions = []
        for x in X:
            neighbor_idx = self._find_k_nearest_neighbors(x)
            votes = {}
            for c in self.classes:
                numerator = 0
                denominator = 0
                for idx in neighbor_idx:
                    dist = np.linalg.norm(x - self.X[idx])
                    dist = max(dist, 1e-10)
                    weight = 1 / (dist ** (2 / (self.m - 1)))
                    numerator += self.memberships[idx][c] * weight
                    denominator += weight
                votes[c] = numerator / denominator
            pred = max(votes.items(), key=operator.itemgetter(1))[0]
            predictions.append(pred)
        return np.array(predictions)

# ============================================================
# MEMUAT ARTIFAK MODEL
# ============================================================

@st.cache_resource
def load_model():
    # Membuka berkas pkl yang diletakkan satu folder dengan app.py
    with open('fknn_fish_model.pkl', 'rb') as f:
        data = pickle.load(f)
    return data['model'], data['scaler'], data['encoder']

try:
    model, scaler, le = load_model()
except FileNotFoundError:
    st.error("Berkas 'fknn_fish_model.pkl' tidak ditemukan di direktori kerja.")
    st.stop()

# ============================================================
# ANTARMUKA PENGGUNA (STREAMLIT UI)
# ============================================================

st.set_page_config(page_title="Prediksi Spesies Ikan FKNN", layout="centered", page_icon="🐟")

st.title("🐟 Aplikasi Prediksi Spesies Ikan")
st.write("Aplikasi ini memprediksi spesies ikan menggunakan algoritma **Fuzzy K-Nearest Neighbors (FKNN)** berdasarkan parameter fisik.")
st.markdown("---")

st.subheader("Input Karakteristik Fisik Ikan")

# Membuat tata letak dua kolom untuk input numerik
col1, col2 = st.columns(2)

with col1:
    weight = st.number_input("Weight (Berat - gram)", min_value=0.0, value=340.0, step=1.0)
    length1 = st.number_input("Length1 (Panjang Vertikal - cm)", min_value=0.0, value=23.9, step=0.1)
    length2 = st.number_input("Length2 (Panjang Diagonal - cm)", min_value=0.0, value=26.2, step=0.1)

with col2:
    length3 = st.number_input("Length3 (Panjang Silang - cm)", min_value=0.0, value=31.1, step=0.1)
    height = st.number_input("Height (Tinggi - cm)", min_value=0.0, value=12.3, step=0.1)
    width = st.number_input("Width (Lebar - cm)", min_value=0.0, value=4.6, step=0.1)

st.markdown("---")

# Tombol Eksekusi Prediksi
if st.button("Prediksi Spesies", type="primary"):
    # Gabungkan input menjadi array matriks 2D
    data_baru = np.array([[weight, length1, length2, length3, height, width]])
    
    # Transformasi menggunakan objek scaler latih
    data_terstandar = scaler.transform(data_baru)
    
    # Prediksi kode kelas numerik
    kode_prediksi = model.predict(data_terstandar)
    
    # Kembalikan kode angka menjadi teks spesies asli
    nama_spesies = le.inverse_transform(kode_prediksi)[0]
    
    # Menampilkan hasil ke layar
    st.success(f"### Hasil Prediksi: Ikan ini termasuk spesies **{nama_spesies}**")
    
    # Tampilkan parameter model yang bekerja di latar belakang
    st.info(f"Model Klasifikasi: Fuzzy KNN (Konfigurasi Optimal GridSearch: k={model.k}, m={model.m})")