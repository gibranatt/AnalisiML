import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Konfigurasi halaman web
st.set_page_config(page_title="Analisis Sentimen ML", page_icon="📊")

st.title("📊 Sistem Analisis Sentimen Ulasan")
st.write("Aplikasi Machine Learning menggunakan algoritma **TF-IDF** dan **Multinomial Naive Bayes**.")

# 1. Load Dataset & Training Model (di-cache agar cepat)
@st.cache_resource
def load_and_train():
    df = pd.read_csv("datasetml.csv", sep=';')
    
    # Split data latih (80%) dan data uji (20%)
    X_train, X_test, y_train, y_test = train_test_split(
        df["ulasan"], df["sentimen"], test_size=0.2, random_state=42
    )
    
    # Ekstraksi fitur teks (TF-IDF)
    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    # Melatih model Naive Bayes
    model = MultinomialNB()
    model.fit(X_train_tfidf, y_train)
    
    # Hitung akurasi pada data uji
    y_pred = model.predict(X_test_tfidf)
    akurasi = accuracy_score(y_test, y_pred) * 100
    
    return df, vectorizer, model, akurasi

df, vectorizer, model, akurasi = load_and_train()

# Sidebar untuk informasi dataset & model
with st.sidebar:
    st.header("ℹ️ Informasi Model")
    st.metric(label="Akurasi Data Uji", value=f"{akurasi:.1f}%")
    st.write(f"**Total Dataset:** {len(df)} ulasan")
    st.write("**Algoritma:** Multinomial Naive Bayes")
    if st.checkbox("Lihat Sampel Dataset"):
        st.dataframe(df.head(10))

# 2. Area Input Pengguna
st.subheader("Uji Coba Analisis Teks")
input_teks = st.text_area(
    "Masukkan kalimat ulasan produk/layanan:",
    placeholder="Contoh: Barangnya bagus banget, pengiriman juga cepat dan kurir ramah!"
)

if st.button("Analisis Sentimen"):
    if input_teks.strip() == "":
        st.warning("⚠️ Silakan masukkan teks ulasan terlebih dahulu.")
    else:
        # Ubah teks input ke vektor TF-IDF dan prediksi
        teks_tfidf = vectorizer.transform([input_teks])
        prediksi = model.predict(teks_tfidf)[0]
        probabilitas = model.predict_proba(teks_tfidf)[0]
        
        # Ambil persentase keyakinan tertinggi
        skor_keyakinan = max(probabilitas) * 100
        
        st.divider()
        st.subheader("Hasil Analisis:")
        
        if prediksi == "Positif":
            st.success(f"**Sentimen: POSITIF** (Tingkat Keyakinan: **{skor_keyakinan:.2f}%**)")
        else:
            st.error(f"**Sentimen: NEGATIF** (Tingkat Keyakinan: **{skor_keyakinan:.2f}%**)")
            
        # Tampilkan detail probabilitas
        st.write("**Rincian Probabilitas:**")
        kelas = model.classes_
        for k, p in zip(kelas, probabilitas):
            st.progress(float(p), text=f"{k}: {p*100:.2f}%")