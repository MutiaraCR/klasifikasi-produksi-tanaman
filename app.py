import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px

st.set_page_config(page_title="Prediksi Produksi Pertanian", page_icon="🌾", layout="wide")

svm_model = joblib.load("best_svm.pkl")
scaler = joblib.load("scaler.pkl")

def predict_tingkat_produksi(df):
    df = df.rename(columns={
        "Produksi per satuan luas": "Produksi per Satuan Luas",
        "Total produksi": "Total Produksi"
    })
    required_features = ["Produksi per Satuan Luas", "Total Produksi"]
    if not all(feature in df.columns for feature in required_features):
        st.error("❌ Kolom yang dibutuhkan tidak ditemukan dalam dataset!")
        return None
    df_scaled = scaler.transform(df[required_features])
    df["Tingkat Produksi"] = svm_model.predict(df_scaled)
    return df

st.markdown("""
    <h1 style='text-align: center; color: #FFD700;'>🌾 Prediksi Produksi Pertanian 🚜</h1>
    <hr style='border-color: #00FF00;'>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("📂 **Upload file Excel**", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    df = predict_tingkat_produksi(df)

    if df is not None:
        selected_columns = ["Kecamatan", "Total Produksi", "Produksi per Satuan Luas", "Tingkat Produksi"]
        df_display = df[selected_columns]

        st.markdown("## 📊 Hasil Prediksi")
        st.success("✅ Prediksi berhasil dilakukan! Berikut hasilnya:")
        st.dataframe(df_display.style.applymap(lambda x: "color: #90EE90;" if x == "Tinggi" else "color: #FFD700;" if x == "Sedang" else "color: #FF6F61;", subset=["Tingkat Produksi"]))

        col1, col2 = st.columns([1, 2])

        with col1:
            st.markdown("## 🎨 Distribusi Kategori Produksi")
            kategori_counts = df_display["Tingkat Produksi"].value_counts()
            kategori_labels = kategori_counts.index
            kategori_values = kategori_counts.values

            fig = px.pie(names=kategori_labels, values=kategori_values, color=kategori_labels,
                         color_discrete_map={"Tinggi": "#90EE90", "Sedang": "#FFD700", "Rendah": "#FF6F61"},
                         hole=0.4)
            st.plotly_chart(fig)

        with col2:
            st.markdown("## ℹ️ Keterangan Jumlah Kategori")
            st.markdown(f"""
                **Jumlah Kategori:**
                - 🟩 **Tinggi:** {kategori_counts.get('Tinggi', 0)} data
                - 🟨 **Sedang:** {kategori_counts.get('Sedang', 0)} data
                - 🟥 **Rendah:** {kategori_counts.get('Rendah', 0)} data
            """)

        # Barchart
        st.markdown("## 📊 Total Produksi per Kecamatan")
        df_sorted = df_display.sort_values(by="Total Produksi", ascending=True)
        fig = px.bar(df_sorted, x="Total Produksi", y="Kecamatan", orientation='h',
                         color="Total Produksi", color_continuous_scale=["#FFF700", "#FFB400", "#FF8800"],
                         title="Total Produksi per Kecamatan")
        st.plotly_chart(fig)

        st.markdown("## 📉 Produksi per Satuan Luas per Kecamatan")
        df_sorted_luas = df_display.sort_values(by="Produksi per Satuan Luas", ascending=True)
        fig_luas = px.bar(df_sorted_luas, x="Produksi per Satuan Luas", y="Kecamatan", orientation='h',
                          color="Produksi per Satuan Luas", color_continuous_scale="blues",
                          title="Produksi per Satuan Luas per Kecamatan")
        st.plotly_chart(fig_luas)
