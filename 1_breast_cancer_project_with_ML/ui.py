"""
Streamlit ile basit frontent yapacağız
Streamlit: bize python kodları ile çok basit frontent arayüzleri yapmamızı sağlar
"""

import streamlit as st
import numpy as np
import requests # server'ımıza request atacağımız için

st.title("Meme Kanseri Tahmin Uygulaması")
st.markdown("Bu uygulama, meme kanseri teşhisi için bir makina öğrenimi modelini kullanır")

API_URL = "http://127.0.0.1:8000/predict"

feature_names = [ # meme kanserinin özellikleri/featureleri
    "radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean",
    "compactness_mean", "concavity_mean", "concave points_mean", "symmetry_mean",
    "fractal_dimension_mean", "radius_se", "texture_se", "perimeter_se", "area_se",
    "smoothness_se", "compactness_se", "concavity_se", "concave points_se",
    "symmetry_se", "fractal_dimension_se", "radius_worst", "texture_worst",
    "perimeter_worst", "area_worst", "smoothness_worst", "compactness_worst",
    "concavity_worst", "concave points_worst", "symmetry_worst", "fractal_dimension_worst"
]

user_input=[] # kullanıcıdan gelen inputları kaydedeceğimiz yer

for name in feature_names:
    value = st.number_input(f"{name.replace('_', ' ').title()}", min_value=0.0, step=0.01)
    user_input.append(value)

# prediction button
if st.button("Tahmi et"):
    try:
        response = requests.post(API_URL, json={"data": user_input})
        if response.status_code == 200:
            result = response.json()
            prediction = result.get("prediction", "Tahmin yapılamadı")
            probability = result.get("probability", "Tahmin yapılamadı")

            st.subheader("Sonuç")
            if prediction == 0:
                st.success(f"Bu örnek muhtemelen **iyi huylu** (benign) bir tümördür")
            else:
                st.error(f"Bu örnek muhtemelen **kötü huylu** (malignant) bir tümördür")
            st.info(f"Modelin tahmin olasılığı: {probability}")
        else:
            st.error(f"Hata: {response.json()['detail']}")
    except requests.exceptions.RequestException as e:
        st.error(f"Bir hata oluştu: {e}")