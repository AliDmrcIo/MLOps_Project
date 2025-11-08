import streamlit as st
import numpy as np
import requests
from PIL import Image, ImageOps


st.title("MNIST Digit Recognition App")
st.write("Upload an image of handwritten digit (0-9) to get predictions")

uploaded_file = st.file_uploader("Choose an image....", type=["png", "jpg", "jpeg"])

if uploaded_file is not None: # eğer resim yüklendiyse, ona client_test'te olan aynı data preprocessing işlemlerini yap
    img = Image.open(uploaded_file).convert("L")  # Convert to grayscale
    img = ImageOps.invert(img)                   # Invert colors if needed
    img = img.resize((28, 28))                   # Resize to 28x28

    st.image(img, caption="Uploaded Image")


    # Normalize and flatten the image
    img_array = np.array(img, dtype=np.float32) / 255.0
    flat_data = img_array.flatten().tolist()

    # API URL
    API_URL = "http://127.0.0.1:8000/predict"

    if st.button("Predict"):
        with st.spinner("Processing...."):

            # Send request to the API - ön işlenmiş görüntüyü bir API'ye gönderip tahmini sonucu alma kısmı
            response = requests.post(API_URL, json={"pixels": flat_data})
            if response.status_code == 200:
                result = response.json()
                prediction = result.get("prediction", "N/A")
                confidence = result.get("confidence", "N/A")
                st.markdown("### Prediction Results ###")
                st.write(f"Predicted Digit: {prediction}, Confidence: {confidence}")
                st.progress(confidence)
            else:
                st.write(f"Error: {response.status_code}, {response.text}")