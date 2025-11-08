import requests
import numpy as np
from PIL import Image, ImageOps # PIL: Python Image Library

API_URL = "http://127.0.0.1:8000/predict"

image_path = "images/img1.png"

img = Image.open(image_path).convert("L") # convert("L") : gri tonlama. Gray Scale
img = ImageOps.invert(img) # Invert colors if needed : gerekirse renkleri tersine çevir demek
img = img.resize((28,28)) # boyutu yeniden 28x28'e getir

# normalize and flatten the image
img_array = np.array(img, dtype=np.float32) / 255.0   # normalize işlemi 
flat_data = img_array.flatten().tolist()              # flatten işlemi

# API'a istek at
response = requests.post(API_URL, json={"pixels":flat_data})
if response.status_code==200:
    result = response.json()
    print(f"Predicted Digit: {result["prediction"]}, Confidence: {result["confidence"]}")
else:
    print(f"Hata: {response.status_code},  {response.text}")

    