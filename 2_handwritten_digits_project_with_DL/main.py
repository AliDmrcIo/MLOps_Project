from fastapi import FastAPI
from pydantic import BaseModel

import numpy as np

from tensorflow.keras.models import load_model

import warnings
warnings.filterwarnings("ignore")

app = FastAPI(title="MNIST Digit Recognition API", 
              description="This api predicts handwritten digits using a pretrained model",
              version="1.0.0")

# modelimizi yükleyelim
model = load_model("mnist_model.h5")

class ImageInput(BaseModel):
    pixels: list = [] # 28x28 boyutunda flatten edilmiş ve 784 yapılmış hali olan liste

@app.post("/predict")
def predict(data:ImageInput):
    """Predict Handwritten Digit from Image Pixels"""

    try:
        img_array = np.array(data.pixels, dtype=np.float32).reshape(1, 28, 28, 1)
        
        prediction = model.predict(img_array)[0]

        predicted_class = int(np.argmax(prediction))

        return {"prediction":predicted_class, "confidence":float(np.max(prediction))}
    
    except Exception as e:
        return f"Hata: {str(e)}"