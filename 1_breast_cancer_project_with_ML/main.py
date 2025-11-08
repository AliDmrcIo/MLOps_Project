"""
2.Dosya: FastAPI işlemleri
"""

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title = "Breast Cancer Prediction API", version = "1.0")

model = joblib.load("breast_cancer_model.pkl") # model_train.py'de oluşturduğumuz modeli burada yükle

class CancerInput(BaseModel): # modele vermek için gelen veri
    data: list

@app.post("/predict") # burada modeli kullanarak, veri verilip tahmin yapılacak
def predict(input: CancerInput):
    try:
        features = np.array(input.data).reshape(1, -1)
        prediction = model.predict(features)[0] # kanser mi değilmiyi söylicek
        probability = model.predict_proba(features)[0].tolist() # probabilityleri(olasılıkları) tahmin edeceğiz
        return {
            "Prediction":int(prediction), # 0 ya da 1. 0: iyi huylu, 1: kötü huylu
            "probability":probability
        }
    except Exception as e:
        return f"Hata: {str(e)}"