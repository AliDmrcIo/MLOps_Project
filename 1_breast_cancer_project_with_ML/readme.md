## TR:
# Proje 1: Meme Kanseri Tahmini (Klasik Makine Öğrenmesi ile MLOps)

Bu proje, MLOps yaşam döngüsünün temel adımlarını, klasik bir makine öğrenmesi problemi üzerinden uygulamalı olarak göstermektedir. Scikit-learn ile eğitilmiş bir `RandomForestClassifier` modeli, FastAPI ile bir API olarak sunulmakta ve Streamlit ile oluşturulan basit bir arayüz üzerinden tahmin yapmaktadır.

---

## Projenin Amacı

Hastaya ait 30 farklı tıbbi özelliğe dayanarak, bir meme kanseri tümörünün **iyi huylu (benign)** mu yoksa **kötü huylu (malignant)** mu olduğunu tahmin eden bir sistem oluşturmaktır.

## 🛠️ Kullanılan Teknolojiler

- **Model:** `Scikit-learn` (RandomForestClassifier)
- **Veri Seti:** Scikit-learn `breast_cancer`
- **Backend (API Sunucusu):** `FastAPI`
- **Frontend (Kullanıcı Arayüzü):** `Streamlit`
- **Model Kaydı:** `Joblib` (`.pkl`)

## 📂 Dosya Yapısı ve Görevleri

-   **`model_train.py`**: Modeli `breast_cancer` veri seti ile eğitir ve `breast_cancer_model.pkl` olarak kaydeder.
-   **`main.py`**: `FastAPI` kullanarak bir web sunucusu oluşturur. Kaydedilen `.pkl` modelini yükler ve gelen veri için tahmin yapan bir `/predict` endpoint'i sunar.
-   **`ui.py`**: `Streamlit` ile bir web arayüzü oluşturur. Kullanıcının 30 farklı tıbbi özelliği girmesine olanak tanır ve FastAPI'deki `/predict` endpoint'ine istek atarak sonucu gösterir.
-   **`client_test.py`**: FastAPI sunucusunun doğru çalışıp çalışmadığını test etmek için `/predict` endpoint'ine programatik olarak örnek bir istek gönderir.

## Nasıl Çalıştırılır?

Bu projeyi çalıştırmak için iki ayrı terminal penceresine ihtiyacınız olacaktır: biri backend (FastAPI) için, diğeri frontend (Streamlit) için.

**Ön Koşullar:**
- Ana `MLOps_Project` klasöründe kurulum adımlarını tamamladığınızdan ve sanal ortamın (`venv`) aktif olduğundan emin olun.

**1. Adım: Backend Sunucusunu Başlatma**
   - Yeni bir terminal açın.
   - Ana `MLOps_Project` dizinine gidin ve sanal ortamı aktif edin:
     ```bash
     cd path/to/MLOps_Project
     .\venv\Scripts\activate
     ```
   - FastAPI sunucusunu çalıştırın:
     ```bash
     uvicorn 1_breast_cancer_project_with_ML.main:app --reload
     ```
   - Sunucu şimdi `http://127.0.0.1:8000` adresinde çalışıyor olmalı.

**2. Adım: Frontend Arayüzünü Başlatma**
   - **İkinci** bir terminal açın.
   - Yine ana `MLOps_Project` dizinine gidin ve sanal ortamı aktif edin:
     ```bash
     cd path/to/MLOps_Project
     .\venv\Scripts\activate
     ```
   - Streamlit arayüzünü çalıştırın:
     ```bash
     streamlit run 1_breast_cancer_project_with_ML/ui.py
     ```
   - Tarayıcınızda otomatik olarak yeni bir sekme açılacak ve uygulamayı görebileceksiniz.

---
---

## ENG:
# Project 1: Breast Cancer Prediction (MLOps with Classic Machine Learning)

This project demonstrates the fundamental steps of the MLOps lifecycle through a classic machine learning problem. A `RandomForestClassifier` model trained with Scikit-learn is served as an API via FastAPI and makes predictions through a simple user interface created with Streamlit.

---

## Project Goal

To create a system that predicts whether a breast cancer tumor is **benign** or **malignant** based on 30 different medical features belonging to a patient.

## 🛠️ Technologies Used

- **Model:** `Scikit-learn` (RandomForestClassifier)
- **Dataset:** Scikit-learn `breast_cancer`
- **Backend (API Server):** `FastAPI`
- **Frontend (User Interface):** `Streamlit`
- **Model Serialization:** `Joblib` (`.pkl`)

## 📂 File Structure and Roles

-   **`model_train.py`**: Trains the model with the `breast_cancer` dataset and saves it as `breast_cancer_model.pkl`.
-   **`main.py`**: Creates a web server using `FastAPI`. It loads the saved `.pkl` model and serves a `/predict` endpoint that makes predictions on incoming data.
-   **`ui.py`**: Creates a web interface with `Streamlit`. It allows the user to input 30 different medical features and displays the result by sending a request to the `/predict` endpoint on FastAPI.
-   **`client_test.py`**: Sends a programmatic sample request to the `/predict` endpoint to test if the FastAPI server is working correctly.

## How to Run

You will need two separate terminal windows to run this project: one for the backend (FastAPI) and one for the frontend (Streamlit).

**Prerequisites:**
- Ensure you have completed the setup steps in the main `MLOps_Project` directory and that the virtual environment (`venv`) is activated.

**Step 1: Start the Backend Server**
   - Open a new terminal.
   - Navigate to the main `MLOps_Project` directory and activate the virtual environment:
     ```bash
     cd path/to/MLOps_Project
     .\venv\Scripts\activate
     ```
   - Run the FastAPI server:
     ```bash
     uvicorn 1_breast_cancer_project_with_ML.main:app --reload
     ```
   - The server should now be running at `http://127.0.0.1:8000`.

**Step 2: Start the Frontend Interface**
   - Open a **second** terminal.
   - Again, navigate to the main `MLOps_Project` directory and activate the virtual environment:
     ```bash
     cd path/to/MLOps_Project
     .\venv\Scripts\activate
     ```
   - Run the Streamlit interface:
     ```bash
     streamlit run 1_breast_cancer_project_with_ML/ui.py
     ```
   - A new tab will automatically open in your browser where you can see and interact with the application.
