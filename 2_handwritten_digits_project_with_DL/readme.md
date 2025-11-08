## TR:
# Proje 2: El Yazısı Rakam Tanıma (Derin Öğrenme ile MLOps)

Bu proje, MLOps yaşam döngüsünü, bir derin öğrenme (Deep Learning) ve bilgisayarlı görü (Computer Vision) problemi üzerinden uygulamalı olarak göstermektedir. TensorFlow/Keras ile MNIST veri seti üzerinde eğitilmiş bir Evrişimli Sinir Ağı (CNN) modeli, FastAPI ile API olarak sunulmakta ve Streamlit arayüzü ile kullanıcıdan alınan görselleri tanımaktadır.

---

## Projenin Amacı

Kullanıcının yüklediği bir resimdeki el yazısı rakamı (0-9) tanıyan bir sistem oluşturmaktır.

## 🛠️ Kullanılan Teknolojiler

- **Model:** `TensorFlow/Keras` (Convolutional Neural Network - CNN)
- **Veri Seti:** `MNIST` Handwritten Digits
- **Backend (API Sunucusu):** `FastAPI`
- **Frontend (Kullanıcı Arayüzü):** `Streamlit`
- **Model Kaydı:** `H5 Format` (`.h5`)
- **Görüntü İşleme:** `Pillow (PIL)`, `NumPy`

## 📂 Dosya Yapısı ve Görevleri

-   **`model_train.py`**: MNIST veri seti ile bir CNN modeli oluşturur, eğitir ve `mnist_model.h5` olarak kaydeder.
-   **`main.py`**: `FastAPI` sunucusu oluşturur. `.h5` modelini yükler ve 28x28 piksel boyutundaki görüntü verisini işleyerek tahmin yapan bir `/predict` endpoint'i sunar.
-   **`ui.py`**: `Streamlit` ile bir web arayüzü oluşturur. Kullanıcının bir rakam resmi yüklemesine izin verir. Yüklenen resim, modelin beklediği formata (gri tonlama, 28x28 boyut, normalizasyon) dönüştürülür ve FastAPI'ye gönderilir.
-   **`client_test.py`**: `images/` klasöründeki bir test görüntüsünü işleyerek API'yi programatik olarak test eder.
-   **`images/`**: Test için kullanılacak örnek resimleri içerir.

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
     uvicorn 2_handwritten_digits_project_with_DL.main:app --reload
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
     streamlit run 2_handwritten_digits_project_with_DL/ui.py
     ```
   - Tarayıcınızda otomatik olarak yeni bir sekme açılacak ve uygulamayı görebileceksiniz.

---
---

## ENG:
# Project 2: Handwritten Digit Recognition (MLOps with Deep Learning)

This project demonstrates the MLOps lifecycle through a deep learning and computer vision problem. A Convolutional Neural Network (CNN) model, trained on the MNIST dataset with TensorFlow/Keras, is served as an API via FastAPI and recognizes digits from images uploaded by the user through a Streamlit interface.

---

## Project Goal

To create a system that recognizes a handwritten digit (0-9) from an image uploaded by the user.

## 🛠️ Technologies Used

- **Model:** `TensorFlow/Keras` (Convolutional Neural Network - CNN)
- **Dataset:** `MNIST` Handwritten Digits
- **Backend (API Server):** `FastAPI`
- **Frontend (User Interface):** `Streamlit`
- **Model Serialization:** `H5 Format` (`.h5`)
- **Image Processing:** `Pillow (PIL)`, `NumPy`

## 📂 File Structure and Roles

-   **`model_train.py`**: Creates, trains, and saves a CNN model with the MNIST dataset as `mnist_model.h5`.
-   **`main.py`**: Creates a `FastAPI` server. It loads the `.h5` model and serves a `/predict` endpoint that processes 28x28 pixel image data to make a prediction.
-   **`ui.py`**: Creates a web interface with `Streamlit`. It allows the user to upload a digit image. The uploaded image is preprocessed into the format expected by the model (grayscale, 28x28, normalization) and sent to the FastAPI server.
-   **`client_test.py`**: Programmatically tests the API by processing a test image from the `images/` folder.
-   **`images/`**: Contains sample images for testing.

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
     uvicorn 2_handwritten_digits_project_with_DL.main:app --reload
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
     streamlit run 2_handwritten_digits_project_with_DL/ui.py
     ```
   - A new tab will automatically open in your browser where you can see and interact with the application.
