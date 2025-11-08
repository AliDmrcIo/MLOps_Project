## TR:
# Pratik MLOps Proje Koleksiyonu

Bu repository, MLOps (Machine Learning Operations) prensiplerini üç farklı yapay zeka alanında uygulamalı olarak gösteren bir proje koleksiyonu içermektedir. Her bir proje, modelin eğitilmesinden başlayarak FastAPI ile bir API olarak sunulmasına, Streamlit ile bir kullanıcı arayüzü oluşturulmasına ve test edilmesine kadar olan tüm yaşam döngüsünü kapsar.

## Projeler ve Kullanılan Teknolojiler

Bu koleksiyon, yapay zekanın üç farklı dalını ele alan projelerden oluşur:

| Proje Adı | Alan | Ana Kütüphaneler | Model Dosyası |
| :--- | :--- | :--- | :--- |
| **1. Meme Kanseri Tahmini** | Klasik Makine Öğrenmesi (ML) | `Scikit-learn` | `.pkl` |
| **2. El Yazısı Rakam Tanıma** | Derin Öğrenme (DL) | `TensorFlow/Keras` | `.h5` |
| **3. LLM Destekli Chatbot** | Büyük Dil Modelleri (LLM) | `LangChain`, `Ollama` | (Yerel Model) |

**Ortak Teknoloji Yığını:**
- **Backend (API Sunucusu):** `FastAPI`
- **Frontend (Kullanıcı Arayüzü):** `Streamlit`
- **Model Serileştirme:** `Joblib`, `H5`

## Kurulum ve Başlangıç

Projeleri çalıştırmak için aşağıdaki genel adımları izleyin. Her projenin kendine özgü çalıştırma komutları ilgili başlık altında verilmiştir.

1.  **Repository'yi Klonlayın:**
    ```bash
    git clone https://github.com/AliDmrcIo/MLOps_Project.git
    cd MLOps_Project
    ```

2.  **Sanal Ortamı Aktif Edin:**
    Proje, gerekli kütüphaneleri içeren bir `venv` sanal ortamı ile birlikte gelir.
    ```bash
    # Windows için
    .\venv\Scripts\activate
    ```

3.  **Gerekli Kütüphaneleri Yükleyin:**
    `requirements.txt` dosyası tüm projeler için gerekli bağımlılıkları içerir.
    ```bash
    pip install -r requirements.txt
    ```

4.  **(Chatbot Projesi için) Ollama Kurulumu:**
    `3_chatbot_with_LLM` projesini çalıştırmak için sisteminizde [Ollama](https://ollama.com/)'nın kurulu olması ve `llama3.2:3b` modelinin indirilmiş olması gerekmektedir.
    ```bash
    ollama pull llama3.2:3b
    ```

---

## Proje Detayları ve Dosya Açıklamaları

### 1. Meme Kanseri Tahmini (Klasik Makine Öğrenmesi)

Bu proje, Scikit-learn kullanılarak eğitilmiş bir `RandomForestClassifier` modelinin, hastaya ait özelliklere göre meme kanserinin iyi huylu mu yoksa kötü huylu mu olduğunu tahmin etmesini sağlar.

**Dosya Yapısı ve Görevleri:**
-   `model_train.py`: Modeli `breast_cancer` veri seti ile eğitir ve `breast_cancer_model.pkl` olarak kaydeder.
-   `main.py`: `FastAPI` kullanarak bir web sunucusu oluşturur. Kaydedilen `.pkl` modelini yükler ve gelen veri için tahmin yapan bir `/predict` endpoint'i sunar.
-   `ui.py`: `Streamlit` ile bir web arayüzü oluşturur. Kullanıcının 30 farklı tıbbi özelliği girmesine olanak tanır ve "Tahmin Et" butonuna basıldığında FastAPI'deki `/predict` endpoint'ine istek atarak sonucu gösterir.
-   `client_test.py`: FastAPI sunucusunun doğru çalışıp çalışmadığını test etmek için `/predict` endpoint'ine programatik olarak örnek bir istek gönderir.

**Nasıl Çalıştırılır?**
1.  Ana **`MLOps_Project`** klasöründeyken sanal ortamı aktif edin.
2.  **Backend'i (API Sunucusu) başlatın:**
    ```bash
    uvicorn 1_breast_cancer_project_with_ML.main:app --reload
    ```
3.  Yeni bir terminal açın, sanal ortamı tekrar aktif edin ve **Frontend'i (Arayüz) başlatın:**
    ```bash
    streamlit run 1_breast_cancer_project_with_ML/ui.py
    ```

### 2. El Yazısı Rakam Tanıma (Derin Öğrenme)

Bu proje, MNIST veri seti üzerinde TensorFlow/Keras ile eğitilmiş bir Evrişimli Sinir Ağı (CNN) kullanarak, kullanıcının yüklediği bir resimdeki el yazısı rakamı tanır.

**Dosya Yapısı ve Görevleri:**
-   `model_train.py`: MNIST veri seti ile bir CNN modeli oluşturur, eğitir ve `mnist_model.h5` olarak kaydeder.
-   `main.py`: `FastAPI` sunucusu oluşturur. `.h5` modelini yükler ve 28x28 piksel boyutundaki görüntü verisini işleyerek tahmin yapan bir `/predict` endpoint'i sunar.
-   `ui.py`: `Streamlit` ile bir web arayüzü oluşturur. Kullanıcının bir rakam resmi yüklemesine izin verir. Yüklenen resim, modelin beklediği formata (gri tonlama, 28x28 boyut, normalizasyon) dönüştürülür ve FastAPI'ye gönderilir.
-   `client_test.py`: `images/` klasöründeki bir test görüntüsünü işleyerek API'yi test eder.
-   `images/`: Test için kullanılacak örnek resimleri içerir.

**Nasıl Çalıştırılır?**
1.  Ana **`MLOps_Project`** klasöründeyken sanal ortamı aktif edin.
2.  **Backend'i (API Sunucusu) başlatın:**
    ```bash
    uvicorn 2_handwritten_digits_project_with_DL.main:app --reload
    ```
3.  Yeni bir terminal açın, sanal ortamı tekrar aktif edin ve **Frontend'i (Arayüz) başlatın:**
    ```bash
    streamlit run 2_handwritten_digits_project_with_DL/ui.py
    ```

### 3. LLM Destekli Chatbot

Bu proje, yerel olarak çalışan bir Büyük Dil Modeli (LLaMA 3.2) ve `LangChain` kütüphanesi kullanarak, hafızası olan (konuşma geçmişini hatırlayan) bir chatbot oluşturur.

**Dosya Yapısı ve Görevleri:**
-   `main.py`: `FastAPI` sunucusu oluşturur. `LangChain` ve `Ollama` kullanarak bir sohbet zinciri (`conversational chain`) kurar. Bu zincir, oturum bazlı hafıza yönetimi yapar. Kullanıcıdan gelen mesajları işleyen bir `/chat` endpoint'i sunar.
-   `ui.py`: `Streamlit` ile klasik bir sohbet arayüzü oluşturur. Kullanıcının girdiği her mesajı FastAPI'deki `/chat` endpoint'ine gönderir, gelen cevabı alır ve sohbet geçmişini ekranda gösterir.
-   *(Not: Bu projede `model_train.py` yoktur çünkü önceden eğitilmiş bir LLM kullanılmaktadır. `client_test.py` ise arayüz üzerinden kolayca test edilebildiği için bu projeye dahil edilmemiştir.)*

**Nasıl Çalıştırılır?**
1.  Ana **`MLOps_Project`** klasöründeyken sanal ortamı aktif edin.
2.  **Backend'i (API Sunucusu) başlatın:**
    ```bash
    uvicorn 3_chatbot_with_LLM.main:app --reload
    ```
3.  Yeni bir terminal açın, sanal ortamı tekrar aktif edin ve **Frontend'i (Arayüz) başlatın:**
    ```bash
    streamlit run 3_chatbot_with_LLM/ui.py
    ```


 
## ENG:
# Practical MLOps Project Collection

This repository contains a collection of projects that demonstrate MLOps (Machine Learning Operations) principles in practice across three different domains of artificial intelligence. Each project covers the entire lifecycle, from model training to deployment as an API with FastAPI, creating a user interface with Streamlit, and testing.

## Projects and Technologies Used

This collection consists of projects covering three different branches of AI:

| Project Name | Domain | Core Libraries | Model File |
| :--- | :--- | :--- | :--- |
| **1. Breast Cancer Prediction** | Classic Machine Learning (ML) | `Scikit-learn` | `.pkl` |
| **2. Handwritten Digit Recognition**| Deep Learning (DL) | `TensorFlow/Keras` | `.h5` |
| **3. LLM-Powered Chatbot** | Large Language Models (LLM) | `LangChain`, `Ollama` | (Local Model) |

**Common Technology Stack:**
- **Backend (API Server):** `FastAPI`
- **Frontend (User Interface):** `Streamlit`
- **Model Serialization:** `Joblib`, `H5`

## Setup and Getting Started

Follow these general steps to run the projects. Specific commands for each project are provided under their respective headings.

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/AliDmrcIo/MLOps_Project.git
    cd MLOps_Project
    ```

2.  **Activate the Virtual Environment:**
    The project comes with a `venv` virtual environment that includes the necessary libraries.
    ```bash
    # For Windows
    .\venv\Scripts\activate
    ```

3.  **Install Dependencies:**
    The `requirements.txt` file contains all the necessary dependencies for all projects.
    ```bash
    pip install -r requirements.txt
    ```

4.  **(For the Chatbot Project) Install Ollama:**
    To run the `3_chatbot_with_LLM` project, you need to have [Ollama](https://ollama.com/) installed on your system and the `llama3.2:3b` model downloaded.
    ```bash
    ollama pull llama3.2:3b
    ```

---

## Project Details and File Descriptions

### 1. Breast Cancer Prediction (Classic Machine Learning)

This project uses a `RandomForestClassifier` model trained with Scikit-learn to predict whether a breast cancer tumor is benign or malignant based on patient features.

**File Structure and Roles:**
-   `model_train.py`: Trains the model with the `breast_cancer` dataset and saves it as `breast_cancer_model.pkl`.
-   `main.py`: Creates a web server using `FastAPI`. It loads the saved `.pkl` model and serves a `/predict` endpoint that makes predictions on incoming data.
-   `ui.py`: Creates a web interface with `Streamlit`. It allows the user to input 30 different medical features and, upon clicking the "Predict" button, sends a request to the FastAPI `/predict` endpoint to display the result.
-   `client_test.py`: Sends a programmatic sample request to the `/predict` endpoint to test if the FastAPI server is working correctly.

**How to Run:**
1.  In your terminal, activate the virtual environment while in the main **`MLOps_Project`** directory.
2.  **Start the Backend (API Server):**
    ```bash
    uvicorn 1_breast_cancer_project_with_ML.main:app --reload
    ```
3.  Open a new terminal, activate the virtual environment again, and **start the Frontend (UI):**
    ```bash
    streamlit run 1_breast_cancer_project_with_ML/ui.py
    ```

### 2. Handwritten Digit Recognition (Deep Learning)

This project uses a Convolutional Neural Network (CNN) trained on the MNIST dataset with TensorFlow/Keras to recognize a handwritten digit from an image uploaded by the user.

**File Structure and Roles:**
-   `model_train.py`: Creates, trains, and saves a CNN model with the MNIST dataset as `mnist_model.h5`.
-   `main.py`: Creates a `FastAPI` server. It loads the `.h5` model and serves a `/predict` endpoint that processes 28x28 pixel image data to make a prediction.
-   `ui.py`: Creates a web interface with `Streamlit`. It allows the user to upload a digit image. The uploaded image is preprocessed into the format expected by the model (grayscale, 28x28, normalization) and sent to the FastAPI server.
-   `client_test.py`: Tests the API by processing a test image from the `images/` folder.
-   `images/`: Contains sample images for testing.

**How to Run:**
1.  In your terminal, activate the virtual environment while in the main **`MLOps_Project`** directory.
2.  **Start the Backend (API Server):**
    ```bash
    uvicorn 2_handwritten_digits_project_with_DL.main:app --reload
    ```
3.  Open a new terminal, activate the virtual environment again, and **start the Frontend (UI):**
    ```bash
    streamlit run 2_handwritten_digits_project_with_DL/ui.py
    ```

### 3. LLM-Powered Chatbot

This project creates a chatbot with memory (it remembers conversation history) using a locally running Large Language Model (LLaMA 3.2) and the `LangChain` library.

**File Structure and Roles:**
-   `main.py`: Creates a `FastAPI` server. It sets up a conversational chain using `LangChain` and `Ollama`. This chain manages session-based memory. It serves a `/chat` endpoint that processes incoming user messages.
-   `ui.py`: Creates a classic chat interface with `Streamlit`. It sends each user message to the FastAPI `/chat` endpoint, receives the response, and displays the entire conversation history on the screen.
-   *(Note: This project does not have a `model_train.py` because it uses a pre-trained LLM. A `client_test.py` is also omitted as it can be easily tested via the user interface.)*

**How to Run:**
1.  In your terminal, activate the virtual environment while in the main **`MLOps_Project`** directory.
2.  **Start the Backend (API Server):**
    ```bash
    uvicorn 3_chatbot_with_LLM.main:app --reload
    ```
3.  Open a new terminal, activate the virtual environment again, and **start the Frontend (UI):**
    ```bash
    streamlit run 3_chatbot_with_LLM/ui.py
    ```
