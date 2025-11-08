## TR:
# Proje 3: LLM Destekli Chatbot (MLOps ile LLM Uygulaması)

Bu proje, MLOps yaşam döngüsünün modern Büyük Dil Modelleri (LLM) alanına nasıl uyarlandığını göstermektedir. Bu uygulamada, `model_train.py` gibi geleneksel bir eğitim adımı yoktur; bunun yerine, yerel olarak çalışan önceden eğitilmiş bir LLaMA modeli (`Ollama` aracılığıyla) kullanılır. `LangChain` kütüphanesi ile hafıza yönetimi sağlanır, `FastAPI` ile bir sohbet API'si oluşturulur ve `Streamlit` ile kullanıcı arayüzü sunulur.

---

## Projenin Amacı

Kullanıcıyla yaptığı sohbetleri hatırlayabilen (hafızalı), yerel bir LLM tarafından desteklenen etkileşimli bir chatbot oluşturmaktır.

## 🛠️ Kullanılan Teknolojiler

- **Model ve Orkestrasyon:** `LangChain`, `Ollama` (LLaMA 3.2)
- **Backend (API Sunucusu):** `FastAPI`
- **Frontend (Kullanıcı Arayüzü):** `Streamlit`
- **Ana Konsept:** Hafızalı Sohbet Zinciri (Conversational Chain with Memory)

## 📂 Dosya Yapısı ve Görevleri

-   **`main.py`**: `FastAPI` sunucusu oluşturur. `LangChain` ve `Ollama` kullanarak, oturum bazlı hafıza yönetimi yapan bir sohbet zinciri kurar. Kullanıcıdan gelen mesajları işleyen bir `/chat` endpoint'i sunar.
-   **`ui.py`**: `Streamlit` ile klasik bir sohbet arayüzü oluşturur. Kullanıcının girdiği her mesajı FastAPI'deki `/chat` endpoint'ine gönderir, gelen cevabı alır ve sohbet geçmişini ekranda gösterir.
-   *(Not: Bu projede `model_train.py` yoktur çünkü önceden eğitilmiş bir LLM kullanılmaktadır. `client_test.py` ise arayüz üzerinden kolayca test edilebildiği için bu projeye dahil edilmemiştir.)*

## Nasıl Çalıştırılır?

Bu projeyi çalıştırmak için iki ayrı terminal penceresine ihtiyacınız olacaktır: biri backend (FastAPI) için, diğeri frontend (Streamlit) için.

**Ön Koşullar:**
- Ana `MLOps_Project` klasöründe kurulum adımlarını tamamladığınızdan, sanal ortamın (`venv`) aktif olduğundan ve **Ollama'nın çalıştığından** emin olun.

**1. Adım: Backend Sunucusunu Başlatma**
   - Yeni bir terminal açın.
   - Ana `MLOps_Project` dizinine gidin ve sanal ortamı aktif edin:
     ```bash
     cd path/to/MLOps_Project
     .\venv\Scripts\activate
     ```
   - FastAPI sunucusunu çalıştırın:
     ```bash
     uvicorn 3_chatbot_with_LLM.main:app --reload
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
     streamlit run 3_chatbot_with_LLM/ui.py
     ```
   - Tarayıcınızda otomatik olarak yeni bir sekme açılacak ve chatbot ile sohbet etmeye başlayabileceksiniz.

---
---

## ENG:
# Project 3: LLM-Powered Chatbot (MLOps with LLMs)

This project demonstrates how the MLOps lifecycle is adapted for the modern Large Language Models (LLM) domain. In this application, there is no traditional training step like `model_train.py`; instead, it utilizes a pre-trained LLaMA model running locally (via `Ollama`). Memory management is handled by `LangChain`, a chat API is created with `FastAPI`, and the user interface is served with `Streamlit`.

---

## Project Goal

To create an interactive chatbot, powered by a local LLM, that can remember the context of the conversation (i.e., has memory).

## 🛠️ Technologies Used

- **Model and Orchestration:** `LangChain`, `Ollama` (LLaMA 3.2)
- **Backend (API Server):** `FastAPI`
- **Frontend (User Interface):** `Streamlit`
- **Core Concept:** Conversational Chain with Memory

## 📂 File Structure and Roles

-   **`main.py`**: Creates a `FastAPI` server. It sets up a conversational chain using `LangChain` and `Ollama` that manages session-based memory. It serves a `/chat` endpoint that processes incoming user messages.
-   **`ui.py`**: Creates a classic chat interface with `Streamlit`. It sends each user message to the FastAPI `/chat` endpoint, receives the response, and displays the entire conversation history on the screen.
-   *(Note: This project does not have a `model_train.py` because it uses a pre-trained LLM. A `client_test.py` is also omitted as it can be easily tested via the user interface.)*

## How to Run

You will need two separate terminal windows to run this project: one for the backend (FastAPI) and one for the frontend (Streamlit).

**Prerequisites:**
- Ensure you have completed the setup steps in the main `MLOps_Project` directory, that the virtual environment (`venv`) is activated, and that **Ollama is running**.

**Step 1: Start the Backend Server**
   - Open a new terminal.
   - Navigate to the main `MLOps_Project` directory and activate the virtual environment:
     ```bash
     cd path/to/MLOps_Project
     .\venv\Scripts\activate
     ```
   - Run the FastAPI server:
     ```bash
     uvicorn 3_chatbot_with_LLM.main:app --reload
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
     streamlit run 3_chatbot_with_LLM/ui.py
     ```
   - A new tab will automatically open in your browser, and you can start chatting with the bot.
