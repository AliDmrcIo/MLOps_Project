import streamlit as st
import requests

st.set_page_config(page_title="Chatbot", page_icon=":robot_face:")
st.title("Chatbot Interface")
st.markdown("This is a simple chatbot interface using the Ollama model.")

API_URL = "http://127.0.0.1:8000/chat"

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.text_input("You:", key="user_input", placeholder="Type your message here...")

if st.button("Send"):
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        response = requests.post(API_URL, json={"query": user_input})

        if response.status_code == 200:
            bot_response = response.json().get("response", "No response from bot.")
            st.session_state.chat_history.append({"role": "bot", "content": bot_response})
        else:
            st.error(f"Error: {response.status_code} - {response.text}")
        
        st.rerun()  # Input'u temizlemek için sayfayı yenile

st.markdown("### Chat History")

for message in st.session_state.chat_history:
    if message["role"] == "user":
        st.markdown(f"**You:** {message['content']}")
    else:
        st.markdown(f"**Bot:** {message['content']}")
        