from fastapi import FastAPI

from pydantic import BaseModel

import numpy as np

from langchain_community.chat_models import ChatOllama
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough

import warnings
warnings.filterwarnings("ignore")

app = FastAPI(title="Chatbot API",
              description="This API provides a simple chatbot interface using the Ollama model.",
              version="1.0.0")

system_prompt = """You are a helpful chef who cooks delicious meals and provides recipes."""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

llm = ChatOllama(model="llama3.2:3b", temperature=0.2)


# conversation chain kısmı:
"""
conversation = ConversationChain(
    llm=llm,
    memory=ConversationBufferMemory(),
    verbose=True
)
yazamadığımızdan(versiyonlardan sebep) bu destansı şeyi yazıyoruz
"""
store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

prompt = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

conversation = RunnableWithMessageHistory(
    prompt | llm,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history"
)

class Message(BaseModel):
    query: str

@app.get("/")
def read_root():
    """Root endpoint."""
    return {"message": "Welcome to the Chatbot API"}

@app.post("/chat")
def chat(message: Message):
    try:
        response = conversation.invoke(
            {"input": message.query},
            config={"configurable": {"session_id": "default_session"}}
        )
        return {
            "response": response.content
        }
    except Exception as e:
        return {"error": str(e)}