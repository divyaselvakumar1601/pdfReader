import os
import httpx
import streamlit as st
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory

# Load API Key
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"

if not MISTRAL_API_KEY:
    st.error("MISTRAL_API_KEY is missing. Please set it in your environment variables.")
    st.stop()

# Streamlit App Title
st.title("PDF-Based QA System with Mistral AI")

# Upload PDF File
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file is not None:
    temp_pdf_path = "temp_uploaded_file.pdf"
    with open(temp_pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Load and Process PDF
    from index import load_and_process_pdf, create_vector_store
    chunks = load_and_process_pdf(temp_pdf_path)
    vector_store = create_vector_store(chunks)

    # Setup Memory for Chat History
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # Conversational QA Function
    def ask_mistral(question, history):
        headers = {"Authorization": f"Bearer {MISTRAL_API_KEY}", "Content-Type": "application/json"}
        messages = [{"role": "system", "content": "You are an AI assistant that answers questions based on a PDF."}]

        # Add conversation history
        for msg in history:
            messages.append({"role": "user", "content": msg["user"]})
            messages.append({"role": "assistant", "content": msg["assistant"]})

        messages.append({"role": "user", "content": question})
        
        payload = {"model": "mistral-tiny", "messages": messages}
        response = httpx.post(MISTRAL_API_URL, headers=headers, json=payload)

        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            return "Error: Unable to get response from Mistral API."

    # User Query Input
    user_query = st.text_input("Ask a question about the PDF:")

    if user_query:
        chat_history = memory.load_memory_variables({}).get("chat_history", [])
        answer = ask_mistral(user_query, chat_history)

        st.write("Answer:", answer)
        memory.save_context({"user": user_query}, {"assistant": answer})

    # Clean up temporary file
    os.remove(temp_pdf_path)
