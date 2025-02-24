import os
import httpx
import streamlit as st
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.embeddings import HuggingFaceEmbeddings
from index import load_and_process_pdf, create_vector_store

# Load API Key
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"

if not MISTRAL_API_KEY:
    st.error("MISTRAL_API_KEY is missing. Please set it in your environment variables.")
    st.stop()

# Streamlit App Title
st.title("📖 PDF-Based QA System with Mistral AI")

# Upload PDF File
uploaded_file = st.file_uploader("📂 Upload a PDF", type=["pdf"])

if uploaded_file:
    temp_pdf_path = "temp_uploaded_file.pdf"
    
    with open(temp_pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Process PDF
    with st.spinner("📄 Processing PDF..."):
        try:
            chunks = load_and_process_pdf(temp_pdf_path)

            # Use a better embedding model
            embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            vector_store = FAISS.from_documents(chunks, embedding_model)

            st.session_state.vector_store = vector_store  # Store FAISS index in session state
            st.success("✅ PDF processed successfully!")
        except Exception as e:
            st.error(f"❌ Error processing PDF: {e}")
            st.stop()

    # Setup Chat Memory
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Conversational QA Function
    def ask_mistral(question, history, vector_store):
        headers = {"Authorization": f"Bearer {MISTRAL_API_KEY}", "Content-Type": "application/json"}
        
        # Retrieve relevant chunks from FAISS
        retrieved_docs = vector_store.similarity_search(question, k=5)
        retrieved_text = "\n\n".join([doc.page_content for doc in retrieved_docs])

        # Debug: Print retrieved text
        print("🔍 FAISS Retrieved Chunks:")
        print(retrieved_text)

        messages = [
            {"role": "system", "content": "You are an AI assistant that answers questions based on provided text."},
            {"role": "user", "content": f"Context:\n{retrieved_text}\n\nQuestion: {question}"}
        ]

        payload = {"model": "mistral-tiny", "messages": messages}
        
        try:
            response = httpx.post(MISTRAL_API_URL, headers=headers, json=payload, timeout=10)
            response.raise_for_status()
            return response.json().get("choices", [{}])[0].get("message", {}).get("content", "No response from Mistral.")
        except httpx.HTTPStatusError as e:
            return f"❌ API error: {e.response.status_code} - {e.response.text}"
        except Exception as e:
            return f"❌ Error connecting to Mistral API: {e}"

    # User Query Input
    user_query = st.text_input("🔎 Ask a question about the PDF:")

    if user_query and "vector_store" in st.session_state:
        with st.spinner("🤖 Thinking..."):
            answer = ask_mistral(user_query, st.session_state.chat_history, st.session_state.vector_store)

        # Display response
        st.write("**💡 Answer:**", answer)

        # Update chat history
        st.session_state.chat_history.append({"user": user_query, "assistant": answer})

    # Don't delete the file immediately
    # os.remove(temp_pdf_path)
