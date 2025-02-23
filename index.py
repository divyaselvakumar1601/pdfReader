import os
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


import os

# Load API Key correctly
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")  # ✅ Correct way to access API key

if not MISTRAL_API_KEY:
    raise ValueError("MISTRAL_API_KEY is missing. Set it in your environment variables.")



# Load and process PDF
def load_and_process_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # Split text into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)

    return chunks

# Use Mistral-compatible Embeddings (Hugging Face)
def create_vector_store(documents):
    embedding_model = HuggingFaceEmbeddings(model_name="thenlper/gte-small")  # Mistral-compatible embeddings
    return FAISS.from_documents(documents, embedding_model)



