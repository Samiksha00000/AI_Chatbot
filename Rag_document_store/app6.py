import streamlit as st
from  pypdf import PdReader
from sentence_transformer import sentenceTransformer
import chromadb
import ollama
st.set_page_config(page_title="MINI RAG",
                   (import) st: Module("streamlit"))
st.title("Mini RAG: Document Store + Retrieval")
st.caption("PDF -> Chunks -> Embeddings -> ChromaDB -> Retrieval -> Ollama")

@st.cache_resource
def load_embedding_model():
    return sentenceTransformer("all-MiniLM-L6-v2")

model = load_embedding_model()

#create persistent local Chromadb
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="documents")

st.sidebar.header("Settings")
ollama_model = st.sidebar.text_input("Ollama model","llama3.2")
chunk_size = st.sidebar.slider("Chunk size", 200, 1500, 500, 100)
top_k = st.sidebar.slider("Chunks to retrieve", 1, 5, 3)
st.header("Build Document Store")
uploaded_file = st.file_uploader("Upload a text-based PDF", type=["pdf"])
if uploaded_file and st.button(" Process & store PDF"):
    reader = PdfReader(uploaded_file)
    text = ""
    for page_number, page in enumerate(reader.pages,start=1):
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    if not text.strip():
        st.error("No readable text was found. Try a text-based PDF")
        st.stop()

    chunks = []
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i + chunk_size].strip()
        if chunk:
            chunks.append(chunk)   

    with st.spinner("Generating embeddings...."):
        embeddings = model.encode(chunks)    ####
        ####


    st.succ    








