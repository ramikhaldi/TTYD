import os
import glob
import asyncio
import hypercorn.asyncio
import hypercorn.config
import re
import nltk

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from chromadb import Client
from chromadb.utils import embedding_functions
from ollama import chat
from PyPDF2 import PdfReader
import docx
import pandas as pd
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from rank_bm25 import BM25Okapi
from rdflib import Graph
from nltk.tokenize import sent_tokenize

# Configuration
MY_FILES_DIR = "./my_files"
INSTRUCTIONS_FILE = "/app/instructions.txt"
SUPPORTED_FILE_TYPES = ["*.json", "*.pdf", "*.docx", "*.xlsx"]
LOCAL_MODEL_NAME = os.getenv("LOCAL_MODEL_NAME", "all-MiniLM-L6-v2")
MODEL_NAME = os.getenv("MODEL_NAME", "llama3.2:1b")

# Get Ollama's external server details
OLLAMA_SCHEMA = os.getenv("OLLAMA_SCHEMA", "http")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "localhost")
OLLAMA_PORT = os.getenv("OLLAMA_PORT", "11434")
OLLAMA_SERVER = f"{OLLAMA_SCHEMA}://{OLLAMA_HOST}:{OLLAMA_PORT}"

CHROMA_DB_PATH = "./chroma_db"  # Local persistent storage

# Initialize FastAPI
app = FastAPI()

# Request model
class QuestionRequest(BaseModel):
    question: str

# Health check endpoint
@app.get("/health")
async def health_check():
    return JSONResponse(content={"status": "ok"})

# Initialize Chroma Client
def initialize_vector_db():
    return Client()
# Load instructions from `instructions.txt`
def load_instructions():
    try:
        with open(INSTRUCTIONS_FILE, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception as e:
        print(f"⚠️ Warning: Could not load instructions.txt ({e}). Using default instructions.")
        return "You are an AI assistant. Answer questions based on the given content."

INSTRUCTIONS = load_instructions()

# Initialize ChromaDB Client
def initialize_vector_db():
    return Client()

# Chunking function for text splitting
def chunk_text(text, max_chunk_size=512):
    sentences = sent_tokenize(text)
    chunks, current_chunk, current_length = [], [], 0

    for sentence in sentences:
        if current_length + len(sentence) > max_chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk, current_length = [], 0
        current_chunk.append(sentence)
        current_length += len(sentence)

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

# Process different file types
def process_json_file(filepath):
    try:
        with open(filepath, "r", encoding="utf-8") as file:
            return file.read()
    except Exception as e:
        print(f"Error processing JSON file {filepath}: {e}")
        return None

def process_pdf_file(filepath):
    try:
        reader = PdfReader(filepath)
        return "\n".join([page.extract_text() or "" for page in reader.pages])
    except Exception as e:
        print(f"Error processing PDF file {filepath}: {e}")
        return None

def process_docx_file(filepath):
    try:
        doc = docx.Document(filepath)
        return "\n".join([para.text for para in doc.paragraphs])
    except Exception as e:
        print(f"Error processing Word file {filepath}: {e}")
        return None

def process_excel_file(filepath):
    try:
        excel_data = pd.ExcelFile(filepath)
        return "\n".join(excel_data.parse(sheet).to_string(index=False, header=True) for sheet in excel_data.sheet_names)
    except Exception as e:
        print(f"Error processing Excel file {filepath}: {e}")
        return None

# Read and chunk supported files
def read_supported_files(directory):
    documents = []
    for file_type in SUPPORTED_FILE_TYPES:
        for filepath in glob.glob(os.path.join(directory, file_type)):
            content = None
            if filepath.endswith(".json"):
                content = process_json_file(filepath)
            elif filepath.endswith(".pdf"):
                content = process_pdf_file(filepath)
            elif filepath.endswith(".docx"):
                content = process_docx_file(filepath)
            elif filepath.endswith(".xlsx"):
                content = process_excel_file(filepath)

            if content:
                chunks = chunk_text(content)  # ✅ Chunk large documents
                for idx, chunk in enumerate(chunks):
                    documents.append({"content": chunk, "source": f"{os.path.basename(filepath)}_chunk_{idx}"})
    return documents

# Embed and store chunks in ChromaDB
def embed_and_store(documents, client, collection_name="my_files"):
    embeddings = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=LOCAL_MODEL_NAME)
    collection = client.get_or_create_collection(name=collection_name, embedding_function=embeddings)

    for doc in documents:
        collection.add(
            documents=[doc["content"]],
            ids=[doc["source"]],
            metadatas=[{"source": doc["source"]}]
        )
    
    return collection

# Tokenization for BM25
def tokenize(text):
    return text.lower().split()

# Build BM25 Index
def build_bm25_index(documents):
    tokenized_corpus = [tokenize(doc["content"]) for doc in documents]
    return BM25Okapi(tokenized_corpus), documents

# Initialize ChromaDB and document indexing
print("Initializing ChromaDB client...")
client = initialize_vector_db()
print("Reading and chunking supported files...")
documents = read_supported_files(MY_FILES_DIR)
print(f"Embedding {len(documents)} document chunks into ChromaDB...")
collection = embed_and_store(documents, client)
bm25, indexed_docs = build_bm25_index(documents)

print(f"✅ Indexed {collection.count()} document chunks in ChromaDB.")

# Hybrid retrieval: BM25 + Chroma Vector Search
def hybrid_retrieval(question, top_k=10):
    query_embedding = SentenceTransformer(LOCAL_MODEL_NAME).encode(question, convert_to_tensor=True)

    # Vector Search in ChromaDB
    vector_results = collection.query(query_texts=[question], n_results=top_k)
    vector_chunks = vector_results["documents"][0] if vector_results["documents"] else []

    # BM25 Search
    bm25_scores = bm25.get_scores(tokenize(question))
    bm25_top_chunks = [indexed_docs[i]["content"] for i in sorted(range(len(bm25_scores)), key=lambda k: bm25_scores[k], reverse=True)[:top_k]]

    # Merge results
    combined_chunks = list(set(vector_chunks + bm25_top_chunks))

    return "\n".join(combined_chunks) if combined_chunks else "No relevant information found."

# Async generator for Ollama response
async def generate_answer_with_ollama(question):
    context = hybrid_retrieval(question)

    prompt = f"""
    {INSTRUCTIONS}

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    #print(prompt)

    try:
        stream = chat(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
        )

        buffer = ""
        for chunk in stream:
            text = chunk.get("message", {}).get("content", "")

            if text:
                buffer += text  # Add new text to buffer
                words = re.findall(r'\S+\s*', buffer)  # Ensure spaces are preserved

                for word in words[:-1]:  # Send all but the last word (keep buffer smooth)
                    yield word
                    await asyncio.sleep(0.01)  # Adjust speed of response

                buffer = words[-1]  # Keep last word to join with the next chunk

        # Send remaining text in the buffer
        if buffer:
            yield buffer
    except Exception as e:
        yield f"An error occurred while querying Ollama: {e}"

# FastAPI endpoint
@app.post("/ask")
async def ask(request: QuestionRequest):
    if not request.question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    return StreamingResponse(generate_answer_with_ollama(request.question), media_type="text/plain")

# Run server
if __name__ == "__main__":
    config = hypercorn.config.Config()
    config.bind = ["0.0.0.0:5000"]
    asyncio.run(hypercorn.asyncio.serve(app, config))
