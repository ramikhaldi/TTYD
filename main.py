import os
import glob
import asyncio
import hypercorn.asyncio
import hypercorn.config
import re

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

# Configuration
MY_FILES_DIR = "./my_files"
INSTRUCTIONS_FILE = "/app/instructions.txt"
SUPPORTED_FILE_TYPES = ["*.json", "*.pdf", "*.docx", "*.xlsx"]
LOCAL_MODEL_NAME = os.getenv("LOCAL_MODEL_NAME", "all-MiniLM-L6-v2")
MODEL_NAME = os.getenv("MODEL_NAME", "llama3.2:1b")
# Get Ollama's external server details from environment variables
OLLAMA_SCHEMA = os.getenv("OLLAMA_SCHEMA", "http")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "localhost")
OLLAMA_PORT = os.getenv("OLLAMA_PORT", "11434")

# Construct the final Ollama server URL
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

# Process JSON files
def process_json_file(filepath):
    try:
        with open(filepath, "r", encoding="utf-8") as file:
            return file.read()
    except UnicodeDecodeError:
        print(f"Warning: Skipping file {filepath} due to encoding issues.")
        return None

# Process PDF files
def process_pdf_file(filepath):
    try:
        reader = PdfReader(filepath)
        text = "".join(page.extract_text() or "" for page in reader.pages)
        return text
    except Exception as e:
        print(f"Error processing PDF file {filepath}: {e}")
        return None

# Process Word files
def process_docx_file(filepath):
    try:
        doc = docx.Document(filepath)
        return "\n".join([para.text for para in doc.paragraphs])
    except Exception as e:
        print(f"Error processing Word file {filepath}: {e}")
        return None

# Process Excel files
def process_excel_file(filepath):
    try:
        excel_data = pd.ExcelFile(filepath)
        return "\n".join(excel_data.parse(sheet).to_string(index=False, header=True) for sheet in excel_data.sheet_names)
    except Exception as e:
        print(f"Error processing Excel file {filepath}: {e}")
        return None

# Read all supported files
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
                documents.append({"content": content, "source": os.path.basename(filepath)})
    return documents

# Embed and store documents in Chroma
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

# Async generator for **continuous word streaming**
async def generate_answer_with_ollama(question, context):
    prompt = f"""
    {INSTRUCTIONS}

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

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


# Build the chatbot retrieval system
def build_retrieval_qa_system(collection):
    def retrieve_and_answer(question):
        results = collection.query(query_texts=[question], n_results=3)
        documents = results["documents"][0] if results["documents"] else []
        context = "\n".join(documents) if documents else "No relevant steps found."
        return context
    return retrieve_and_answer

# Initialize Chroma and load documents
print("Initializing Chroma client...")
client = initialize_vector_db()
print("Reading supported files...")
documents = read_supported_files(MY_FILES_DIR)
print(f"Embedding documents using {LOCAL_MODEL_NAME} and storing in Chroma...")
collection = embed_and_store(documents, client)
print(f"✅ Indexed {collection.count()} documents in ChromaDB.")
print("Building the retrieval-augmented chatbot...")
chatbot = build_retrieval_qa_system(collection)

# FastAPI endpoint for chatbot interaction (POST)
@app.post("/ask")
async def ask(request: QuestionRequest):
    if not request.question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    context = chatbot(request.question)
    return StreamingResponse(generate_answer_with_ollama(request.question, context), media_type="text/plain")

# Run the Hypercorn server
if __name__ == "__main__":
    config = hypercorn.config.Config()
    config.bind = ["0.0.0.0:5000"]  # Listen on port 5000
    config.alpn_protocols = ["h2", "http/1.1"]  # Support both HTTP/2 and HTTP/1.1
    config.accesslog = "-"  # Log access to stdout
    
    print("🚀 Starting server on port 5000 (HTTP/2 enabled)...")
    asyncio.run(hypercorn.asyncio.serve(app, config))
