import os
import glob
import asyncio
import hypercorn.asyncio
import hypercorn.config
import re
import nltk
import weaviate

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ollama import chat
from pypdf import PdfReader  # ✅ Use pypdf instead of PyPDF2
import docx
import pandas as pd
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from rank_bm25 import BM25Okapi
from rdflib import Graph
from nltk.tokenize import sent_tokenize
from weaviate.connect import ConnectionParams
from weaviate.classes.config import Configure, Property
from weaviate.exceptions import WeaviateBaseError  # ✅ Correct Exception Handling
from sentence_transformers import SentenceTransformer  # ✅ Needed for encoding queries
from rank_bm25 import BM25Okapi  # ✅ Needed for BM25 scoring
from dotenv import load_dotenv
import tiktoken
import aiohttp
import asyncio
import json
import re
from pptx import Presentation  # Import pptx library

# Configuration
load_dotenv()
MY_FILES_DIR = "./my_files"
INSTRUCTIONS_FILE = "/app/instructions.txt"
SUPPORTED_FILE_TYPES = ["*.json", "*.pdf", "*.docx", "*.xlsx", "*.pptx", "*.ppt"]
LOCAL_MODEL_NAME = os.getenv("LOCAL_MODEL_NAME")
MODEL_NAME = os.getenv("MODEL_NAME")

# Get Ollama's external server details
TTYD_API_PORT = os.getenv("TTYD_API_PORT")
OLLAMA_TEMPERATURE = os.getenv("OLLAMA_TEMPERATURE")
OLLAMA_SERVER = f"http://ollama:11434"
OLLAMA_TEMPERATURE = os.getenv("OLLAMA_TEMPERATURE")
TTYD_UI_PORT = os.getenv("TTYD_UI_PORT")

# Weaviate Configuration
WEAVIATE_HOST = "weaviate"
WEAVIATE_PORT = 8080
WEAVIATE_ALPHA = float(os.getenv("WEAVIATE_ALPHA"))  # ✅ Configurable hybrid search parameter

# Initialize FastAPI
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"], 
)

# ✅ Define `ensure_collection_exists` **before** it's used
def ensure_collection_exists(client, collection_name="Document"):
    try:
        return client.collections.get(collection_name)
    except WeaviateBaseError:  # ✅ Catch exception when collection does not exist
        return client.collections.create(
            name=collection_name,
            configure_vectorizer=True,  # ✅ Required for text2vec-transformers
            vectorizer_config=Configure.Vectorizer.text2vec_transformers(
                model=LOCAL_MODEL_NAME,  # ✅ Make sure it matches your SentenceTransformer model
                pooling_strategy="cls",
                vectorize_property_name=True  # ✅ Enable automatic vectorization of `content`
            ),
            properties=[
                Property(name="content", data_type="text"),
                Property(name="source", data_type="text"),
            ]
        )


# ✅ Set Weaviate connection details (local-only, no external API calls)
client = weaviate.WeaviateClient(
    connection_params=ConnectionParams.from_params(
        http_host=WEAVIATE_HOST,
        http_port=int(WEAVIATE_PORT),
        http_secure=False,
        grpc_host=WEAVIATE_HOST,
        grpc_port=50051,
        grpc_secure=False,
    ),
    skip_init_checks=False
)

client.connect()

# ✅ Load instructions
def load_instructions():
    try:
        with open(INSTRUCTIONS_FILE, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception as e:
        print(f"⚠️ Warning: Could not load instructions.txt ({e}). Using default instructions.")
        return "You are an AI assistant. Answer questions based on the given content."

INSTRUCTIONS = load_instructions()

# ✅ Chunking function
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

# ✅ Read supported files
def read_supported_files(directory):
    documents = []
    for file_type in SUPPORTED_FILE_TYPES:
        for filepath in glob.glob(os.path.join(directory, file_type)):
            # Skip temporary files that start with '~$'
            if os.path.basename(filepath).startswith("~$"):
                print(f"skip temporary file: {filepath}")
                continue

            content = None  # initialize content to None

            if filepath.endswith(".json"):
                with open(filepath, "r", encoding="utf-8", errors="replace") as file:
                    content = file.read()
            elif filepath.endswith(".pdf"):
                content = "\n".join([page.extract_text() or "" for page in PdfReader(filepath).pages])
            elif filepath.endswith(".docx"):
                content = "\n".join([para.text for para in docx.Document(filepath).paragraphs])
            elif filepath.endswith(".xlsx"):
                excel_data = pd.ExcelFile(filepath)
                content = "\n".join(excel_data.parse(sheet).to_string(index=False, header=True) for sheet in excel_data.sheet_names)
            elif filepath.endswith(".pptx") or filepath.endswith(".ppt"):
                # Handle PPTX and PPT files
                presentation = Presentation(filepath)
                
                # Iterate over slides
                for slide_idx, slide in enumerate(presentation.slides):
                    content = ""
                    
                    # Iterate over shapes to collect text from all text-containing shapes
                    for shape in slide.shapes:
                        if hasattr(shape, "text"):  # Check if the shape contains text
                            # Clean up extra new lines and spaces
                            cleaned_text = shape.text.strip().replace('\r', '')  # Remove carriage returns
                            cleaned_text = re.sub(r'(\n\s*){2,}', '\n', cleaned_text)  # Replace multiple new lines with a single new line
                            content += cleaned_text + " "  # Add space between texts from shapes
                    
                    # Instead of chunking the text, store the whole slide text as one chunk
                    documents.append({"content": content.strip(), "source": f"{os.path.basename(filepath)}_slide_{slide_idx + 1}_chunk_0"})
                    print("------------")
                    print(content)
                    print("------------------")
                # Clear content to avoid re-processing the last slide's text
                content = None
            else:
                content = None

            if content:
                if filepath.endswith(".json"):
                    # For JSON files, use the full content without chunking.
                    documents.append({"content": content, "source": f"{os.path.basename(filepath)}_chunk_0"})
                else:
                    # For other file types, chunk the content.
                    for idx, chunk in enumerate(chunk_text(content)):
                        # print("-----------")
                        # print(chunk)
                        # print("-----------")
                        documents.append({"content": chunk, "source": f"{os.path.basename(filepath)}_chunk_{idx}"})
    return documents

# ✅ Embed and store in Weaviate
def embed_and_store(documents, client, collection_name="Document"):
    embedding_model = SentenceTransformer(LOCAL_MODEL_NAME)
    collection = ensure_collection_exists(client, collection_name)
    for doc in documents:
        embedding = embedding_model.encode(doc["content"]).tolist()
        collection.data.insert(
            properties={
                "content": doc["content"],
                "source": doc["source"],
                "vector": embedding
            }
        )

# ✅ BM25 Indexing
def tokenize(text):
    return text.lower().split()

def build_bm25_index(documents):
    return BM25Okapi([tokenize(doc["content"]) for doc in documents]), documents



# Initialize Weaviate and document indexing
print("Initializing Weaviate client...")
documents = read_supported_files(MY_FILES_DIR)
print(f"Embedding {len(documents)} document chunks into Weaviate...")
embed_and_store(documents, client)
bm25, indexed_docs = build_bm25_index(documents)  # ✅ Ensure these variables are set


# ✅ Hybrid retrieval
def hybrid_retrieval(question, top_k=10):
    embedding_model = SentenceTransformer(LOCAL_MODEL_NAME)
    question_embedding = embedding_model.encode(question).tolist()

    collection = client.collections.get("Document")
    response = collection.query.hybrid(
        query=question,
        vector=question_embedding,
        alpha=WEAVIATE_ALPHA,
        limit=top_k,
        return_properties=["content", "source"]
    )

    vector_chunks = response.objects  # Correct response parsing
    bm25_scores = bm25.get_scores(tokenize(question))
    bm25_top_chunks = [
        {"content": indexed_docs[i]["content"], "source": indexed_docs[i]["source"]}
        for i in sorted(range(len(bm25_scores)), key=lambda k: bm25_scores[k], reverse=True)[:top_k]
    ]

    if not vector_chunks and not bm25_top_chunks:
        return [], "No relevant information found."

    retrieved_docs = [
        {"content": doc.properties["content"], "source": doc.properties["source"]}
        for doc in vector_chunks
    ] + bm25_top_chunks

    return retrieved_docs




# ✅ Ensure this class is defined at the beginning
class QuestionRequest(BaseModel):
    question: str

# ✅ Keep generate_answer_with_ollama before calling it
# ✅ Improved citation system and removed chunk details
async def generate_answer_with_ollama(question):
    retrieved_docs = hybrid_retrieval(question)

    if not retrieved_docs:
        yield "No relevant information found."
        return

    # ✅ Deduplicate retrieved content to avoid repeated paragraphs
    unique_content = {}
    for doc in retrieved_docs:
        file_name = doc['source'].split("_chunk_")[0]  # Extract actual file name without chunk details
        if file_name not in unique_content:
            unique_content[file_name] = doc['content']

    # ✅ Format retrieved documents with citations
    formatted_context = "\n\n".join(
        [f"[{i+1}] {text} (Source: {source})" for i, (source, text) in enumerate(unique_content.items())]
    )

    # ✅ Construct the AI prompt with explicit citation instructions
    prompt = f"""
    {INSTRUCTIONS}

    Context:
    {formatted_context}

    Question:
    {question}

    Answer:
    Please generate an answer **based on the provided context** and cite sources using [number] format where applicable.
    """
    print(prompt)  # Debug: Print prompt for verification

    try:
        # Use tiktoken to count tokens
        encoding = tiktoken.get_encoding("cl100k_base")
        prompt_tokens = len(encoding.encode(prompt))
        max_context_tokens = 8000
        print(f"Prompt tokens: {prompt_tokens}")
        print(f"Max context tokens: {max_context_tokens}")
        if prompt_tokens >= max_context_tokens:
            #TODO: More work to do here!
            print(f"⚠️ Warning: prompt_tokens exceeded max_context_tokens!")

        # Build the payload for the API call
        payload = {
            "model": MODEL_NAME,
            "prompt": prompt,
            "options": {
                "temperature": float(OLLAMA_TEMPERATURE),
                "num_ctx": max_context_tokens  # Provide the full context window; the server reserves prompt tokens internally.
            },
            "stream": True
        }

        url = f"{OLLAMA_SERVER}/api/generate"

        # Make an asynchronous POST request using aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as resp:
                buffer = ""
                # Process each line (each JSON object) from the streaming response
                async for line in resp.content:
                    line = line.decode('utf-8').strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        # Extract the response text from the JSON object
                        chunk_text = data.get("response", "")
                        if chunk_text:
                            buffer += chunk_text
                            # Optionally, yield word by word for smooth streaming
                            words = re.findall(r'\S+\s*', buffer)
                            for word in words[:-1]:
                                yield word
                                await asyncio.sleep(0.01)  # Adjust the pace as needed
                            buffer = words[-1] if words else ""
                    except json.JSONDecodeError:
                        print("Error decoding JSON:", line)
                # Yield any remaining text in the buffer
                if buffer:
                    yield buffer

        # ✅ Append the source list at the end (without chunk numbers)
        yield "\n\n**Sources:**\n" + "\n".join(
            [f"- **[{i+1}]** {source}" for i, source in enumerate(unique_content.keys())]
        )

    except Exception as e:
        yield f"An error occurred: {e}"


# Health check endpoint
@app.get("/health")
async def health_check():
    return JSONResponse(content={"status": "ok"})

# ✅ Define FastAPI endpoint AFTER defining QuestionRequest and generate_answer_with_ollama
@app.post("/ask")
async def ask(request: QuestionRequest):
    if not request.question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    return StreamingResponse(generate_answer_with_ollama(request.question), media_type="text/plain")


# Determine the number of CPU cores and set the number of workers accordingly
cpu_cores = os.cpu_count()  # Get the number of CPU cores
workers = cpu_cores if cpu_cores is not None else 1  # Use number of cores, default to 1 if unavailable

# Set Hypercorn configuration with workers based on the number of CPU cores
config = hypercorn.config.Config()
config.bind = [f"0.0.0.0:{TTYD_API_PORT}"]  # Ensure binding is correct
config.alpn_protocols = ["h2", "http/1.1"]  # Explicitly allow HTTP/2 first
config.workers = workers  # Set the number of workers based on the number of CPU cores
config.accesslog = "-"

# Run the Hypercorn server with multiprocessing support (using asyncio)
if __name__ == "__main__":
    print(f"🚀 Starting server with {workers} workers on port {TTYD_API_PORT} ...")
    # Use the correct asyncio method to start the server
    asyncio.run(hypercorn.asyncio.serve(app, config))