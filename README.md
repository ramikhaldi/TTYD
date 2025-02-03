# 🗣️ Talk to Your Data (TTYD)

**AI-Powered Local Chatbot for Secure & Private Data Querying**

## 🚀 Motivation

**Talk to Your Data (TTYD)** is an **AI-powered chatbot** that allows you to interact with your own documents locally, without relying on external APIs. Unlike cloud-based AI models that expose your sensitive data to third parties, TTYD processes **everything on your machine**, ensuring **maximum privacy and security**.

### 🔒 Why Local AI?

- ✅ **Domain-Specific Accuracy** – Fine-tuned for your own knowledge base with minimal risk of hallucination.

- ✅ **Full Data Privacy** – Your data **never leaves** your machine.

- ✅ **Customizable AI** – Tune AI parameters to fit your needs.

- ✅ **Works Offline** – No internet dependency.

TTYD is ideal for **researchers, businesses, and privacy-conscious users** who want **secure, local AI-driven document querying**. 🧠💡

---

## 🏗️ How It Works

TTYD combines **retrieval-augmented generation (RAG)** with hybrid search techniques:

- 1️⃣ **Document Ingestion** – It processes PDFs, Word, Excel, and JSON files, chunking them into smaller pieces.
- 2️⃣ **Hybrid Search Engine** – Uses **BM25** (statistical search) and **Weaviate vector search** for accurate information retrieval.
- 3️⃣ **AI Answering** – The retrieved context is passed to a **hosted Ollama model** (e.g., Llama 3) for response generation.

---

## 🛠️ Prerequisites

### 🧠 Ensure Ollama is Installed and Running

Before using TTYD, ensure **[Ollama](https://ollama.com)** is installed and running with your preferred model. Ollama can run **locally, on-premise, or a remote host** of your choice.

### 🐳 Install Docker

#### 🖥️ **For Windows Users**

- Install **[Docker Desktop](https://www.docker.com/products/docker-desktop/)**
- Ensure WSL 2 backend is enabled (recommended for performance)

#### 🐧 **For Linux Users**

- Install **Docker Engine** ([Guide](https://docs.docker.com/engine/install/))
- Install **Docker Compose** ([Guide](https://docs.docker.com/compose/install/))

📌 **Ensure Docker is running before proceeding with setup by checking:**
```sh
docker ps
```

---

## 🛠️ Installation & Setup

### 1️⃣ **Clone the Repository**

```sh
git clone https://github.com/ramikhaldi/TTYD
cd ttyd
```

### 2️⃣ **Run with Docker**

```sh
docker compose up --build
```

### 3️⃣ **Access TTYD**

Once running, access the chatbot via:

```sh
http://localhost:5000
```

To send a test query:

```sh
curl -X POST "http://localhost:5000/ask" -H "Content-Type: application/json" -d '{"question": "Summarize my files."}'
```

---

## ⚙️ Configurable Parameters

TTYD allows you to **fine-tune** its behavior via **environment variables** in `docker-compose.yml`.

| Parameter            | Default Value      | Description                                                             |
| -------------------- | ------------------ | ----------------------------------------------------------------------- |
| `OLLAMA_SCHEMA`      | `http`             | Communication protocol for Ollama                                       |
| `OLLAMA_HOST`        | `localhost`        | Host where Ollama is running (local, or on-premise)                     |
| `OLLAMA_PORT`        | `11434`            | Port for Ollama API                                                     |
| `OLLAMA_TEMPERATURE` | `0.5`              | Adjusts response creativity (0 = deterministic, 1+ = diverse)           |
| `WEAVIATE_HOST`      | `weaviate`         | Hostname for Weaviate (vector database)                                 |
| `WEAVIATE_PORT`      | `8080`             | Port for Weaviate                                                       |
| `WEAVIATE_ALPHA`     | `0.5`              | Hybrid search weight (0 = BM25 only, 1 = vector search only)            |
| `MODEL_NAME`         | `llama3.2:3b`      | Local AI model used by Ollama                                           |
| `LOCAL_MODEL_NAME`   | `all-MiniLM-L6-v2` | Sentence Transformer model for vector search                            |

🔹 Adjust these in `docker-compose.yml`

🔹 The `instructions.txt` file can be adapted to fit your specific needs.

---

## 📝 File Support

TTYD processes the following document types:

- 📄 **PDF** (`.pdf`)
- 📝 **Word Docs** (`.docx`)
- 📊 **Excel Sheets** (`.xlsx`)
- 📜 **JSON Files** (`.json`)

To use your own files, **place them in** `my_files/` **before starting TTYD.**

---

## 🔬 Advanced Features

- **Multi-Document Querying** – Ask questions across multiple files.
- **Weaviate Hybrid Search** – Combines **semantic search (AI-based)** and **keyword search (BM25)**.
- **Ollama AI Customization** – Easily swap models (Llama, DeepSeek, Mistral, Gemma, etc.).
- **Live Streaming Responses** – Faster interactions via **FastAPI Streaming API**.
- **Domain-Specific Accuracy** – Tailor AI to your knowledge base with **minimal hallucination risk**.

---

## 🛠️ Development & Contribution

TTYD is **open-source**, and contributions are welcome! 🎉

### 🔨 **Local Development**

1. Fork & clone the repo.
2. Modify/extend/Improve.
3. Run, Test, and benchmark.
4. Submit a pull request. 🚀

---

## 📜 License

TTYD is licensed under the **MIT License** – free to use and modify! 🛡️

---

## 💡 Future Roadmap

✅ Support more document types

✅ GUI for non-technical users

✅ Maintain & Control History

✅ Integration with additional AI model frameworks beyond Ollama

✅ Any Suggestions?

**Enjoy private AI-powered document chat! 🏆**

