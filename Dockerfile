FROM python:3.10

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Set default environment variables for Ollama
ENV OLLAMA_SCHEMA="http"
ENV OLLAMA_HOST="host.docker.internal"
ENV OLLAMA_PORT="11434"

ENV PYTHONUNBUFFERED=1

# Start the chatbot
CMD ["python", "main.py"]