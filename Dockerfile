FROM python:3.10

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data inside the container
RUN python -m nltk.downloader punkt punkt_tab

#TODO: move it into requirements.txt 
RUN pip install weaviate-client
RUN pip install pypdf
RUN pip install sentence-transformers

COPY . .

ENV PYTHONUNBUFFERED=1

# Start the chatbot
CMD ["python", "main.py"]