FROM python:latest

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data inside the container
RUN python -m nltk.downloader punkt punkt_tab

COPY . .

ENV PYTHONUNBUFFERED=1

# Start the chatbot
CMD ["python", "main.py"]