#!/bin/sh

echo "🔍 Checking and cloning models..."

if [ ! -d "/app/models/bge-large-en" ]; then
  echo "⬇️ Cloning bge-large-en model..."
  git clone https://huggingface.co/BAAI/bge-large-en /app/models/bge-large-en
else
  echo "✅ bge-large-en model already exists."
fi

if [ ! -d "/app/models/ms-marco-MiniLM-L6-v2" ]; then
  echo "⬇️ Cloning ms-marco-MiniLM-L6-v2 model..."
  git clone https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2 /app/models/ms-marco-MiniLM-L6-v2
else
  echo "✅ ms-marco-MiniLM-L6-v2 model already exists."
fi

echo "🚀 Starting application..."
exec uvicorn source.backend.app:app --host 0.0.0.0 --port 5000