## Preparing
* install Ollama
* Execute the command ```ollama pull mistral:7b-instruct```
* ```cd models```
* ```git clone https://huggingface.co/BAAI/bge-large-en```
* ```git clone https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2```

* ```uvicorn source.backend.app:app --host 0.0.0.0 --port 5000```