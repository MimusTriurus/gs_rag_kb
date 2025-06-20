## Preparing
* install Ollama
* Execute the command ```ollama pull mistral:7b-instruct```
* ```cd models```
* ```git clone https://huggingface.co/BAAI/bge-large-en```
* ```git clone https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2```

* Execute the command inside Poweshell (by administrator) to open  the port ```New-NetFirewallRule -DisplayName "FastAPI 5000" -Direction Inbound -LocalPort 5000 -Protocol TCP -Action Allow```
* (OPTIONAL) Execute the command inside Poweshell (by administrator) to close the port ```Remove-NetFirewallRule -DisplayName "FastAPI 5000"```
* Execute inside venv to start the service: ```uvicorn source.backend.app:app --host 0.0.0.0 --port 5000```

## Preferable models
* mistral-nemo:12b
* mistral:7b-instruct