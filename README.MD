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

## yaml frontmatter generation prompt:
```
Проанализируй текст и сформируй краткое описание и набор тэгов для наиболее эффективного поиска в RAG системе (yaml frantmatter).  
Все должно быть на английском 
Первая строка текста это поле title, вторая - url, третья - author
Шаблон:
---
title: <text>
url: https://<address>
author: <author>
summarized_text: <text>
tags:
- <tag#1>
- <tag#2>
---

Текст для анализа:
<текст>
```
