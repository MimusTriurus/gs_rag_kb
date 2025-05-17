# app.py
import os
import asyncio
from pathlib import Path
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from sentence_transformers import SentenceTransformer, CrossEncoder
from llama_cpp import Llama
from document_utils import parse_documents, build_or_load_index_for_file, select_best_files, retrieve_and_rerank
from interaction import refine_user_prompt, answer_question
from settings import (
    DOCUMENTS_PATH,
    CACHE_DIR,
    EMBED_MODEL_NAME,
    CROSS_ENCODER_NAME,
    GGUF_MODEL_PATH,
    need_2_refine_query
)

import functools
import concurrent.futures

executor = concurrent.futures.ThreadPoolExecutor()


async def run_in_thread(func, *args, **kwargs):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(executor, functools.partial(func, *args, **kwargs))


# Initialize FastAPI
app = FastAPI(title="Async RAG API", version="1.0", description="RAG search with local LLM")

# Enable CORS if needed
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Models
class QueryInput(BaseModel):
    query: str


class ResponseOutput(BaseModel):
    answer: str
    url: str
    author: str


class Feedback(BaseModel):
    query: str
    liked: bool


feedback_store = []

# Load models and indices at startup
os.makedirs(CACHE_DIR, exist_ok=True)
embed_model = SentenceTransformer(EMBED_MODEL_NAME)
cross_encoder = CrossEncoder(CROSS_ENCODER_NAME)
llm = Llama(
    model_path=GGUF_MODEL_PATH,
    embedding=False,
    n_ctx=2048,
    n_threads=4,
    verbose=False,
    n_gpu_layers=256,
)
file_indices, file_titles, file_paths, file_meta = parse_documents(DOCUMENTS_PATH, embed_model)


@app.post("/rag/search", response_model=ResponseOutput)
async def rag_search(input_data: QueryInput):
    user_query = input_data.query
    query = await run_in_thread(refine_user_prompt, llm, user_query) if need_2_refine_query else user_query
    selected = await run_in_thread(select_best_files, query, file_titles, file_paths, embed_model)
    results = []
    for fname in selected:
        index, chunks, sources = file_indices[fname]
        res = await run_in_thread(retrieve_and_rerank, embed_model, cross_encoder, index, chunks, sources, query)
        results.extend(res)
    context_parts = []
    url = ''
    author = ''
    for text, src in results:
        url, author = file_meta.get(src, ('', ''))
        header = f"Source: {src} (URL: {url}, Author: {author})"
        context_parts.append(f"{header}\n{text}")
    context = '\n---\n'.join(context_parts)
    answer = await run_in_thread(answer_question, llm, context, query)
    return ResponseOutput(answer=answer, url=url, author=author)


@app.post("/feedback")
async def submit_feedback(feedback: Feedback):
    feedback_store.append(feedback.dict())
    return {"status": "received"}


@app.get("/chat")
async def chat_ui():
    content = ''
    with open('frontend/index.html', 'r', encoding='utf-8') as f:
        content = f.read()
    return HTMLResponse(content)


@app.get("/openapi.json")
def custom_openapi():
    return get_openapi(
        title=app.title,
        version=app.version,
        routes=app.routes,
        description=app.description
    )


if __name__ == '__main__':
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=5000, reload=False)
