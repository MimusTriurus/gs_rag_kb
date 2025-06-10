import os
import asyncio
from pathlib import Path
from typing import List, Tuple, Dict, Any

from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from sentence_transformers import SentenceTransformer, CrossEncoder
from starlette.responses import JSONResponse

from source.backend.document_utils import load_index_data, retrieve_and_rerank, select_best_files
from source.backend.interaction import refine_user_prompt, answer_question
from source.backend.settings import (
    DOCUMENTS_PATH,
    CACHE_DIR,
    EMBED_MODEL_NAME,
    CROSS_ENCODER_NAME,
    LLM_MODEL,
    need_2_refine_query,
    missing_info_text,
    no_info_in_knowledge_base_message
)
from source.backend.db_utils import (
    init_db,
    insert_feedback,
    insert_not_found_query,
    get_all_feedback,
    get_all_not_found_queries
)

import functools
import concurrent.futures

executor = concurrent.futures.ThreadPoolExecutor()


async def run_in_thread(func, *args, **kwargs):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(executor, functools.partial(func, *args, **kwargs))


app = FastAPI(title="Async RAG API", version="1.0", description="RAG search with local LLM")

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
    answer: str


os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs("db", exist_ok=True)
db_path = os.path.join("db", "data.db")
init_db(db_path)

os.makedirs(CACHE_DIR, exist_ok=True)
embed_model = SentenceTransformer(EMBED_MODEL_NAME)
cross_encoder = CrossEncoder(CROSS_ENCODER_NAME)
file_indices, file_titles, file_paths, file_meta = load_index_data(Path(DOCUMENTS_PATH))


@app.post("/rag/search", response_model=ResponseOutput, include_in_schema=False)
async def rag_search(input_data: QueryInput):
    user_query = input_data.query

    query = await run_in_thread(refine_user_prompt, user_query, LLM_MODEL) \
        if need_2_refine_query else user_query
    # 1. select best candidates for the data extraction
    selected_files = await run_in_thread(
        select_best_files, query, file_paths, file_meta, embed_model
    )

    all_retrieved_results: List[Tuple[str, Dict[str, Any], float]] = []

    # 2. extract and rerank chunks for the each selected file
    for fname in selected_files:
        faiss_index, chunks_content_list, chunks_metadata_list = file_indices[fname]
        retrieved_and_ranked_for_file = await run_in_thread(
            retrieve_and_rerank, embed_model, cross_encoder,
            faiss_index, chunks_content_list, chunks_metadata_list, query
        )
        all_retrieved_results.extend(retrieved_and_ranked_for_file)

    # 3. sort results by score
    all_retrieved_results.sort(key=lambda x: x[2], reverse=True)  # x[2] is a score (float)

    context_parts: List[str] = []
    best_url: str = ''
    best_author: str = ''

    # 4. Generate context for the LLM and extract metadata for the response
    # Go through the sorted results and gather context.
    # Take the URL and author from the most relevant chunk (first in the sorted list),
    # If they are not already set.
    for text_content, metadata_dict, score in all_retrieved_results:
        context_parts.append(text_content)

        if not best_url and metadata_dict.get('source_url'):
            best_url = metadata_dict['source_url']
        if not best_author and metadata_dict.get('author'):
            best_author = metadata_dict['author']

        # optional. limit the content size
        # if get_token_length('\n---\n'.join(context_parts)) > SOME_LLM_MAX_CONTEXT_TOKENS:
        #     break

    context = '\n---\n'.join(context_parts) if context_parts else ""

    # 5. get an answer from LLM
    answer = await run_in_thread(answer_question, context, query, LLM_MODEL)

    # 6. handle "no info" event
    if missing_info_text in answer:
        await run_in_thread(insert_not_found_query, db_path, user_query)
        answer = no_info_in_knowledge_base_message
        best_url = ''
        best_author = ''

    return ResponseOutput(answer=answer, url=best_url, author=best_author)


@app.post("/feedback", include_in_schema=False)
async def submit_feedback(feedback: Feedback):
    insert_feedback(db_path, feedback.query, feedback.liked)
    return {"status": "received"}


@app.get("/get_feedback", response_class=JSONResponse)
async def get_feedback():
    data = get_all_feedback(db_path)
    return JSONResponse([{"query": q, "liked": bool(l)} for q, l in data])

@app.get("/get_not_found_data", response_class=JSONResponse)
async def get_not_found_data():
    data = get_all_not_found_queries(db_path)
    return JSONResponse([q for (q,) in data])


@app.get("/chat", include_in_schema=False)
async def chat_ui():
    content = ''
    with open('source/frontend/index.html', 'r', encoding='utf-8') as f:
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
