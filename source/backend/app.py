import os
import asyncio
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from sentence_transformers import SentenceTransformer, CrossEncoder
from starlette.responses import JSONResponse

from source.backend.document_utils import parse_documents, select_best_files, retrieve_and_rerank, load_index_data, \
    retrieve_and_rerank_new, select_best_files_new
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
#file_indices, file_titles, file_paths, file_meta = parse_documents(DOCUMENTS_PATH, embed_model)
file_indices, file_titles, file_paths, file_meta = load_index_data(Path(DOCUMENTS_PATH))


@app.post("/rag/search", response_model=ResponseOutput, include_in_schema=False)
async def rag_search(input_data: QueryInput):
    """
    Выполняет поиск по базе знаний, используя RAG-пайплайн.
    """
    user_query = input_data.query

    # 1. Очистка/уточнение запроса пользователя (если включено)
    query = await run_in_thread(refine_user_prompt, user_query, LLM_MODEL) \
        if need_2_refine_query else user_query

    # 2. Выбор наиболее релевантных файлов
    # Здесь `file_titles` и `file_paths` содержат высокоуровневую информацию о документах
    selected_files = await run_in_thread(
        select_best_files_new, query, file_paths, file_meta, embed_model  # embed_model должен быть доступен
    )

    all_retrieved_results: List[Tuple[str, Dict[str, Any], float]] = []  # Теперь храним (текст, метаданные, оценка)

    # 3. Извлечение и переранжирование чанков для каждого выбранного файла
    for fname in selected_files:
        # Извлекаем все три компонента из file_indices
        faiss_index, chunks_content_list, chunks_metadata_list = file_indices[fname]

        # Вызываем retrieve_and_rerank с новой сигнатурой
        # bi_encoder и cross_encoder должны быть доступны здесь (глобально или через DI)
        retrieved_and_ranked_for_file = await run_in_thread(
            retrieve_and_rerank_new, embed_model, cross_encoder,
            # Замените embed_model и cross_encoder на ваши фактические переменные
            faiss_index, chunks_content_list, chunks_metadata_list, query
        )
        all_retrieved_results.extend(retrieved_and_ranked_for_file)

    # 4. Сортировка всех полученных результатов по релевантности (оценке)
    # Это важно, так как retrieve_and_rerank возвращает TOP_K_RERANK для каждого файла.
    # Нам нужен глобальный топ-K.
    all_retrieved_results.sort(key=lambda x: x[2], reverse=True)  # x[2] это оценка (float)

    context_parts: List[str] = []
    best_url: str = ''
    best_author: str = ''

    # 5. Формирование контекста для LLM и извлечение метаданных для ответа
    # Проходим по отсортированным результатам и собираем контекст.
    # Берем URL и автора из самого релевантного чанка (первого в отсортированном списке),
    # если они еще не установлены.
    for text_content, metadata_dict, score in all_retrieved_results:
        context_parts.append(text_content)  # Добавляем текст чанка в контекст

        # Берем URL и автора из *первого* самого релевантного чанка,
        # который содержит эту информацию.
        if not best_url and metadata_dict.get('source_url'):
            best_url = metadata_dict['source_url']
        if not best_author and metadata_dict.get('author'):
            best_author = metadata_dict['author']

        # Можно добавить условие для ограничения размера контекста, чтобы не превысить LLM-лимит
        # if get_token_length('\n---\n'.join(context_parts)) > SOME_LLM_MAX_CONTEXT_TOKENS:
        #     break

    # Объединяем части контекста
    context = '\n---\n'.join(context_parts) if context_parts else ""

    # 6. Получение ответа от LLM
    answer = await run_in_thread(answer_question, context, query, LLM_MODEL)

    # 7. Обработка случая "нет информации"
    if missing_info_text in answer:
        await run_in_thread(insert_not_found_query, db_path, user_query)  # db_path должен быть доступен
        answer = no_info_in_knowledge_base_message
        best_url = ''  # Сбрасываем URL и автора, если информация не найдена
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
