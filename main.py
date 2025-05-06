import os
import re

import faiss
import nltk
import joblib
import numpy as np
from pathlib import Path
from llama_cpp import Llama
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

d = nltk.download("punkt_tab")

# === НАСТРОЙКИ ===
DOCUMENTS_PATH = "documents/"
CACHE_DIR = "cache/"
EMBED_MODEL_NAME = "intfloat/e5-small"
GGUF_MODEL_PATH = "model/Meta-Llama-3-8B-Instruct.Q8_0.gguf"
CHUNK_SIZE = 300
TOP_K_CATEGORIES = 3
TOP_K_RETRIEVAL = 5
TOP_K_RERANK = 3

os.makedirs(CACHE_DIR, exist_ok=True)


# === Функции ===

def chunk_text(text, size):
    sentences = nltk.sent_tokenize(text)
    chunks, chunk = [], ""
    for sent in sentences:
        if len(chunk) + len(sent) <= size:
            chunk += " " + sent
        else:
            chunks.append(chunk.strip())
            chunk = sent
    if chunk:
        chunks.append(chunk.strip())
    return chunks


def safe_filename(name, replace_with="_", max_length=255):
    # Удаляем или заменяем недопустимые символы
    safe = re.sub(r'[<>:"/\\|?*\x00-\x1F]', replace_with, name)
    # Удаляем лишние пробелы и точки в конце
    safe = safe.strip().rstrip(". ")
    safe = safe.replace(' ', '_')
    return safe[:max_length]


def parse_documents():
    docs_by_category = {}  # tag -> [(chunk, source)]
    for file in Path(DOCUMENTS_PATH).glob("*.txt"):
        with open(file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            if len(lines) < 3:
                continue
            title = lines[0].strip()
            path = lines[1].strip()
            path = path.replace('Path: ', '')
            tags = path.split("/")  # категориями будут все элементы пути
            body = "".join(lines[2:])
            chunks = chunk_text(body, CHUNK_SIZE)

            for tag in tags:
                tag = safe_filename(tag)
                tag = tag.strip()

                if tag not in docs_by_category:
                    docs_by_category[tag] = []
                docs_by_category[tag].extend([(chunk, file.name) for chunk in chunks])
    return docs_by_category


def build_or_load_index(tag, embed_model, docs):
    tag_dir = os.path.join(CACHE_DIR, tag)
    os.makedirs(tag_dir, exist_ok=True)

    emb_path = os.path.join(tag_dir, "embeddings.npy")
    chunks_path = os.path.join(tag_dir, "chunks.pkl")
    sources_path = os.path.join(tag_dir, "sources.pkl")
    index_path = os.path.join(tag_dir, "faiss.index")

    if all(os.path.exists(p) for p in [emb_path, chunks_path, sources_path, index_path]):
        embeddings = np.load(emb_path)
        chunks = joblib.load(chunks_path)
        sources = joblib.load(sources_path)
        index = faiss.read_index(index_path)
    else:
        chunks, sources = zip(*docs)
        embeddings = embed_model.encode(chunks, convert_to_numpy=True, normalize_embeddings=True)
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)

        np.save(emb_path, embeddings)
        joblib.dump(chunks, chunks_path)
        joblib.dump(sources, sources_path)
        faiss.write_index(index, index_path)

    return index, list(chunks), list(sources), embeddings


def select_categories(llm, query, tag_list):
    prompt = f"""You are an intelligent search engine. 
At the entrance, you receive a user request and a list of categories.
Choose up to 3 categories that best match your query.

Query: "{query}"
Categories: {', '.join(tag_list)}
Response (separated by commas):"""
    response = llm(prompt, max_tokens=50, stop=["\n"])
    text = response["choices"][0]["text"]
    selected = [tag.strip() for tag in text.split(",") if tag.strip() in tag_list]
    return selected[:TOP_K_CATEGORIES]


def retrieve_chunks(index, embeddings, chunks, sources, query_emb):
    D, I = index.search(query_emb, TOP_K_RETRIEVAL)
    return [(chunks[i], sources[i], float(D[0][j])) for j, i in enumerate(I[0])]


def rerank(llm, query, passages):
    scored = []
    for text, source, score in passages:
        prompt = f"Evaluate the relevance of the text to the query:\n\n Query: {query}\text: {text}\score from 1 to 10:"
        out = llm(prompt, max_tokens=5, stop=["\n"])
        try:
            rating = float(out["choices"][0]["text"].strip().split()[0])
        except:
            rating = 0.0
        scored.append((rating, text, source))
    scored.sort(reverse=True)
    return scored[:TOP_K_RERANK]


def answer_question(llm, context, query):
    prompt = f"""You're a useful assistant.
Answer the question using only the context below.

Context:
{context}

Question: {query}
Answer:"""
    response = llm(prompt, max_tokens=256, stop=["\n\n"])
    return response["choices"][0]["text"].strip()


# === Главная логика ===

def main():
    print("Загрузка моделей...")
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    llm = Llama(model_path=GGUF_MODEL_PATH, embedding=False, n_ctx=2048, n_threads=4)

    print("Загрузка и индексирование документов...")
    docs_by_category = parse_documents()
    tag_list = list(docs_by_category.keys())
    tag_indexes = {}

    # Предзагрузка (кэширование) индексов
    for tag in tag_list:
        index, chunks, sources, embeddings = build_or_load_index(tag, embed_model, docs_by_category[tag])
        tag_indexes[tag] = (index, chunks, sources, embeddings)

    print("Готово. Введите ваш запрос:")

    while True:
        query = input("\nВопрос (или 'exit'): ")
        if query.lower() == "exit":
            break

        query_emb = embed_model.encode([query], convert_to_numpy=True, normalize_embeddings=True)

        # Выбор категорий
        relevant_tags = select_categories(llm, query, tag_list)
        print(f"Выбраны категории: {', '.join(relevant_tags)}")

        all_matches = []
        for tag in relevant_tags:
            index, chunks, sources, _ = tag_indexes[tag]
            results = retrieve_chunks(index, None, chunks, sources, query_emb)
            all_matches.extend(results)

        reranked = rerank(llm, query, all_matches)
        context = "\n---\n".join([f"{src}:\n{text}" for _, text, src in reranked])
        answer = answer_question(llm, context, query)

        print(f"\n📚 Ответ:\n{answer}")


if __name__ == "__main__":
    main()
