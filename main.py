import os
import faiss
import nltk
import joblib
import numpy as np
from pathlib import Path
from llama_cpp import Llama
from sentence_transformers import SentenceTransformer

nltk.download("punkt_tab")

# === НАСТРОЙКИ ===
DOCUMENTS_PATH = "documents/"
CACHE_DIR = "cache/"
EMBED_MODEL_NAME = "bge-large-en"
GGUF_MODEL_PATH = "model/Meta-Llama-3-8B-Instruct.Q8_0.gguf"
CHUNK_SIZE = 300
TOP_K_CATEGORIES = 3
TOP_K_RETRIEVAL = 5
TOP_K_RERANK = 3

os.makedirs(CACHE_DIR, exist_ok=True)


# === Функции для разбора и чанкинга документов ===

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


def parse_documents():
    """
    Читает .txt файлы, где
     - 1-я строка = заголовок
     - 2-я строка = путь/тэги (GameDesign/Combat/Weapons)
     - дальше = тело
    """
    docs_by_category = {}
    for file in Path(DOCUMENTS_PATH).glob("*.txt"):
        with open(file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            if len(lines) < 3:
                continue
            path = lines[1].strip()
            tags = path.split("/")  # тэги из пути
            body = "".join(lines[2:])
            chunks = chunk_text(body, CHUNK_SIZE)
            for tag in tags:
                docs_by_category.setdefault(tag, []).extend(
                    [(chunk, file.name) for chunk in chunks]
                )
    return docs_by_category


# === Индексация и кэширование по категориям ===

def build_or_load_index(tag, embed_model, docs):
    """
    Для каждого тега создаём свой FAISS-индекс и кэшируем:
      embeddings.npy, chunks.pkl, sources.pkl, faiss.index
    """
    tag_dir = os.path.join(CACHE_DIR, tag)
    os.makedirs(tag_dir, exist_ok=True)

    emb_path = os.path.join(tag_dir, "embeddings.npy")
    chunks_path = os.path.join(tag_dir, "chunks.pkl")
    sources_path = os.path.join(tag_dir, "sources.pkl")
    index_path = os.path.join(tag_dir, "faiss.index")

    if all(os.path.exists(p) for p in (emb_path, chunks_path, sources_path, index_path)):
        embeddings = np.load(emb_path)
        chunks = joblib.load(chunks_path)
        sources = joblib.load(sources_path)
        index = faiss.read_index(index_path)
    else:
        chunks, sources = zip(*docs)
        # ✂️ BGE: используем BGE для эмбеддингов
        embeddings = embed_model.encode(chunks, convert_to_numpy=True, normalize_embeddings=True)
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)

        np.save(emb_path, embeddings)
        joblib.dump(chunks, chunks_path)
        joblib.dump(sources, sources_path)
        faiss.write_index(index, index_path)

    return index, list(chunks), list(sources), embeddings


# === Определение релевантных категорий через LLM ===

def select_categories(llm, query, tag_list):
    prompt = (
        f"Ты — интеллектуальная система классификации.\n"
        f"На вход ты получаешь запрос и список категорий.\n"
        f"Выбери до {TOP_K_CATEGORIES} категорий, наиболее подходящих к запросу.\n\n"
        f"Запрос: «{query}»\n"
        f"Категории: {', '.join(tag_list)}\n"
        f"Ответ (через запятую):"
    )
    resp = llm(prompt, max_tokens=50, stop=["\n"])
    text = resp["choices"][0]["text"]
    selected = [t.strip() for t in text.split(",") if t.strip() in tag_list]
    return selected[:TOP_K_CATEGORIES]


# === Retrieval & Reranking ===

def retrieve_chunks(index, chunks, sources, query_emb):
    D, I = index.search(query_emb, TOP_K_RETRIEVAL)
    return [(chunks[i], sources[i], float(D[0][j])) for j, i in enumerate(I[0])]


def rerank(llm, query, passages):
    scored = []
    for text, source, _ in passages:
        prompt = (
            f"Оцени релевантность этого фрагмента к запросу от 1 до 10.\n\n"
            f"Запрос: {query}\n"
            f"Текст: {text}\n"
            f"Оценка:"
        )
        out = llm(prompt, max_tokens=5, stop=["\n"])
        try:
            score = float(out["choices"][0]["text"].strip().split()[0])
        except:
            score = 0.0
        scored.append((score, text, source))
    scored.sort(reverse=True, key=lambda x: x[0])
    return scored[:TOP_K_RERANK]


# === Генерация ответа ===

def answer_question(llm, context, query):
    prompt = (
        f"Ты — полезный ассистент.\n"
        f"Используй только предоставленный контекст.\n\n"
        f"Контекст:\n{context}\n\n"
        f"Вопрос: {query}\nОтвет:"
    )
    resp = llm(prompt, max_tokens=256, stop=["\n\n"])
    return resp["choices"][0]["text"].strip()


# === Main ===

def main():
    print("Загрузка моделей…")
    # ✂️ BGE: инициализируем SentenceTransformer на BGE
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    llm = Llama(
        model_path=GGUF_MODEL_PATH,
        embedding=False,
        n_ctx=2048,
        n_threads=4,
        verbose=False
    )

    print("Парсинг документов и подготовка индексов…")
    docs_by_category = parse_documents()
    tag_list = list(docs_by_category.keys())
    tag_indexes = {}

    # Построение/загрузка индексов по категориям
    for tag in tag_list:
        idx, chunks, sources, _ = build_or_load_index(tag, embed_model, docs_by_category[tag])
        tag_indexes[tag] = (idx, chunks, sources)

    print("Система готова! Введите ваш вопрос.")

    while True:
        query = input("\nВопрос (или 'exit'): ")
        if query.lower() == "exit":
            break

        # 1) Эмбеддинг запроса через BGE
        query_emb = embed_model.encode([query], convert_to_numpy=True, normalize_embeddings=True)

        # 2) Определение релевантных категорий через LLM
        relevant = select_categories(llm, query, tag_list)
        print(f"→ Выбраны категории: {', '.join(relevant)}")

        # 3) Поиск и сбор кандидатов
        candidates = []
        for tag in relevant:
            idx, chunks, sources = tag_indexes[tag]
            candidates += retrieve_chunks(idx, chunks, sources, query_emb)

        # 4) Reranking через LLM
        top_passages = rerank(llm, query, candidates)

        # 5) Формирование контекста и генерация ответа
        context = "\n---\n".join(f"{src}:\n{text}" for _, text, src in top_passages)
        answer = answer_question(llm, context, query)

        print(f"\n📚 Ответ:\n{answer}")


if __name__ == "__main__":
    main()
