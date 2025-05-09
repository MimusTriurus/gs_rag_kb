import os
import faiss
import nltk
import joblib
import numpy as np
from pathlib import Path
from llama_cpp import Llama
from sentence_transformers import SentenceTransformer, CrossEncoder

nltk.download("punkt_tab")

# === SETTINGS ===
DOCUMENTS_PATH = "documents/"
CACHE_DIR = "cache/"
EMBED_MODEL_NAME = "bge-large-en"
CROSS_ENCODER_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
GGUF_MODEL_PATH = "model/mistral-7b-instruct-v0.2.Q8_0.gguf"
CHUNK_SIZE = 300
TOP_K_CATEGORIES = 3
TOP_K_RETRIEVAL = 50
TOP_K_RERANK = 5

os.makedirs(CACHE_DIR, exist_ok=True)


# === FUNCTIONS ===

def chunk_text(text, size):
    print("\n[chunk_text] Splitting text into chunks...")
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
    print(f"[chunk_text] Total chunks created: {len(chunks)}")
    return chunks


def parse_documents():
    print("\n[parse_documents] Parsing documents...")
    docs_by_category = {}
    for file in Path(DOCUMENTS_PATH).glob("*.txt"):
        print(f"[parse_documents] Reading file: {file.name}")
        with open(file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            if len(lines) < 3:
                continue
            path = lines[1].strip()
            tags = path.split("/")
            body = "".join(lines[2:])
            chunks = chunk_text(body, CHUNK_SIZE)
            for tag in tags:
                docs_by_category.setdefault(tag, []).extend(
                    [(chunk, file.name) for chunk in chunks]
                )
    print(f"[parse_documents] Categories found: {len(docs_by_category)}")
    return docs_by_category


def build_or_load_index(tag, embed_model, docs):
    print(f"\n[build_or_load_index] Processing category: {tag}")
    tag_dir = os.path.join(CACHE_DIR, tag)
    os.makedirs(tag_dir, exist_ok=True)

    emb_path = os.path.join(tag_dir, "embeddings.npy")
    chunks_path = os.path.join(tag_dir, "chunks.pkl")
    sources_path = os.path.join(tag_dir, "sources.pkl")
    index_path = os.path.join(tag_dir, "faiss.index")

    if all(os.path.exists(p) for p in (emb_path, chunks_path, sources_path, index_path)):
        print("[build_or_load_index] Loading from cache...")
        embeddings = np.load(emb_path)
        chunks = joblib.load(chunks_path)
        sources = joblib.load(sources_path)
        index = faiss.read_index(index_path)
    else:
        print("[build_or_load_index] Generating new embeddings...")
        chunks, sources = zip(*docs)
        embeddings = embed_model.encode(chunks, convert_to_numpy=True, normalize_embeddings=True)
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)

        np.save(emb_path, embeddings)
        joblib.dump(chunks, chunks_path)
        joblib.dump(sources, sources_path)
        faiss.write_index(index, index_path)

    return index, list(chunks), list(sources)


def select_categories(llm, query, tag_list):
    print("\n[select_categories] Selecting relevant categories for the query...")
    prompt = (
        f"You are a classification system. Select up to {TOP_K_CATEGORIES} relevant categories for the given query.\n"
        f"Query: \"{query}\"\n"
        f"Categories: {', '.join(tag_list)}\n"
        f"Answer (comma-separated):"
    )
    resp = llm(prompt, max_tokens=50, stop=["\n"])
    text = resp["choices"][0]["text"]
    selected = [t.strip() for t in text.split(",") if t.strip() in tag_list]
    print(f"[select_categories] Selected categories: {selected}")
    return selected[:TOP_K_CATEGORIES]


def retrieve_and_rerank(bi_model, cross_model, index, chunks, sources, query, query_emb):
    print("\n[retrieve_and_rerank] Retrieving relevant chunks...")
    D, I = index.search(query_emb, TOP_K_RETRIEVAL)
    candidates = [(chunks[i], sources[i]) for i in I[0]]

    print(f"[retrieve_and_rerank] Candidates before rerank: {len(candidates)}")
    docs = [text for text, _ in candidates]
    pairs = [(query, doc) for doc in docs]
    scores = cross_model.predict(pairs)
    ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
    top = ranked[:TOP_K_RERANK]
    print(f"[retrieve_and_rerank] Selected after rerank: {len(top)}")
    return [(text, src) for _, (text, src) in top]


def answer_question(llm, context, query):
    print("\n[answer_question] Generating answer with LLM...")
    prompt = (
        f"You are an assistant. Use only the context below to answer.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\nAnswer:"
    )
    resp = llm(prompt, max_tokens=256, stop=["\n\n"])
    return resp["choices"][0]["text"].strip()


# === MAIN ===

def main():
    print("[main] Loading models...")
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    cross_encoder = CrossEncoder(CROSS_ENCODER_NAME)
    llm = Llama(
        model_path=GGUF_MODEL_PATH,
        embedding=False,
        n_ctx=2048,
        n_threads=4,
        verbose=False
    )

    docs_by_category = parse_documents()
    tag_list = list(docs_by_category.keys())
    tag_indexes = {}

    print("\n[main] Building indices...")
    for tag, docs in docs_by_category.items():
        idx, chunks, sources = build_or_load_index(tag, embed_model, docs)
        tag_indexes[tag] = (idx, chunks, sources)

    print("\n[main] Ready! Enter your query:")
    while True:
        query = input("\nQuery (or 'exit'): ")
        if query.lower() == "exit":
            break

        query_emb = embed_model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        relevant = select_categories(llm, query, tag_list)

        all_results = []
        for tag in relevant:
            idx, chunks, sources = tag_indexes[tag]
            results = retrieve_and_rerank(
                embed_model, cross_encoder, idx, chunks, sources, query, query_emb
            )
            all_results.extend(results)

        context = "\n---\n".join([f"{src}:\n{text}" for text, src in all_results])
        answer = answer_question(llm, context, query)

        print(f"\nAnswer:\n{answer}\n")


if __name__ == "__main__":
    main()
