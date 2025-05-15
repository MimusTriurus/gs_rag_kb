import os
import re
import faiss
import joblib
import numpy as np
from pathlib import Path
from llama_cpp import Llama
from sentence_transformers import SentenceTransformer, CrossEncoder
from chunking import split_md_file

# === SETTINGS ===
DOCUMENTS_PATH = "documents/"
CACHE_DIR = "cache/"
EMBED_MODEL_NAME = "models/bge-large-en"
CROSS_ENCODER_NAME = "models/ms-marco-MiniLM-L6-v2"
GGUF_MODEL_PATH = "models/mistral-7b-instruct-v0.2.Q8_0.gguf"
MAX_CHUNK_SIZE = 1500
OVERLAP_BLOCKS = 2
TOP_K_RETRIEVAL = 50
TOP_K_RERANK = 2

os.makedirs(CACHE_DIR, exist_ok=True)


# === FUNCTIONS ===

def chunk_text(file_path, max_chunk_size=MAX_CHUNK_SIZE, overlap=OVERLAP_BLOCKS):
    """
    Read a markdown file and split into cleaned chunks using specialized functions.
    """
    md_chunks = split_md_file(file_path, max_chunk_size, overlap, clean_markdown=True)
    print(f"[chunk_text] Split file {os.path.basename(file_path)} into {len(md_chunks)} chunks.")
    return md_chunks


def parse_documents():
    print("\n[parse_documents] Parsing documents and creating chunks...")
    docs = []  # list of (chunk, source)
    for file in Path(DOCUMENTS_PATH).glob("*.md"):
        print(f"[parse_documents] Processing file: {file.name}")
        chunks = chunk_text(file)
        for chunk in chunks:
            docs.append((chunk, file.name))
    print(f"[parse_documents] Total chunks across all documents: {len(docs)}")
    return docs


def build_or_load_index(embed_model, docs):
    print("\n[build_or_load_index] Building or loading global index...")
    emb_path = os.path.join(CACHE_DIR, "embeddings.npy")
    chunks_path = os.path.join(CACHE_DIR, "chunks.pkl")
    sources_path = os.path.join(CACHE_DIR, "sources.pkl")
    index_path = os.path.join(CACHE_DIR, "faiss_global.index")

    if all(os.path.exists(p) for p in (emb_path, chunks_path, sources_path, index_path)):
        print("[build_or_load_index] Loading index from cache...")
        embeddings = np.load(emb_path)
        chunks = joblib.load(chunks_path)
        sources = joblib.load(sources_path)
        index = faiss.read_index(index_path)
    else:
        print("[build_or_load_index] Generating embeddings and building index...")
        chunks, sources = zip(*docs)
        embeddings = embed_model.encode(chunks, convert_to_numpy=True, normalize_embeddings=True)
        index = faiss.IndexFlatIP(embeddings.shape[1])
        # Save index to cache
        print(f"[build_or_load_index] Writing index to {index_path}...")
        faiss.write_index(index, index_path)
        index.add(embeddings)
        np.save(emb_path, embeddings)
        joblib.dump(chunks, chunks_path)
        joblib.dump(sources, sources_path)
        faiss.write_index(index, index_path)

    return index, list(chunks), list(sources)


def retrieve_and_rerank(bi_model, cross_model, index, chunks, sources, query):
    print("\n[retrieve_and_rerank] Retrieving relevant chunks...")
    query_emb = bi_model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    D, I = index.search(query_emb, TOP_K_RETRIEVAL)
    candidates = [(chunks[i], sources[i]) for i in I[0]]

    print(f"[retrieve_and_rerank] {len(candidates)} candidates before rerank.")
    docs = [text for text, _ in candidates]
    pairs = [(query, doc) for doc in docs]
    scores = cross_model.predict(pairs)
    ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
    top = ranked[:TOP_K_RERANK]
    print(f"[retrieve_and_rerank] Selected {len(top)} after rerank.")
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

    docs = parse_documents()
    index, chunks, sources = build_or_load_index(embed_model, docs)

    print("\n[main] Ready! Enter your query:")
    while True:
        query = input("\nQuery (or 'exit'): ")
        if query.lower() == "exit":
            break

        results = retrieve_and_rerank(
            embed_model,
            cross_encoder,
            index,
            chunks,
            sources,
            query
        )

        context = "\n---\n".join([f"{src}:\n{text}" for text, src in results])
        answer = answer_question(llm, context, query)

        print(f"\nAnswer:\n{answer}\n")


if __name__ == "__main__":
    main()
