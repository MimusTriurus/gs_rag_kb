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
TOP_K_FILE_SELECT = 1

os.makedirs(CACHE_DIR, exist_ok=True)


def select_best_files(query, file_titles, file_paths, title_encoder, top_k=TOP_K_FILE_SELECT):
    """
    Select top_k files whose title+path embeddings best match the query.
    """
    title_embs = title_encoder.encode(file_titles, convert_to_numpy=True, normalize_embeddings=True)
    query_emb = title_encoder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    # cosine via inner product
    scores = np.dot(title_embs, query_emb.T).squeeze()
    best_idxs = np.argsort(-scores)[:top_k]
    selected = [file_paths[i] for i in best_idxs]
    print(f"[select_best_files] Selected files: {selected}")
    return selected


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


def build_or_load_index_for_file(file_name, embed_model, chunks):
    """
    Build or load a FAISS index for a single document (file_name).
    """
    tag = Path(file_name).stem
    tag_dir = os.path.join(CACHE_DIR, tag)
    os.makedirs(tag_dir, exist_ok=True)

    emb_path = os.path.join(tag_dir, "embeddings.npy")
    chunks_path = os.path.join(tag_dir, "chunks.pkl")
    sources_path = os.path.join(tag_dir, "sources.pkl")
    index_path = os.path.join(tag_dir, "faiss.index")

    if all(os.path.exists(p) for p in (emb_path, chunks_path, sources_path, index_path)):
        print(f"[build_or_load_index] Loading cache for '{tag}'...")
        embeddings = np.load(emb_path)
        saved_chunks = joblib.load(chunks_path)
        sources = joblib.load(sources_path)
        index = faiss.read_index(index_path)
    else:
        print(f"[build_or_load_index] Building new index for '{tag}'...")
        embeddings = embed_model.encode(chunks, convert_to_numpy=True, normalize_embeddings=True)
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        # Save to cache
        np.save(emb_path, embeddings)
        joblib.dump(chunks, chunks_path)
        joblib.dump(chunks, sources_path)
        faiss.write_index(index, index_path)
        saved_chunks = chunks
        sources = [file_name] * len(chunks)

    return index, saved_chunks, sources


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
    # load models
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    cross_encoder = CrossEncoder(CROSS_ENCODER_NAME)
    llm = Llama(
        model_path=GGUF_MODEL_PATH,
        embedding=False,
        n_ctx=2048,
        n_threads=4,
        verbose=False
    )

    # prepare per-file indices and titles
    file_indices = {}
    file_titles = []
    file_paths = []
    for file in Path(DOCUMENTS_PATH).glob("*.md"):
        lines = file.read_text(encoding="utf-8").splitlines()
        title = lines[0].strip() if lines else file.stem
        path = lines[1].strip() if len(lines) > 1 else ""
        file_titles.append(f"{title} | {path}")
        file_paths.append(file.name)
        # prepare chunks and index
        chunks = chunk_text(file)
        idx, ch, src = build_or_load_index_for_file(file.name, embed_model, chunks)
        file_indices[file.name] = (idx, ch, src)

    print("[main] Ready to answer questions. Type 'exit' to quit.")
    while True:
        query = input("Enter your question: ")
        if query.lower() in ("exit", "quit"):  # allow exit
            print("Exiting...")
            break

        # select best file(s)
        selected_files = select_best_files(query, file_titles, file_paths, embed_model)
        results = []
        for fname in selected_files:
            idx, chunks, sources = file_indices[fname]
            results.extend(
                retrieve_and_rerank(embed_model, cross_encoder, idx, chunks, sources, query)
            )

        # generate answer
        context = "---".join([f"{src}:{text}" for text, src in results])
        answer = answer_question(llm, context, query)
        print(f"Answer:{answer}")


if __name__ == "__main__":
    main()
