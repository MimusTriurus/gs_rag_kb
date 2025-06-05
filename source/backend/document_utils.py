import os
import joblib
import numpy as np
import faiss
from pathlib import Path
from source.backend.chunking import split_md_file
from source.backend.settings import TOP_K_FILE_SELECT, CACHE_DIR, TOP_K_RERANK, TOP_K_RETRIEVAL, clean_chunk_markdown


def parse_documents(doc_path, embed_model):
    file_indices = {}
    file_titles = []
    file_paths = []
    file_meta = {}
    for file in Path(doc_path).glob("*.md"):
        lines = file.read_text(encoding="utf-8").splitlines()
        title = lines[0].strip() if lines else file.stem
        url = lines[1].strip() if len(lines) > 1 else ""
        author = lines[2].strip() if len(lines) > 2 else ""
        file_titles.append(title)
        file_paths.append(file.name)
        file_meta[file.name] = (url, author)

        chunks = split_md_file(file, clean_markdown=clean_chunk_markdown)
        idx, ch, src = build_or_load_index_for_file(file.name, embed_model, chunks)
        file_indices[file.name] = (idx, ch, src)

    return file_indices, file_titles, file_paths, file_meta


def build_or_load_index_for_file(file_name, embed_model, chunks):
    tag = Path(file_name).stem
    tag_dir = os.path.join(CACHE_DIR, tag)
    os.makedirs(tag_dir, exist_ok=True)

    emb_path = os.path.join(tag_dir, "embeddings.npy")
    chunks_path = os.path.join(tag_dir, "chunks.pkl")
    sources_path = os.path.join(tag_dir, "sources.pkl")
    index_path = os.path.join(tag_dir, "faiss.index")

    if all(os.path.exists(p) for p in (emb_path, chunks_path, sources_path, index_path)):
        embeddings = np.load(emb_path)
        saved_chunks = joblib.load(chunks_path)
        sources = joblib.load(sources_path)
        index = faiss.read_index(index_path)
    else:
        embeddings = embed_model.encode(chunks, convert_to_numpy=True, normalize_embeddings=True)
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        np.save(emb_path, embeddings)
        joblib.dump(chunks, chunks_path)
        sources = [file_name] * len(chunks)
        joblib.dump(sources, sources_path)
        faiss.write_index(index, index_path)
        saved_chunks = chunks

    return index, saved_chunks, sources


def select_best_files(query, titles, paths, encoder, top_k=TOP_K_FILE_SELECT):
    embs = encoder.encode(titles, convert_to_numpy=True, normalize_embeddings=True)
    q_emb = encoder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    scores = np.dot(embs, q_emb.T).squeeze()
    idxs = np.argsort(-scores)[:top_k]
    return [paths[i] for i in idxs]


def retrieve_and_rerank(bi, cross, index, chunks, sources, query):
    emb = bi.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    D, I = index.search(emb, TOP_K_RETRIEVAL)
    cands = [(chunks[i], sources[i]) for i in I[0]]
    pairs = [(query, text) for text, _ in cands]
    scores = cross.predict(pairs)
    ranked = sorted(zip(scores, cands), key=lambda x: x[0], reverse=True)[:TOP_K_RERANK]
    return [(text, src) for _, (text, src) in ranked]
