from typing import Dict, Any, List, Tuple, Optional

import joblib
import numpy as np
import faiss
from pathlib import Path
from source.backend.settings import TOP_K_FILE_SELECT, CACHE_DIR, TOP_K_RERANK, TOP_K_RETRIEVAL, clean_chunk_markdown


def load_index_data(doc_path: Path) -> Tuple[Dict[str, Any], List[str], List[str], Dict[str, Any]]:
    """
    Loads indexed data (FAISS index, chunks, and metadata) from cache.

    Args:
        doc_path (Path): Path to the directory where Markdown files were indexed.
                         Used to locate corresponding cached directories.

    Returns:
        Tuple[Dict[str, Any], List[str], List[str], Dict[str, Any]]:
            - file_indices: Dictionary containing loaded FAISS indices, chunk content, and their metadata.
                            Format: {file_name: (faiss_index, list_of_chunk_contents, list_of_chunk_metadata_dicts)}
            - file_titles: List of titles of all documents (extracted from metadata).
            - file_paths: List of file names for which indices were found.
            - file_meta: Dictionary containing global metadata for each file.
                         Format: {file_name: {'title': ..., 'url': ..., 'author': ...}}
    """

    file_indices: Dict[str, Any] = {}
    file_titles: List[str] = []
    file_paths: List[str] = []
    file_meta: Dict[str, Any] = {}

    cache_base_path = Path(CACHE_DIR)

    for file_path_obj in doc_path.glob("*.md"):
        file_name_str = str(file_path_obj.name)
        tag = file_path_obj.stem
        tag_dir = cache_base_path / tag

        emb_path = tag_dir / "embeddings.npy"
        chunks_content_path = tag_dir / "chunks_content.pkl"
        chunks_metadata_path = tag_dir / "chunks_metadata.pkl"
        index_path = tag_dir / "faiss.index"

        if all(p.exists() for p in (emb_path, chunks_content_path, chunks_metadata_path, index_path)):
            try:
                print(f"Load data for '{file_name_str}' from the cache.")
                # np.load(emb_path)

                saved_chunks_content = joblib.load(chunks_content_path)
                saved_chunks_metadata = joblib.load(chunks_metadata_path)
                index = faiss.read_index(str(index_path))

                if saved_chunks_metadata:
                    first_chunk_metadata = saved_chunks_metadata[0]
                    doc_title = first_chunk_metadata.get('document_title', file_path_obj.stem)
                    doc_url = first_chunk_metadata.get('source_url', '')
                    doc_author = first_chunk_metadata.get('author', '')
                else:
                    doc_title = file_path_obj.stem
                    doc_url = ""
                    doc_author = ""
                file_titles.append(doc_title)
                file_paths.append(file_name_str)
                file_meta[file_name_str] = {
                    'title': doc_title,
                    'url': doc_url,
                    'author': doc_author
                }
                file_indices[file_name_str] = (index, saved_chunks_content, saved_chunks_metadata)

            except Exception as e:
                print(f"Erron on loading cached data for '{file_name_str}': {e}. Does the cache valid?")
                # todo: maybe we can delete cache
                # for p in (emb_path, chunks_content_path, chunks_metadata_path, index_path):
                #     if p.exists(): os.remove(p)
                continue
        else:
            print(f"Data for '{file_name_str}' didn't find. Skipping...")

    return file_indices, file_titles, file_paths, file_meta


def _get_doc_embedding_string(doc_meta: Dict[str, Any]) -> str:
    """
    Generates an embedding string from document metadata.
    Fields to use can be customized.
    """
    title = doc_meta.get('title', '')
    url = doc_meta.get('url', '')
    author = doc_meta.get('author', '')
    return f"Title: {title}. Author: {author}. Url: {url}."


_cached_doc_embeddings: Optional[np.ndarray] = None
_cached_doc_paths: Optional[List[str]] = None
_cached_doc_metadata_source_strings: Optional[List[str]] = None


def select_best_files(
        query: str,
        file_paths: List[str],
        file_meta: Dict[str, Dict[str, str]],
        encoder: Any,
        top_k: int = TOP_K_FILE_SELECT
) -> List[str]:
    """
    Selects the most relevant files based on the query and document metadata,
    using document embedding caching.

    Args:
        query (str): User query.
        file_paths (List[str]): List of names of all available files.
        file_meta (Dict[str, Dict[str, str]]): Dictionary containing global metadata for each file
                                               (e.g., {file_name: {'title': ..., 'url': ..., 'author': ...}}).
        encoder (Any): Model for generating embeddings (bi-encoder).
        top_k (int): Number of top files to select.

    Returns:
        List[str]: List of file names (paths) that are the most relevant.
    """

    global _cached_doc_embeddings, _cached_doc_paths, _cached_doc_metadata_source_strings
    current_doc_metadata_source_strings = []
    for f_path in file_paths:
        current_doc_metadata_source_strings.append(_get_doc_embedding_string(file_meta.get(f_path, {})))

    if (_cached_doc_embeddings is None or
            _cached_doc_paths != file_paths or
            _cached_doc_metadata_source_strings != current_doc_metadata_source_strings):
        # Create a list of strings for embedding from each document's metadata
        doc_embedding_strings = [_get_doc_embedding_string(file_meta.get(f_path, {})) for f_path in file_paths]

        _cached_doc_embeddings = encoder.encode(
            doc_embedding_strings, convert_to_numpy=True, normalize_embeddings=True
        )
        _cached_doc_paths = list(file_paths)
        _cached_doc_metadata_source_strings = list(current_doc_metadata_source_strings)
    else:
        print("Use cached document embeddings.")

    q_emb = encoder.encode([query], convert_to_numpy=True, normalize_embeddings=True)

    # similarity calculation
    scores = np.dot(_cached_doc_embeddings, q_emb.T).squeeze()

    # Select top_k indices
    idxs = np.argsort(-scores)[:top_k]

    selected_paths = [file_paths[i] for i in idxs]
    print(f"Select {len(selected_paths)} files: {selected_paths}")
    return selected_paths


def retrieve_and_rerank(
        bi_encoder: Any,
        cross_encoder: Any,
        faiss_index: Any,
        chunks_content_list: List[str],
        chunks_metadata_list: List[Dict[str, Any]],
        query: str
) -> List[Tuple[str, Dict[str, Any], float]]:
    """
    Extracts relevant chunks from the FAISS index, re-ranks them using cross-encoder
    and returns the ranked chunks along with their metadata and score.

    Args:
        bi_encoder (Any): The bi encoder model for embedding a query.
        cross_encoder (Any): A cross encoder model for reranking.
        faiss_index (Any): FAISS index built on embeddings of chunks.
        chunks_content_list (List[str]): A list of the text content of all chunks.
        chunks_metadata_list (List[Dict[str, Any]]): List of metadata dictionaries of all chunks.
        query (str): The user's input query.

    Returns:
        List[Tuple[str, Dict[str, Any], float]]: A ranked list of tuples,
        where each tuple contains (text_chunk, dictionary_metadata_chunk, relevance_score).
    """
    # 1. Candidate extraction with FAISS (bi-encoder)
    query_emb = bi_encoder.encode([query], convert_to_numpy=True, normalize_embeddings=True)

    # D: distances/estimates, I: indices of found candidates
    D, I = faiss_index.search(query_emb, TOP_K_RETRIEVAL)

    # Get the found chunks and their metadata at indexes I[0]
    # cands_with_meta = [(chunks_content_list[i], chunks_metadata_list[i]) for i in I[0]]

    # Create a list of extracted candidates, including their indexes
    # This is useful for debugging and for keeping track of the original position of the chunks
    retrieved_candidates = []
    for i in I[0]:   # I[0] contains indexes from FAISS for the first (single) query
        if 0 <= i < len(chunks_content_list):
            retrieved_candidates.append({
                'content': chunks_content_list[i],
                'metadata': chunks_metadata_list[i],
                'index': i
            })
        else:
            print(f" Warning: Index {i} is out of range for chunks list (size: {len(chunks_content_list)}). Skip.")

    if not retrieved_candidates:
        return []

    # 2. Preparing pairs for cross-encoder
    pairs = [(query, cand['content']) for cand in retrieved_candidates]

    # 3. Re-ranking with a cross-encoder
    scores = cross_encoder.predict(pairs)

    # 4. Matching scores to candidates and sorting them
    # Now each `ranked_results` element is (score, {'content':..., 'metadata':..., 'index':...})
    ranked_results = []
    for score, cand_data in zip(scores, retrieved_candidates):
        ranked_results.append((score, cand_data))

    # Sorted by grade in descending order
    ranked_results.sort(key=lambda x: x[0], reverse=True)

    # 5. Return TOP_K_RERANK of the best results
    final_ranked_output = []
    for score, cand_data in ranked_results[:TOP_K_RERANK]:
        final_ranked_output.append(
            (cand_data['content'], cand_data['metadata'], float(score)))

    return final_ranked_output
