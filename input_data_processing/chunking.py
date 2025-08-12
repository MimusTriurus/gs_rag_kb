import re
from pathlib import Path
from typing import List, Dict, Any, Tuple

import faiss
import joblib
import numpy as np
from langchain_text_splitters import MarkdownHeaderTextSplitter
from ollama import Client
from sentence_transformers import SentenceTransformer
import frontmatter

from source.backend.settings import (
    OLLAMA_BASE_URL,
    CACHE_DIR,
    DEFAULT_MAX_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    CLEAN_MARKDOWN_CONTENT,
    EMBED_MODEL_NAME,
    DOCUMENTS_PATH
)
from langchain.text_splitter import MarkdownTextSplitter

ollama_client = Client(
    host=OLLAMA_BASE_URL
)


def clean_chunk_content(chunk: str) -> str:
    # delete images
    chunk = re.sub(r'!\[.*?\]\(.*?\)', '', chunk)
    # transform links: [текст](url) -> текст (url)
    chunk = re.sub(r'\[(.*?)\]\((.*?)\)', r'\1 (\2)', chunk)
    # remove bold\italic
    chunk = re.sub(r'[\*_]{1,2}(.*?)[\*_]{1,2}', r'\1', chunk)
    # remove numeration\lists
    chunk = re.sub(r'^[ \t]*[\*\-+]\s+', '', chunk, flags=re.MULTILINE)
    chunk = re.sub(r'^[ \t]*\d+\.\s+', '', chunk, flags=re.MULTILINE)
    # remove quotes
    chunk = re.sub(r'^[ \t]*>\s*', '', chunk, flags=re.MULTILINE)
    # remove horizontal lines
    chunk = re.sub(r'^---+$', '', chunk, flags=re.MULTILINE)
    # remove html tags
    chunk = re.sub(r'<.*?>', '', chunk)
    # reduce amount of \n
    chunk = re.sub(r'\n{3,}', '\n\n', chunk)
    return chunk.strip()


def extract_document_metadata(content: str) -> Tuple[Dict[str, str], str]:
    processed_content = frontmatter.loads(content)
    return processed_content.metadata, processed_content.content


def split_md_file(
        file_path: str,
        max_chunk_size: int = 500,
        chunk_overlap: int = 100,
        clean_markdown_content: bool = True
) -> List[Dict[str, Any]]:
    with open(file_path, 'r', encoding='utf-8') as f:
        full_content = f.read()
    doc_metadata, md_content_for_chunking = extract_document_metadata(full_content)

    doc_metadata['source_file'] = file_path

    markdown_splitter = MarkdownTextSplitter(
        chunk_size=max_chunk_size,
        chunk_overlap=chunk_overlap,
        # todo: need to investigate
        # length_function=get_token_length,
    )

    raw_chunks = markdown_splitter.split_text(md_content_for_chunking)

    processed_chunks: List[Dict[str, Any]] = []

    for i, raw_chunk in enumerate(raw_chunks):
        section_heading = ""
        match = re.search(r'^(#+)\s*(.*)', raw_chunk, re.MULTILINE)
        if match:
            heading_level = len(match.group(1))
            heading_text = match.group(2).strip()
            section_heading = f"[{'#' * heading_level}] {heading_text}"
        cleaned_content = raw_chunk
        if clean_markdown_content:
            cleaned_content = clean_chunk_content(raw_chunk)
        if cleaned_content.strip():
            chunk_metadata = {
                **doc_metadata,
                'chunk_id': f"{file_path}_{i}",
                'section_heading': section_heading if section_heading else None,
                # 'start_index': raw_chunk.start_index, # todo: only if splitter supports that. need to investigate
            }
            processed_chunks.append({
                'content': cleaned_content.strip(),
                'metadata': chunk_metadata
            })

    return processed_chunks


def split_md_file_by_header(
        file_path: str,
        max_chunk_size: int = 500,
        chunk_overlap: int = 100,
        clean_markdown_content: bool = True
) -> List[Dict[str, Any]]:
    with open(file_path, 'r', encoding='utf-8') as f:
        full_content = f.read()
    doc_metadata, md_content_for_chunking = extract_document_metadata(full_content)

    doc_metadata['source_file'] = file_path

    headers_to_split_on = [
        ("#", "0"),
        ("##", "1"),
        ("###", "2"),
    ]

    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=True
    )

    raw_chunks = markdown_splitter.split_text(md_content_for_chunking)

    processed_chunks: List[Dict[str, Any]] = []

    for i, chunk in enumerate(raw_chunks):

        chunk_metadata = chunk.metadata
        metadata_content = ''
        for k, v in chunk_metadata.items():
            metadata_content += f'{v}\n'

        raw_chunk = chunk.page_content
        section_heading = ""
        match = re.search(r'^(#+)\s*(.*)', raw_chunk, re.MULTILINE)
        if match:
            heading_level = len(match.group(1))
            heading_text = match.group(2).strip()
            section_heading = f"[{'#' * heading_level}] {heading_text}"
        cleaned_content = raw_chunk
        if clean_markdown_content:
            cleaned_content = clean_chunk_content(raw_chunk)

        cleaned_content = f"{doc_metadata.get('title', '')}\n{metadata_content}{cleaned_content.strip()}"

        if cleaned_content:
            chunk_metadata = {
                **doc_metadata,
                'chunk_id': f"{file_path}_{i}",
                'section_heading': section_heading if section_heading else None,
                # 'start_index': raw_chunk.start_index, # todo: only if splitter supports that. need to investigate
            }
            processed_chunks.append({
                'content': cleaned_content,
                'metadata': chunk_metadata
            })

    return processed_chunks


def split_md_by_headers(md_text):
    # Разделяем по заголовкам первого уровня
    level1_blocks = re.split(r'(?m)(?=^#\s)', md_text.strip())

    chunks = []
    for block in level1_blocks:
        if not block.strip():
            continue

        # Извлекаем заголовок первого уровня
        level1_match = re.match(r'^#\s+(.*)', block.strip())
        if not level1_match:
            continue
        level1_title = level1_match.group(1).strip()

        # Убираем заголовок первого уровня из текста блока
        block_body = block.strip()

        # Проверяем, есть ли заголовки второго уровня
        level2_blocks = re.split(r'(?m)(?=^##\s)', block_body)

        if len(level2_blocks) > 1:
            # Разбиваем по ## и добавляем в каждый заголовок уровень 1
            for subblock in level2_blocks:
                if not subblock.strip():
                    continue
                # Добавляем инфо о родителе
                subblock = f"# {level1_title}\n" + subblock.strip()
                chunks.append(subblock)
        else:
            chunks.append(block_body)

    return chunks


def build_index_for_file(
        file_name: str,
        embed_model: Any,
        chunks_content_only: List[str],
        chunks_metadata_list: List[Dict[str, Any]]
) -> Tuple[Any, List[str], List[Dict[str, Any]]]:
    """
       Constructs a FAISS index for a given file, storing the chunk text and its metadata in the cache.
       the text of the chunks and their metadata in the cache.

       Args:
           file_name (str): The name of the file (e.g., "document.md").
           embed_model (Any): The model for embedding generation.
           chunks_content_only (List[str]): A list of the textual content of the chunks.
           chunks_metadata_list (List[Dict[str, Any]]): A list of metadata dictionaries for each chunk.

       Returns:
           Tuple[Any, List[str], List[Dict[str, Any]]]:
               - index: FAISS index.
               - saved_chunks_content: A saved (or loaded) list of the textual content of the chunks.
               - saved_chunks_metadata: A saved (or downloaded) list of chunks metadata.
    """
    tag = Path(file_name).stem
    tag_dir = Path(CACHE_DIR) / tag
    tag_dir.mkdir(parents=True, exist_ok=True)

    emb_path = tag_dir / "embeddings.npy"
    chunks_content_path = tag_dir / "chunks_content.pkl"
    chunks_metadata_path = tag_dir / "chunks_metadata.pkl"
    index_path = tag_dir / "faiss.index"

    print(f"Build index and embeddings for '{file_name}'.")
    embeddings = embed_model.encode(chunks_content_only, convert_to_numpy=True, normalize_embeddings=True)

    index = faiss.IndexFlatIP(
        embeddings.shape[1])
    index.add(embeddings)

    np.save(emb_path, embeddings)
    joblib.dump(chunks_content_only, chunks_content_path)
    joblib.dump(chunks_metadata_list, chunks_metadata_path)
    faiss.write_index(index, str(index_path))

    saved_chunks_content = chunks_content_only
    saved_chunks_metadata = chunks_metadata_list

    return index, saved_chunks_content, saved_chunks_metadata


def parse_documents(doc_path: Path, embed_model: Any) -> Tuple[Dict[str, Any], List[str], List[str], Dict[str, Any]]:
    """
       Parses Markdown documents from the specified path, breaks them into chunks
       and prepares the data for indexing in the RAG system.

       Args:
           doc_path (Path): The path to the directory containing the Markdown files.
           embed_model (Any): The model for embedding generation (e.g. SentenceTransformer).

       Returns:
           Tuple[Dict[str, Any], List[str], List[str], Dict[str, Any]]:
               - file_indices: Dictionary with FAISS indexes, chunks, and their metadata by file name.
                               Format: {file_name: (faiss_index, list_of_chunk_contents, list_of_chunk_metadata_dicts)}
               - file_titles: List of titles of all documents.
               - file_paths: List of all file names.
               - file_meta: A dictionary with global metadata for each file.
                               Format: {file_name: {'title': ..., 'url': ..., 'author': ...}}
       """
    file_indices: Dict[str, Any] = {}
    file_titles: List[str] = []
    file_paths: List[str] = []
    file_meta: Dict[str, Any] = {}
    try:
        for file_path_obj in doc_path.glob("*.md"):
            file_name_str = str(file_path_obj.name)
            # for testrails
            chunk_data_list = split_md_file_by_header(
            #chunk_data_list = split_md_file(
                file_path_obj,
                max_chunk_size=DEFAULT_MAX_CHUNK_SIZE,
                chunk_overlap=DEFAULT_CHUNK_OVERLAP,
                clean_markdown_content=CLEAN_MARKDOWN_CONTENT,
            )

            if not chunk_data_list:
                print(f"Warning: File '{file_name_str}' doesn't have valid chunks after processing. Skipping...")
                continue

            first_chunk_metadata = chunk_data_list[0]['metadata']
            doc_title = first_chunk_metadata.get('document_title', file_path_obj.stem)
            doc_url = first_chunk_metadata.get('source_url', '')
            doc_author = first_chunk_metadata.get('author', '')

            file_titles.append(doc_title)
            file_paths.append(file_name_str)
            file_meta[file_name_str] = {
                'title': doc_title,
                'url': doc_url,
                'author': doc_author
            }

            chunks_content_only = [cd['content'] for cd in chunk_data_list]
            chunks_metadata_only = [cd['metadata'] for cd in chunk_data_list]

            idx, saved_chunks_content, saved_chunks_metadata = build_index_for_file(
                file_name_str, embed_model, chunks_content_only, chunks_metadata_only
            )

            file_indices[file_name_str] = (idx, saved_chunks_content, saved_chunks_metadata)
    except Exception as e:
        print(f'{file_name_str} Error: {e}')
    return file_indices, file_titles, file_paths, file_meta


if __name__ == "__main__1":
    md_path = Path("D:/Projects/Python/gs_rag_kb/documents/tech_writer_gt/gametech_digest_2022_2023.md")
    md_text = md_path.read_text(encoding="utf-8")

    chunks = split_md_by_headers(md_text)

    for i, ch in enumerate(chunks, 1):
        print(f"\n--- CHUNK {i} ---\n{ch}")


if __name__ == '__main__':
    try:
        embed_model = SentenceTransformer(EMBED_MODEL_NAME)
        print(f'==> Chunking docs in {DOCUMENTS_PATH} to {CACHE_DIR}...')
        file_indices, file_titles, file_paths, file_meta = parse_documents(Path(DOCUMENTS_PATH), embed_model)
    except Exception as e:
        print(e)
