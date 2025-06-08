import re
from pathlib import Path
from typing import List, Dict, Any, Tuple

import faiss
import joblib
import numpy as np
from ollama import Client
from sentence_transformers import SentenceTransformer

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


def get_token_length(prompt_text: str, model: str = "mistral:7b-instruct") -> int:
    if not prompt_text:
        return -1
    try:
        response = ollama_client.generate(
            model=model,
            prompt=prompt_text,
            options={
                'num_predict': 0
            }
        )
        if 'prompt_eval_count' in response:
            return response['prompt_eval_count']
        elif 'usage' in response and 'prompt_tokens' in response['usage']:
            return response['usage']['prompt_tokens']
        else:
            print("Не удалось найти счетчик токенов промпта в ответе Ollama.")
            return -1
    except Exception as e:
        print(f"Ошибка при запросе к Ollama для подсчета токенов: {e}")
        return -1


def clean_chunk_content(chunk: str) -> str:
    # Удалить изображения
    chunk = re.sub(r'!\[.*?\]\(.*?\)', '', chunk)
    # Преобразовать ссылки: [текст](url) -> текст (url)
    chunk = re.sub(r'\[(.*?)\]\((.*?)\)', r'\1 (\2)', chunk)
    # Удалить жирный/курсивный шрифт (если LLM плохо их обрабатывает, иначе можно оставить)
    chunk = re.sub(r'[\*_]{1,2}(.*?)[\*_]{1,2}', r'\1', chunk)
    # Удалить маркеры списков и нумерацию
    chunk = re.sub(r'^[ \t]*[\*\-+]\s+', '', chunk, flags=re.MULTILINE)
    chunk = re.sub(r'^[ \t]*\d+\.\s+', '', chunk, flags=re.MULTILINE)
    # Удалить символы цитат
    chunk = re.sub(r'^[ \t]*>\s*', '', chunk, flags=re.MULTILINE)
    # Удалить горизонтальные линии
    chunk = re.sub(r'^---+$', '', chunk, flags=re.MULTILINE)
    # Удалить HTML-теги
    chunk = re.sub(r'<.*?>', '', chunk)
    # Сократить множественные переносы строк до максимум двух
    chunk = re.sub(r'\n{3,}', '\n\n', chunk)
    return chunk.strip()


def extract_document_metadata(content: str) -> Tuple[Dict[str, str], str]:
    """
    Извлекает первые три строки как метаданные документа и возвращает оставшийся контент.
    Предполагается, что формат первых трех строк:
    1. Заголовок документа / набор тегов
    2. Ссылка на источник
    3. Автор
    """
    lines = content.split('\n')

    doc_metadata = {
        "document_title": lines[0].strip() if len(lines) > 0 else "N/A",
        "source_url": lines[1].strip() if len(lines) > 1 else "N/A",
        "author": lines[2].strip() if len(lines) > 2 else "N/A"
    }

    remaining_content = '\n'.join(lines[4:])

    return doc_metadata, remaining_content


def split_md_file_effective(
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
        length_function=get_token_length,
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
                # 'start_index': raw_chunk.start_index, # Если splitter это поддерживает
            }
            processed_chunks.append({
                'content': cleaned_content.strip(),
                'metadata': chunk_metadata
            })

    return processed_chunks


def new_build_or_load_index_for_file(
        file_name: str,
        embed_model: Any,
        chunks_content_only: List[str],  # Список только текстового контента чанков
        chunks_metadata_list: List[Dict[str, Any]]  # Список словарей с метаданными чанков
) -> Tuple[Any, List[str], List[Dict[str, Any]]]:
    """
    Строит или загружает FAISS-индекс для заданного файла, сохраняя
    текст чанков и их метаданные в кэше.

    Args:
        file_name (str): Имя файла (например, "document.md").
        embed_model (Any): Модель для генерации эмбеддингов.
        chunks_content_only (List[str]): Список текстового содержимого чанков.
        chunks_metadata_list (List[Dict[str, Any]]): Список словарей с метаданными для каждого чанка.

    Returns:
        Tuple[Any, List[str], List[Dict[str, Any]]]:
            - index: FAISS индекс.
            - saved_chunks_content: Сохраненный (или загруженный) список текстового контента чанков.
            - saved_chunks_metadata: Сохраненный (или загруженный) список метаданных чанков.
    """
    tag = Path(file_name).stem
    tag_dir = Path(CACHE_DIR) / tag  # Использование pathlib для путей
    tag_dir.mkdir(parents=True, exist_ok=True)

    emb_path = tag_dir / "embeddings.npy"
    chunks_content_path = tag_dir / "chunks_content.pkl"  # Путь для текста чанков
    chunks_metadata_path = tag_dir / "chunks_metadata.pkl"  # Путь для метаданных чанков
    index_path = tag_dir / "faiss.index"

    # Проверяем наличие всех файлов кэша
    if all(p.exists() for p in (emb_path, chunks_content_path, chunks_metadata_path, index_path)):
        print(f"Загружаем индекс и чанки для '{file_name}' из кэша.")
        embeddings = np.load(emb_path)
        saved_chunks_content = joblib.load(chunks_content_path)
        saved_chunks_metadata = joblib.load(chunks_metadata_path)
        index = faiss.read_index(str(index_path))  # faiss.read_index ожидает str
    else:
        print(f"Строим индекс и эмбеддинги для '{file_name}'.")
        # Генерируем эмбеддинги только для текстового контента
        embeddings = embed_model.encode(chunks_content_only, convert_to_numpy=True, normalize_embeddings=True)

        index = faiss.IndexFlatIP(
            embeddings.shape[1])  # Используем Cosine Similarity (Inner Product на нормализованных векторах)
        index.add(embeddings)

        # Сохраняем все компоненты
        np.save(emb_path, embeddings)
        joblib.dump(chunks_content_only, chunks_content_path)
        joblib.dump(chunks_metadata_list, chunks_metadata_path)  # Сохраняем список словарей метаданных
        faiss.write_index(index, str(index_path))  # faiss.write_index ожидает str

        saved_chunks_content = chunks_content_only
        saved_chunks_metadata = chunks_metadata_list

    return index, saved_chunks_content, saved_chunks_metadata


def new_parse_documents(doc_path: Path, embed_model: Any) -> Tuple[Dict[str, Any], List[str], List[str], Dict[str, Any]]:
    """
    Парсит Markdown-документы из указанного пути, разбивает их на чанки
    и подготавливает данные для индексации в RAG-системе.

    Args:
        doc_path (Path): Путь к директории с Markdown-файлами.
        embed_model (Any): Модель для генерации эмбеддингов (например, SentenceTransformer).

    Returns:
        Tuple[Dict[str, Any], List[str], List[str], Dict[str, Any]]:
            - file_indices: Словарь с индексами FAISS, чанками и их метаданными по имени файла.
                            Формат: {file_name: (faiss_index, list_of_chunk_contents, list_of_chunk_metadata_dicts)}
            - file_titles: Список заголовков всех документов.
            - file_paths: Список имен всех файлов.
            - file_meta: Словарь с глобальными метаданными для каждого файла.
                            Формат: {file_name: {'title': ..., 'url': ..., 'author': ...}}
    """
    file_indices: Dict[str, Any] = {}
    file_titles: List[str] = []
    file_paths: List[str] = []
    file_meta: Dict[str, Any] = {}

    for file_path_obj in Path(doc_path).glob("*.md"):
        file_name_str = str(file_path_obj.name)

        # Используем новую функцию split_md_file_effective
        # Она возвращает список словарей: [{'content': '...', 'metadata': {...}}, ...]
        chunk_data_list = split_md_file_effective(
            file_path_obj,
            max_chunk_size=DEFAULT_MAX_CHUNK_SIZE,
            chunk_overlap=DEFAULT_CHUNK_OVERLAP,
            clean_markdown_content=CLEAN_MARKDOWN_CONTENT
        )

        if not chunk_data_list:
            print(f"Предупреждение: Файл '{file_name_str}' не содержит валидных чанков после обработки. Пропускаем.")
            continue

        # Глобальные метаданные для документа (берем из первого чанка, т.к. они общие)
        first_chunk_metadata = chunk_data_list[0]['metadata']
        doc_title = first_chunk_metadata.get('document_title', file_path_obj.stem)
        doc_url = first_chunk_metadata.get('source_url', '')
        doc_author = first_chunk_metadata.get('author', '')

        # Заполняем списки и словари для возврата
        file_titles.append(doc_title)
        file_paths.append(file_name_str)
        file_meta[file_name_str] = {
            'title': doc_title,
            'url': doc_url,
            'author': doc_author
        }

        # Отделяем контент чанков от их метаданных для дальнейшей обработки
        chunks_content_only = [cd['content'] for cd in chunk_data_list]
        chunks_metadata_only = [cd['metadata'] for cd in chunk_data_list]

        # Передаем обе части (контент и метаданные) в build_or_load_index_for_file
        idx, saved_chunks_content, saved_chunks_metadata = new_build_or_load_index_for_file(
            file_name_str, embed_model, chunks_content_only, chunks_metadata_only
        )

        # Сохраняем индекс, список контента чанков и список метаданных чанков
        file_indices[file_name_str] = (idx, saved_chunks_content, saved_chunks_metadata)

    return file_indices, file_titles, file_paths, file_meta


if __name__ == '__main__':
    try:
        embed_model = SentenceTransformer(EMBED_MODEL_NAME)
        file_indices, file_titles, file_paths, file_meta = new_parse_documents(DOCUMENTS_PATH, embed_model)
    except Exception as e:
        print(e)
