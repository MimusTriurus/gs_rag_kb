import os
from typing import Dict, Any, List, Tuple

import joblib
import numpy as np
import faiss
from pathlib import Path
from source.backend.chunking import split_md_file
from source.backend.settings import TOP_K_FILE_SELECT, CACHE_DIR, TOP_K_RERANK, TOP_K_RETRIEVAL, clean_chunk_markdown


def load_index_data(doc_path: Path) -> Tuple[Dict[str, Any], List[str], List[str], Dict[str, Any]]:
    """
    Загружает индексированные данные (FAISS индекс, чанки и метаданные) из кэша.

    Args:
        doc_path (Path): Путь к директории, откуда были проиндексированы Markdown-файлы.
                         Используется для нахождения соответствующих кэшированных директорий.

    Returns:
        Tuple[Dict[str, Any], List[str], List[str], Dict[str, Any]]:
            - file_indices: Словарь с загруженными индексами FAISS, контентом чанков и их метаданными.
                            Формат: {file_name: (faiss_index, list_of_chunk_contents, list_of_chunk_metadata_dicts)}
            - file_titles: Список заголовков всех документов (извлечены из метаданных).
            - file_paths: Список имен всех файлов, для которых найдены индексы.
            - file_meta: Словарь с глобальными метаданными для каждого файла.
                            Формат: {file_name: {'title': ..., 'url': ..., 'author': ...}}
    """
    file_indices: Dict[str, Any] = {}
    file_titles: List[str] = []
    file_paths: List[str] = []
    file_meta: Dict[str, Any] = {}

    cache_base_path = Path(CACHE_DIR)  # Базовый путь для кэша

    # Итерируем по всем MD-файлам в исходной директории,
    # чтобы найти соответствующие им кэшированные данные.
    for file_path_obj in doc_path.glob("*.md"):
        file_name_str = str(file_path_obj.name)
        tag = file_path_obj.stem  # Имя файла без расширения
        tag_dir = cache_base_path / tag  # Директория кэша для этого файла

        emb_path = tag_dir / "embeddings.npy"
        chunks_content_path = tag_dir / "chunks_content.pkl"
        chunks_metadata_path = tag_dir / "chunks_metadata.pkl"
        index_path = tag_dir / "faiss.index"

        # Проверяем, существуют ли все необходимые файлы для этого документа в кэше
        if all(p.exists() for p in (emb_path, chunks_content_path, chunks_metadata_path, index_path)):
            try:
                print(f"Загружаем данные для '{file_name_str}' из кэша.")
                # Эмбеддинги не нужны напрямую здесь, но их наличие является индикатором полноты
                # np.load(emb_path)

                saved_chunks_content = joblib.load(chunks_content_path)
                saved_chunks_metadata = joblib.load(chunks_metadata_path)
                index = faiss.read_index(str(index_path))

                # Извлекаем глобальные метаданные из первого чанка (они общие для всего документа)
                if saved_chunks_metadata:
                    first_chunk_metadata = saved_chunks_metadata[0]
                    doc_title = first_chunk_metadata.get('document_title', file_path_obj.stem)
                    doc_url = first_chunk_metadata.get('source_url', '')
                    doc_author = first_chunk_metadata.get('author', '')
                else:  # Если метаданных нет, используем базовые значения
                    doc_title = file_path_obj.stem
                    doc_url = ""
                    doc_author = ""

                # Заполняем выходные списки и словари
                file_titles.append(doc_title)
                file_paths.append(file_name_str)
                file_meta[file_name_str] = {
                    'title': doc_title,
                    'url': doc_url,
                    'author': doc_author
                }
                file_indices[file_name_str] = (index, saved_chunks_content, saved_chunks_metadata)

            except Exception as e:
                print(f"Ошибка при загрузке данных для '{file_name_str}' из кэша: {e}. Возможно, кэш поврежден.")
                # Можно удалить поврежденный кэш для этого файла здесь:
                # for p in (emb_path, chunks_content_path, chunks_metadata_path, index_path):
                #     if p.exists(): os.remove(p)
                continue
        else:
            print(f"Данные для '{file_name_str}' не найдены в кэше. Необходимо индексировать.")
            # Этот файл будет пропущен или должен быть обработан parse_documents

    return file_indices, file_titles, file_paths, file_meta


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


def _get_doc_embedding_string(doc_meta: Dict[str, Any]) -> str:
    """
    Формирует строку для эмбеддинга из метаданных документа.
    Можно настроить, какие поля использовать.
    """
    title = doc_meta.get('title', '')
    url = doc_meta.get('url', '')
    author = doc_meta.get('author', '')
    # Объединяем важные метаданные в одну строку для эмбеддинга.
    # Это позволяет encoder'у учитывать больше контекста, чем просто заголовок.
    return f"Title: {title}. Author: {author}. Url: {url}."


_cached_doc_embeddings: np.ndarray = None
_cached_doc_paths: List[str] = None # Храним пути кэшированных документов
_cached_doc_metadata_source_strings: List[str] = None


def select_best_files_new(
        query: str,
        file_paths: List[str],  # Список имен файлов (например, "document.md")
        file_meta: Dict[str, Dict[str, str]],  # Словарь с глобальными метаданными для каждого файла
        encoder: Any,  # Ваш би-энкодер
        top_k: int = TOP_K_FILE_SELECT
) -> List[str]:
    """
    Отбирает наиболее релевантные файлы на основе запроса и метаданных документа,
    используя кэширование эмбеддингов документов.

    Args:
        query (str): Запрос пользователя.
        file_paths (List[str]): Список имен всех доступных файлов.
        file_meta (Dict[str, Dict[str, str]]): Словарь с глобальными метаданными для каждого файла
                                                (например, {file_name: {'title': ..., 'url': ..., 'author': ...}}).
        encoder (Any): Модель для генерации эмбеддингов (би-энкодер).
        top_k (int): Количество лучших файлов для отбора.

    Returns:
        List[str]: Список имен файлов (путей), которые являются наиболее релевантными.
    """
    global _cached_doc_embeddings, _cached_doc_paths, _cached_doc_metadata_source_strings

    # Проверяем, нужно ли перегенерировать кэш эмбеддингов документов
    # Кэш перегенерируется, если он пуст, или если список файлов изменился
    # или если изменились сами строки метаданных, из которых генерируются эмбеддинги.
    current_doc_metadata_source_strings = []
    for f_path in file_paths:
        current_doc_metadata_source_strings.append(_get_doc_embedding_string(file_meta.get(f_path, {})))

    if (_cached_doc_embeddings is None or
            _cached_doc_paths != file_paths or
            _cached_doc_metadata_source_strings != current_doc_metadata_source_strings):

        print("Перегенерируем кэш эмбеддингов документов...")
        # Создаем список строк для эмбеддинга из метаданных каждого документа
        doc_embedding_strings = [_get_doc_embedding_string(file_meta.get(f_path, {})) for f_path in file_paths]

        _cached_doc_embeddings = encoder.encode(
            doc_embedding_strings, convert_to_numpy=True, normalize_embeddings=True
        )
        _cached_doc_paths = list(file_paths)  # Сохраняем копию
        _cached_doc_metadata_source_strings = list(current_doc_metadata_source_strings)
        print(f"Кэш эмбеддингов документов содержит {len(_cached_doc_embeddings)} элементов.")
    else:
        print("Используем кэшированные эмбеддинги документов.")

    # Эмбеддинг запроса
    q_emb = encoder.encode([query], convert_to_numpy=True, normalize_embeddings=True)

    # Вычисление косинусного сходства
    # .squeeze() удаляет оси размера 1 (превращает (N, 1) в (N,))
    scores = np.dot(_cached_doc_embeddings, q_emb.T).squeeze()

    # Отбор top_k индексов по убыванию сходства
    # np.argsort(-scores) возвращает индексы, которые отсортировали бы массив по убыванию
    idxs = np.argsort(-scores)[:top_k]

    # Возвращаем пути к файлам по отобранным индексам
    selected_paths = [file_paths[i] for i in idxs]
    print(f"Отобрано {len(selected_paths)} файлов: {selected_paths}")
    return selected_paths


def retrieve_and_rerank(bi, cross, index, chunks, sources, query):
    emb = bi.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    D, I = index.search(emb, TOP_K_RETRIEVAL)
    cands = [(chunks[i], sources[i]) for i in I[0]]
    pairs = [(query, text) for text, _ in cands]
    scores = cross.predict(pairs)
    ranked = sorted(zip(scores, cands), key=lambda x: x[0], reverse=True)[:TOP_K_RERANK]
    return [(text, src) for _, (text, src) in ranked]


def retrieve_and_rerank_new(
        bi_encoder: Any,  # Ваш bi-encoder модель (например, SentenceTransformer)
        cross_encoder: Any,  # Ваш cross-encoder модель
        faiss_index: Any,  # FAISS индекс (idx из file_indices)
        chunks_content_list: List[str],  # Список текстового контента чанков (saved_chunks_content)
        chunks_metadata_list: List[Dict[str, Any]],  # Список словарей метаданных чанков (saved_chunks_metadata)
        query: str
) -> List[Tuple[str, Dict[str, Any], float]]:  # Теперь возвращаем (текст, метаданные, оценка)
    """
    Извлекает релевантные чанки из FAISS индекса, переранжирует их с помощью cross-encoder
    и возвращает отранжированные чанки вместе с их метаданными и оценкой.

    Args:
        bi_encoder (Any): Би-энкодер модель для эмбеддинга запроса.
        cross_encoder (Any): Кросс-энкодер модель для переранжирования.
        faiss_index (Any): FAISS индекс, построенный на эмбеддингах чанков.
        chunks_content_list (List[str]): Список текстового контента всех чанков.
        chunks_metadata_list (List[Dict[str, Any]]): Список словарей метаданных всех чанков.
        query (str): Входной запрос пользователя.

    Returns:
        List[Tuple[str, Dict[str, Any], float]]: Отранжированный список кортежей,
        где каждый кортеж содержит (текст_чанка, словарь_метаданных_чанка, оценка_релевантности).
    """
    # 1. Извлечение кандидатов с помощью FAISS (bi-encoder)
    query_emb = bi_encoder.encode([query], convert_to_numpy=True, normalize_embeddings=True)

    # D: расстояния/оценки, I: индексы найденных кандидатов
    D, I = faiss_index.search(query_emb, TOP_K_RETRIEVAL)

    # Получаем найденные чанки и их метаданные по индексам I[0]
    # cands_with_meta = [(chunks_content_list[i], chunks_metadata_list[i]) for i in I[0]]

    # Создаем список извлеченных кандидатов, включая их индексы
    # Это полезно для отладки и для отслеживания исходного положения чанков
    retrieved_candidates = []
    for i in I[0]:  # I[0] содержит индексы из FAISS для первого (единственного) запроса
        # Убедитесь, что индекс 'i' находится в пределах списка
        if 0 <= i < len(chunks_content_list):
            retrieved_candidates.append({
                'content': chunks_content_list[i],
                'metadata': chunks_metadata_list[i],
                'index': i  # Сохраняем исходный индекс в списке
            })
        else:
            print(
                f"Предупреждение: Индекс {i} вне диапазона для списка чанков (размер: {len(chunks_content_list)}). Пропускаем.")

    if not retrieved_candidates:
        return []  # Если кандидатов не найдено, возвращаем пустой список

    # 2. Подготовка пар для кросс-энкодера
    # pairs = [(query, text_content_of_candidate) for candidate in retrieved_candidates]
    pairs = [(query, cand['content']) for cand in retrieved_candidates]

    # 3. Переранжирование с помощью кросс-энкодера
    scores = cross_encoder.predict(pairs)

    # 4. Сопоставление оценок с кандидатами и сортировка
    # ranked_results = sorted(zip(scores, retrieved_candidates), key=lambda x: x[0], reverse=True)
    # Теперь каждый элемент `ranked_results` это (score, {'content':..., 'metadata':..., 'index':...})
    ranked_results = []
    for score, cand_data in zip(scores, retrieved_candidates):
        ranked_results.append((score, cand_data))

    # Сортируем по оценке в убывающем порядке
    ranked_results.sort(key=lambda x: x[0], reverse=True)

    # 5. Возвращаем TOP_K_RERANK лучших результатов
    final_ranked_output = []
    for score, cand_data in ranked_results[:TOP_K_RERANK]:
        final_ranked_output.append(
            (cand_data['content'], cand_data['metadata'], float(score)))  # float(score) для безопасности типа

    return final_ranked_output
