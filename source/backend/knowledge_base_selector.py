from typing import List, Dict

from source.backend.new_interaction import OLLAMA_MODEL, ollama_client
from source.backend.settings import TOP_K_FILE_SELECT


def select_best_files_using_ollama(
        query: str,
        file_paths: List[str],
        file_meta: Dict[str, Dict[str, str]],
        model: str = OLLAMA_MODEL,
        top_k: int = TOP_K_FILE_SELECT
) -> List[str]:
    """
    Выбирает наиболее релевантные файлы для запроса, используя Ollama LLM для оценки соответствия.

    Args:
        query: Пользовательский запрос
        file_paths: Список путей к файлам
        file_meta: Метаданные файлов {путь: {ключ: значение}}
        model: Модель Ollama
        top_k: Количество файлов для возврата

    Returns:
        Список путей к наиболее релевантным файлам
    """
    file_scores = []

    for file_path in file_paths:
        metadata = file_meta.get(file_path, {})
        score = _evaluate_file_relevance_with_ollama(query, file_path, metadata, model)
        file_scores.append((file_path, score))
        print(f"File: {file_path}, Score: {score}")

    # Сортируем по убыванию score и берем top_k
    file_scores.sort(key=lambda x: x[1], reverse=True)
    selected_files = [file_path for file_path, _ in file_scores[:top_k]]

    print(f"Selected {len(selected_files)} files: {selected_files}")
    return selected_files


def _evaluate_file_relevance_with_ollama(
        query: str,
        file_path: str,
        metadata: Dict[str, str],
        model: str
) -> float:
    """
    Оценивает релевантность файла для запроса с помощью Ollama.

    Returns:
        Числовая оценка релевантности от 0.0 до 1.0
    """
    # Формируем описание файла из метаданных
    file_description = _format_file_metadata(file_path, metadata)

    prompt = f'''Rate relevance to "{query}" (0.0-1.0): {metadata['title']}'''

    try:
        response = ollama_client.generate(
            model=model,
            prompt=prompt,
            options={
                'temperature': 0.1,  # Низкая температура для более консистентных оценок
                'top_p': 0.9,
                'num_predict': 10,  # Ограничиваем длину ответа
            }
        )

        # Извлекаем числовую оценку из ответа
        output = response['response'].strip()
        score = _extract_score_from_response(output)
        return score

    except Exception as e:
        print(f"Error evaluating file {file_path}: {e}")
        return 0.0  # Возвращаем минимальную оценку в случае ошибки


def _format_file_metadata(file_path: str, metadata: Dict[str, str]) -> str:
    """Форматирует метаданные файла в читаемый вид."""
    file_name = file_path.split('/')[-1]  # Извлекаем имя файла

    description_parts = [f"Filename: {file_name}"]

    # Добавляем метаданные
    for key, value in metadata.items():
        if value:  # Пропускаем пустые значения
            description_parts.append(f"{key.title()}: {value}")

    return '\n'.join(description_parts)


def _extract_score_from_response(response: str) -> float:
    """Извлекает числовую оценку из ответа LLM."""
    import re

    # Ищем числа от 0.0 до 1.0 в тексте
    score_pattern = r'\b(0\.\d+|1\.0|0\.0)\b'
    matches = re.findall(score_pattern, response)

    if matches:
        try:
            score = float(matches[0])
            # Убеждаемся, что оценка в допустимом диапазоне
            return max(0.0, min(1.0, score))
        except ValueError:
            pass

    # Если не удалось извлечь оценку, пытаемся найти любое число
    number_pattern = r'\b\d+\.?\d*\b'
    numbers = re.findall(number_pattern, response)

    if numbers:
        try:
            score = float(numbers[0])
            # Нормализуем к диапазону [0, 1]
            if score > 1.0:
                score = score / 10.0 if score <= 10.0 else 1.0
            return max(0.0, min(1.0, score))
        except ValueError:
            pass

    print(f"Could not extract score from response: {response}")
    return 0.5  # Возвращаем средную оценку по умолчанию


# Альтернативная версия с batch-обработкой для оптимизации
def select_best_files_using_ollama_batch(
        query: str,
        file_paths: List[str],
        file_meta: Dict[str, Dict[str, str]],
        model: str = OLLAMA_MODEL,
        top_k: int = TOP_K_FILE_SELECT,
        batch_size: int = 5
) -> List[str]:
    """
    Батчевая версия для обработки нескольких файлов одновременно.
    Может быть более эффективной для большого количества файлов.
    """
    all_scores = []

    # Обрабатываем файлы батчами
    for i in range(0, len(file_paths), batch_size):
        batch_paths = file_paths[i:i + batch_size]
        batch_scores = _evaluate_files_batch(query, batch_paths, file_meta, model)
        all_scores.extend(zip(batch_paths, batch_scores))

    # Сортируем и выбираем top_k
    all_scores.sort(key=lambda x: x[1], reverse=True)
    selected_files = [file_path for file_path, _ in all_scores[:top_k]]

    print(f"Selected {len(selected_files)} files from batch processing: {selected_files}")
    return selected_files


def _evaluate_files_batch(
        query: str,
        file_paths: List[str],
        file_meta: Dict[str, Dict[str, str]],
        model: str
) -> List[float]:
    """Оценивает релевантность нескольких файлов одним запросом к LLM."""

    files_info = []
    for i, file_path in enumerate(file_paths, 1):
        metadata = file_meta.get(file_path, {})
        file_description = _format_file_metadata(file_path, metadata)
        files_info.append(f"File {i}:\n{file_description}")

    files_text = '\n\n'.join(files_info)

    prompt = f'''
    Analyze how relevant each of the following files is to the user query.

    User Query: "{query}"

    Files to evaluate:
    {files_text}

    Task: Rate the relevance of each file to the user query on a scale from 0.0 to 1.0.

    Respond with ONLY the scores separated by commas (e.g., 0.7, 0.3, 0.9, 0.1, 0.5).
    The order must match the file order above.
    '''

    try:
        response = ollama_client.generate(
            model=model,
            prompt=prompt,
            options={
                'temperature': 0.1,
                'top_p': 0.9,
            }
        )

        scores_text = response['response'].strip()
        scores = _parse_batch_scores(scores_text, len(file_paths))
        return scores

    except Exception as e:
        print(f"Error in batch evaluation: {e}")
        return [0.5] * len(file_paths)  # Возвращаем средние оценки


def _parse_batch_scores(scores_text: str, expected_count: int) -> List[float]:
    """Парсит оценки из батчевого ответа."""
    import re

    # Ищем числа в тексте
    score_pattern = r'\b\d+\.?\d*\b'
    matches = re.findall(score_pattern, scores_text)

    scores = []
    for match in matches[:expected_count]:
        try:
            score = float(match)
            # Нормализуем к [0, 1]
            if score > 1.0:
                score = score / 10.0 if score <= 10.0 else 1.0
            scores.append(max(0.0, min(1.0, score)))
        except ValueError:
            scores.append(0.5)

    # Дополняем до нужного количества, если не хватает
    while len(scores) < expected_count:
        scores.append(0.5)

    return scores[:expected_count]