from typing import List, Dict, Set, Optional
from dataclasses import dataclass
from datetime import datetime
import hashlib

from input_data_processing.chunks_merger import merge_advanced


@dataclass
class ContextPart:
    """Представляет часть контекста с метаданными"""
    content: str
    source: Optional[str] = None
    score: Optional[float] = None
    metadata: Optional[Dict] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    def get_hash(self) -> str:
        """Генерирует хеш для идентификации уникальности контекста"""
        content_hash = hashlib.md5(self.content.encode('utf-8')).hexdigest()
        return content_hash

    def to_dict(self) -> Dict:
        return {
            'content': self.content,
            'source': self.source,
            'score': self.score,
            'metadata': self.metadata,
            'hash': self.get_hash()
        }


@dataclass
class HistoryEntry:
    """Запись в истории вопросов и ответов"""
    question: str
    answer: str
    context_parts: List[ContextPart]
    timestamp: datetime
    session_id: str

    def to_dict(self) -> Dict:
        return {
            'question': self.question,
            'answer': self.answer,
            'context_parts': [cp.to_dict() for cp in self.context_parts],
            'timestamp': self.timestamp.isoformat(),
            'session_id': self.session_id
        }


class RAGHistoryManager:
    """
    Менеджер истории для RAG-системы с поддержкой сессий
    и предотвращением дублирования контекста
    """

    def __init__(self, max_history_per_session: int = 3):
        self.max_history_per_session = max_history_per_session
        # Хранилище истории по сессиям
        self.sessions_history: Dict[str, List[HistoryEntry]] = {}
        # Хеши уникальных контекстов по сессиям
        self.session_context_hashes: Dict[str, Set[str]] = {}

    def add_entry(self,
                  question: str,
                  best_answer: str,
                  context_parts: List[ContextPart],
                  session_id: str) -> None:
        """
        Добавляет новую запись в историю сессии

        Args:
            question: Заданный вопрос
            best_answer: Найденный лучший ответ
            context_parts: Части контекста, использованные для ответа
            session_id: Идентификатор сессии
        """
        # Инициализация сессии если она новая
        if session_id not in self.sessions_history:
            self.sessions_history[session_id] = []
            self.session_context_hashes[session_id] = set()

        # Фильтрация уникальных контекстов
        unique_context_parts = self._filter_unique_contexts(context_parts, session_id)

        # Создание записи истории
        entry = HistoryEntry(
            question=question,
            answer=best_answer,
            context_parts=unique_context_parts,
            timestamp=datetime.now(),
            session_id=session_id
        )

        # Добавление в историю сессии
        self.sessions_history[session_id].append(entry)

        # Добавление хешей новых контекстов
        for context in unique_context_parts:
            self.session_context_hashes[session_id].add(context.get_hash())

        # Ограничение размера истории
        self._trim_session_history(session_id)

    def _filter_unique_contexts(self,
                                context_parts: List[ContextPart],
                                session_id: str) -> List[ContextPart]:
        """Фильтрует контексты, оставляя только уникальные для сессии"""
        unique_contexts = []
        session_hashes = self.session_context_hashes.get(session_id, set())

        for context in context_parts:
            context_hash = context.get_hash()
            if context_hash not in session_hashes:
                unique_contexts.append(context)

        return unique_contexts

    def _trim_session_history(self, session_id: str) -> None:
        """Обрезает историю сессии до максимального размера"""
        if len(self.sessions_history[session_id]) > self.max_history_per_session:
            # Удаляем самые старые записи
            removed_entries = self.sessions_history[session_id][:-self.max_history_per_session]
            self.sessions_history[session_id] = self.sessions_history[session_id][-self.max_history_per_session:]

            # Пересчитываем хеши контекстов для оставшихся записей
            self._recalculate_context_hashes(session_id)

    def _recalculate_context_hashes(self, session_id: str) -> None:
        """Пересчитывает хеши контекстов для сессии"""
        self.session_context_hashes[session_id] = set()
        for entry in self.sessions_history[session_id]:
            for context in entry.context_parts:
                self.session_context_hashes[session_id].add(context.get_hash())

    def get_session_history(self, session_id: str) -> List[HistoryEntry]:
        """Возвращает историю для конкретной сессии"""
        return self.sessions_history.get(session_id, [])

    def get_recent_contexts(self,
                            session_id: str,
                            limit: int = 3) -> List[ContextPart]:
        """
        Возвращает последние уникальные контексты из истории сессии

        Args:
            session_id: Идентификатор сессии
            limit: Максимальное количество контекстов

        Returns:
            Список последних уникальных контекстов
        """
        history = self.get_session_history(session_id)
        if not history:
            return []

        # Собираем контексты из последних записей
        recent_contexts = []
        seen_hashes = set()

        # Идем по истории в обратном порядке (от новых к старым)
        for entry in reversed(history):
            for context in entry.context_parts:
                context_hash = context.get_hash()
                if context_hash not in seen_hashes and len(recent_contexts) < limit:
                    recent_contexts.append(context)
                    seen_hashes.add(context_hash)
                if len(recent_contexts) >= limit:
                    break
            if len(recent_contexts) >= limit:
                break

        return recent_contexts


    def get_formatted_history(
            self,
            session_id: str,
            new_context: List[str],
            limit: int = 3
    ) -> str:
        history = self.get_session_history(session_id)
        # Собираем контексты из последних записей
        recent_contexts = []
        seen_hashes = set()
        questions = ''
        # Идем по истории в обратном порядке (от новых к старым)
        for entry in reversed(history):
            questions += f'question: {entry.question} answer: {entry.answer}\n'
            for context in entry.context_parts:
                context_hash = context.get_hash()
                if context_hash not in seen_hashes and len(recent_contexts) < limit:
                    recent_contexts.append(context.content)
                    seen_hashes.add(context_hash)
                if len(recent_contexts) >= limit:
                    break
            if len(recent_contexts) >= limit:
                break

        if new_context:
            recent_contexts = recent_contexts + new_context
        recent_contexts = set(recent_contexts)
        merged_result = merge_advanced(list(recent_contexts))

        result: str = f'''
        ----\ncontext:\n{merged_result}\n----
        '''

        result += f'\n{questions}'
        return result.strip()


    def answer_question_with_history(self,
                                     question: str,
                                     new_context_parts: List[ContextPart],
                                     session_id: str,
                                     history_context_limit: int = 3) -> Dict:
        """
        Отвечает на вопрос используя новый контекст и историю сессии

        Args:
            question: Вопрос пользователя
            new_context_parts: Новые части контекста из поиска
            session_id: Идентификатор сессии
            history_context_limit: Лимит контекстов из истории

        Returns:
            Словарь с ответом и использованными контекстами
        """
        # Получаем исторический контекст
        historical_contexts = self.get_recent_contexts(session_id, history_context_limit)

        # Фильтруем новые контексты от дублирования с историей
        unique_new_contexts = self._filter_unique_contexts(new_context_parts, session_id)

        # Объединяем контексты (приоритет новым)
        all_contexts = unique_new_contexts + historical_contexts

        # Генерируем ответ (здесь упрощенная логика)
        answer = self._generate_answer(question, all_contexts)

        # Сохраняем в историю
        self.add_entry(question, answer, unique_new_contexts, session_id)

        return {
            'question': question,
            'answer': answer,
            'new_contexts_used': len(unique_new_contexts),
            'historical_contexts_used': len(historical_contexts),
            'total_contexts': len(all_contexts),
            'contexts': [ctx.to_dict() for ctx in all_contexts]
        }

    def clear_session(self, session_id: str) -> None:
        """Очищает историю конкретной сессии"""
        if session_id in self.sessions_history:
            del self.sessions_history[session_id]
        if session_id in self.session_context_hashes:
            del self.session_context_hashes[session_id]


# Пример использования
if __name__ == "__main__":
    # Создаем менеджер истории
    history_manager = RAGHistoryManager(max_history_per_session=5)

    # Создаем тестовые контексты
    context1 = ContextPart("Python - это высокоуровневый язык программирования", "doc1.txt", 0.9)
    context2 = ContextPart("Python поддерживает ООП и функциональное программирование", "doc2.txt", 0.8)
    context3 = ContextPart("Python имеет большую экосистему библиотек", "doc3.txt", 0.7)

    context4 = ContextPart("Python имеет большую экосистему библиотек", "doc4.txt", 0.8)

    session_id = "user_123"

    history_manager.add_entry('Что такое Python?', 'Это язык программирования.', [context1, context2, context3], '1')
    history_manager.add_entry('Какие особенности у Python?', 'Он относится к интерпритируемым языкам программирования.', [context1, context2, context4], '1')
    #r = history_manager.get_session_history('1')
    formatted_history = history_manager.get_formatted_history('1')
    print(formatted_history)
