#!/usr/bin/env python3
"""
Скрипт для разбиения Markdown файлов на чанки на основе заголовков
Поддерживает иерархическое разбиение и дополнительное деление длинных секций
"""

import os
import re
import json
import argparse
from typing import List, Dict, Any
from dataclasses import dataclass


@dataclass
class Chunk:
    """Класс для представления чанка текста"""
    content: str
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            'content': self.content,
            'metadata': self.metadata
        }


class MarkdownSplitter:
    """Класс для разбиения Markdown файлов на чанки"""

    def __init__(self,
                 max_chunk_size: int = 1000,
                 chunk_overlap: int = 200,
                 strip_headers: bool = False):
        """
        Инициализация сплиттера

        Args:
            max_chunk_size: Максимальный размер чанка в символах
            chunk_overlap: Размер перекрытия между чанками
            strip_headers: Удалять ли заголовки из содержимого чанков
        """
        self.max_chunk_size = max_chunk_size
        self.chunk_overlap = chunk_overlap
        self.strip_headers = strip_headers

        # Иерархия заголовков для разбиения
        self.headers_hierarchy = [
            (r'^# (.+)$', 1, 'h1'),
            (r'^## (.+)$', 2, 'h2'),
            (r'^### (.+)$', 3, 'h3'),
            (r'^#### (.+)$', 4, 'h4'),
            (r'^##### (.+)$', 5, 'h5'),
            (r'^###### (.+)$', 6, 'h6')
        ]

    def _parse_headers(self, text: str) -> List[Dict[str, Any]]:
        """Парсинг заголовков в тексте"""
        lines = text.split('\n')
        headers = []

        for i, line in enumerate(lines):
            for pattern, level, tag in self.headers_hierarchy:
                match = re.match(pattern, line.strip())
                if match:
                    headers.append({
                        'line_num': i,
                        'level': level,
                        'title': match.group(1),
                        'tag': tag,
                        'full_line': line
                    })
                    break

        return headers

    def _build_section_path(self, headers_stack: List[Dict[str, Any]]) -> str:
        """Построение пути секции из стека заголовков"""
        return ' -> '.join([h['title'] for h in headers_stack])

    def _split_by_headers(self, text: str) -> List[Chunk]:
        """Основное разбиение по заголовкам"""
        lines = text.split('\n')
        headers = self._parse_headers(text)
        chunks = []

        if not headers:
            # Если заголовков нет, создаем один чанк
            return [Chunk(
                content=text.strip(),
                metadata={
                    'section_path': 'Root',
                    'headers_stack': [],
                    'chunk_type': 'no_headers',
                    'char_count': len(text.strip())
                }
            )]

        # Добавляем виртуальный заголовок в конец для обработки последней секции
        headers.append({
            'line_num': len(lines),
            'level': 0,
            'title': 'END',
            'tag': 'end',
            'full_line': ''
        })

        current_headers_stack = []

        for i in range(len(headers)):
            current_header = headers[i]

            # Определяем начало и конец текущей секции
            start_line = current_header['line_num']
            end_line = headers[i + 1]['line_num'] if i + 1 < len(headers) else len(lines)

            # Обновляем стек заголовков
            # Удаляем заголовки того же или более глубокого уровня
            current_headers_stack = [
                h for h in current_headers_stack
                if h['level'] < current_header['level']
            ]

            if current_header['tag'] != 'end':
                current_headers_stack.append(current_header)

            # Извлекаем содержимое секции
            if current_header['tag'] != 'end':
                section_lines = lines[start_line:end_line]

                if self.strip_headers and section_lines:
                    section_lines = section_lines[1:]  # Удаляем строку заголовка

                section_content = '\n'.join(section_lines).strip()

                if section_content:  # Пропускаем пустые секции
                    chunk = Chunk(
                        content=section_content,
                        metadata={
                            'section_path': self._build_section_path(current_headers_stack),
                            #'headers_stack': [
                            #    {
                            #        'level': h['level'],
                            #        'title': h['title'],
                            #        'tag': h['tag']
                            #    } for h in current_headers_stack
                            #],
                            'current_header': {
                                'level': current_header['level'],
                                'title': current_header['title'],
                                'tag': current_header['tag']
                            },
                            #'chunk_type': 'header_based',
                            #'char_count': len(section_content),
                            #'line_start': start_line,
                            #'line_end': end_line - 1
                        }
                    )
                    chunks.append(chunk)

        return chunks

    def _split_long_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        """Дополнительное разбиение длинных чанков"""
        final_chunks = []

        for chunk in chunks:
            if len(chunk.content) <= self.max_chunk_size:
                final_chunks.append(chunk)
            else:
                # Разбиваем длинный чанк на части
                sub_chunks = self._split_text_recursively(
                    chunk.content,
                    chunk.metadata
                )
                final_chunks.extend(sub_chunks)

        return final_chunks

    def _split_text_recursively(self, text: str, base_metadata: Dict[str, Any]) -> List[Chunk]:
        """Рекурсивное разбиение текста с сохранением структуры"""
        if len(text) <= self.max_chunk_size:
            return [Chunk(content=text, metadata={
                **base_metadata,
                'chunk_type': 'recursive_split',
                'char_count': len(text)
            })]

        # Попытка разбиения по разделителям в порядке приоритета
        separators = ['\n\n', '\n', '. ', '! ', '? ', '; ', ' ']

        for separator in separators:
            if separator in text:
                parts = text.split(separator)
                if len(parts) > 1:
                    chunks = []
                    current_chunk = ""

                    for i, part in enumerate(parts):
                        test_chunk = current_chunk + (separator if current_chunk else "") + part

                        if len(test_chunk) <= self.max_chunk_size:
                            current_chunk = test_chunk
                        else:
                            '''
                            if current_chunk:
                                chunks.append(Chunk(
                                    content=current_chunk,
                                    metadata={
                                        **base_metadata,
                                        'chunk_type': 'recursive_split',
                                        'char_count': len(current_chunk),
                                        'split_by': separator.strip() or 'space'
                                    }
                                ))
                            '''
                            current_chunk = part
                    '''
                    if current_chunk:
                        chunks.append(Chunk(
                            content=current_chunk,
                            metadata={
                                **base_metadata,
                                'chunk_type': 'recursive_split',
                                'char_count': len(current_chunk),
                                'split_by': separator.strip() or 'space'
                            }
                        ))
                    '''
                    return chunks

        # Если не удалось разбить по разделителям, делим по символам
        chunks = []
        start = 0

        while start < len(text):
            end = min(start + self.max_chunk_size, len(text))
            chunk_text = text[start:end]
            '''
            chunks.append(Chunk(
                content=chunk_text,
                metadata={
                    **base_metadata,
                    'chunk_type': 'character_split',
                    'char_count': len(chunk_text),
                    'split_by': 'character'
                }
            ))
            '''
            start = end - self.chunk_overlap if end < len(text) else end

        return chunks

    def split_file(self, file_path: str) -> List[Chunk]:
        """Основной метод разбиения файла"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Первичное разбиение по заголовкам
        header_chunks = self._split_by_headers(content)

        # Дополнительное разбиение длинных чанков
        final_chunks = self._split_long_chunks(header_chunks)

        # Добавляем общие метаданные
        for i, chunk in enumerate(final_chunks):
            chunk.metadata.update({
                'source_file': os.path.basename(file_path),
                #'chunk_id': i,
                #'total_chunks': len(final_chunks)
            })

        return final_chunks


def save_chunks(chunks: List[Chunk], output_path: str, format_type: str = 'json'):
    """Сохранение чанков в файл"""
    chunks_data = [chunk.to_dict() for chunk in chunks]

    if format_type == 'json':
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(chunks_data, f, ensure_ascii=False, indent=2)

    elif format_type == 'txt':
        with open(output_path, 'w', encoding='utf-8') as f:
            for i, chunk in enumerate(chunks):
                f.write(f"=== CHUNK {i + 1} ===\n")
                f.write(f"Section: {chunk.metadata.get('section_path', 'Unknown')}\n")
                f.write(f"Size: {chunk.metadata.get('char_count', 0)} chars\n")
                f.write(f"Type: {chunk.metadata.get('chunk_type', 'unknown')}\n")
                f.write("-" * 50 + "\n")
                f.write(chunk.content)
                f.write("\n" + "=" * 70 + "\n\n")


def main():
    parser = argparse.ArgumentParser(description='Split Markdown file into chunks based on headers')
    parser.add_argument('input_file', help='Path to input Markdown file')
    parser.add_argument('-o', '--output', help='Output file path')
    parser.add_argument('-f', '--format', choices=['json', 'txt'], default='json',
                        help='Output format (default: json)')
    parser.add_argument('-s', '--chunk-size', type=int, default=1000,
                        help='Maximum chunk size in characters (default: 1000)')
    parser.add_argument('--overlap', type=int, default=200,
                        help='Chunk overlap size (default: 200)')
    parser.add_argument('--strip-headers', action='store_true',
                        help='Remove headers from chunk content')
    parser.add_argument('--stats', action='store_true',
                        help='Print splitting statistics')

    args = parser.parse_args()

    # Проверка существования входного файла
    if not os.path.exists(args.input_file):
        print(f"Ошибка: файл {args.input_file} не найден")
        return

    # Создание сплиттера
    splitter = MarkdownSplitter(
        max_chunk_size=args.chunk_size,
        chunk_overlap=args.overlap,
        strip_headers=args.strip_headers
    )

    # Разбиение файла
    print(f"Разбиение файла: {args.input_file}")
    chunks = splitter.split_file(args.input_file)
    print(f"Создано чанков: {len(chunks)}")

    # Статистика
    if args.stats:
        total_chars = sum(chunk.metadata.get('char_count', 0) for chunk in chunks)
        avg_size = total_chars / len(chunks) if chunks else 0
        chunk_types = {}

        for chunk in chunks:
            chunk_type = chunk.metadata.get('chunk_type', 'unknown')
            chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1

        print("\n=== СТАТИСТИКА ===")
        print(f"Общее количество символов: {total_chars}")
        print(f"Средний размер чанка: {avg_size:.0f} символов")
        print("Типы чанков:")
        for chunk_type, count in chunk_types.items():
            print(f"  {chunk_type}: {count}")

    # Сохранение результата
    if args.output:
        output_path = args.output
    else:
        base_name = os.path.splitext(args.input_file)[0]
        extension = 'json' if args.format == 'json' else 'txt'
        output_path = f"{base_name}_chunks.{extension}"

    save_chunks(chunks, output_path, args.format)
    print(f"Результат сохранен в: {output_path}")


if __name__ == "__main__":
    main()