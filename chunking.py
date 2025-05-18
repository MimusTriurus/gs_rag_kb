import re

from settings import clean_chunk_markdown


def split_md_into_blocks(md_content):
    blocks = []
    current_block = []
    in_code_block = False

    for line in md_content.split('\n'):
        if line.strip().startswith('```'):
            in_code_block = not in_code_block
        if not in_code_block and line.strip() == '':
            if current_block:
                blocks.append('\n'.join(current_block))
                current_block = []
        else:
            current_block.append(line)
    if current_block:
        blocks.append('\n'.join(current_block))
    return blocks


def split_large_block(block, max_size):
    if len(block) <= max_size:
        return [block]
    sub_blocks = []
    current_sub = []
    current_length = 0
    for line in block.split('\n'):
        line_length = len(line) + 1
        if current_length + line_length > max_size:
            if current_sub:
                sub_blocks.append('\n'.join(current_sub))
                current_sub = []
                current_length = 0
        current_sub.append(line)
        current_length += line_length
    if current_sub:
        sub_blocks.append('\n'.join(current_sub))
    return sub_blocks


def process_blocks(blocks, max_block_size=1500):
    processed = []
    for block in blocks:
        if len(block) > max_block_size:
            processed.extend(split_large_block(block, max_block_size))
        else:
            processed.append(block)
    return processed


def create_chunks(processed_blocks, max_chunk_size=1500, overlap=2):
    chunks = []
    current_chunk = []
    current_length = 0

    for i, block in enumerate(processed_blocks):
        block_len = len(block)
        if current_length + block_len <= max_chunk_size:
            current_chunk.append(block)
            current_length += block_len
        else:
            chunks.append('\n\n'.join(current_chunk))
            overlap_blocks = current_chunk[-overlap:] if len(current_chunk) >= overlap else []
            current_chunk = overlap_blocks.copy()
            current_length = sum(len(b) for b in current_chunk) + 2 * (len(current_chunk) - 1)
            current_chunk.append(block)
            current_length += block_len + 2

            while current_length > max_chunk_size:
                removed = current_chunk.pop(0)
                current_length -= len(removed) + 2
                if not current_chunk:
                    current_chunk.append(block)
                    current_length = block_len

    if current_chunk:
        chunks.append('\n\n'.join(current_chunk))
    return chunks


def clean_md_content(chunk):
    chunk = re.sub(r'```.*?```', '', chunk, flags=re.DOTALL)
    chunk = re.sub(r'!\[.*?\]\(.*?\)', '', chunk)
    chunk = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', chunk)
    chunk = re.sub(r'`(.*?)`', r'\1', chunk)
    chunk = re.sub(r'\*\*(.*?)\*\*', r'\1', chunk)
    chunk = re.sub(r'__(.*?)__', r'\1', chunk)
    chunk = re.sub(r'\*(.*?)\*', r'\1', chunk)
    chunk = re.sub(r'_(.*?)_', r'\1', chunk)
    chunk = re.sub(r'^#+\s*', '', chunk, flags=re.MULTILINE)
    chunk = re.sub(r'^[\*\-+]\s+', '', chunk, flags=re.MULTILINE)
    chunk = re.sub(r'^\d+\.\s+', '', chunk, flags=re.MULTILINE)
    chunk = re.sub(r'^>\s*', '', chunk, flags=re.MULTILINE)
    chunk = re.sub(r'^---+$', '', chunk, flags=re.MULTILINE)
    chunk = re.sub(r'<.*?>', '', chunk)
    chunk = re.sub(r'\n{3,}', '\n\n', chunk)
    return chunk.strip()


def split_md_file(file_path, max_chunk_size=1500, overlap=2, clean_markdown=True):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    blocks = split_md_into_blocks(content)
    processed_blocks = process_blocks(blocks, max_chunk_size)
    chunks = create_chunks(processed_blocks, max_chunk_size, overlap)

    if clean_markdown:
        chunks = [clean_md_content(chunk) for chunk in chunks]

    return [chunk for chunk in chunks if len(chunk) <= max_chunk_size]
