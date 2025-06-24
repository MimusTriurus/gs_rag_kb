import os
import json
from typing import List, Dict
import re
from settings import *
from ollama import Client
import frontmatter

# === Настройки ===
FOLDER_PATH = 'documents_no_meta/'
OLLAMA_MODEL = LLM_MODEL
MAX_TAGS_PER_FILE = 5

ollama_client = Client(
    host=OLLAMA_BASE_URL
)


def load_markdown_files(folder_path: str) -> Dict[str, str]:
    md_files = {}
    for filename in os.listdir(folder_path):
        if filename.endswith(".md"):
            with open(os.path.join(folder_path, filename), "r", encoding="utf-8") as f:
                md_files[filename] = f.read()
    return md_files


def generate_tags(text: str, model: str = OLLAMA_MODEL) -> List[str]:
    prompt = (
        "You are an assistant that assigns relevant search tags to a list of questions. "
        "Your task: read the user's questions and return a list of several keywords (tags) without any description, "
        "only plain words separated by commas.\n"
        f"Document text: ```{text}```\n"
        "Response: a list of tags separated by commas."
    )

    response = ollama_client.generate(
        model=model,
        prompt=prompt,
        options={
            'temperature': 0.7,
        }
    )

    output = response['response'].strip()

    tags = [tag.strip().lower() for tag in output.split(",") if tag.strip()]
    return tags[:MAX_TAGS_PER_FILE]


def generate_questions(text: str, model: str = OLLAMA_MODEL) -> str:
    prompt = f'''
    Imagine you are an inquisitive user who has just finished reading the document below.\n
    Your goal is to better understand its content, uncover any ambiguities, and clarify key details.\n
    Please generate a list of general questions that you, as a thoughtful reader, might ask to gain a deeper understanding of the document.\n 
    Your questions should address aspects such as the objectives of the document, 
    underlying assumptions, the context behind certain statements, possible implications, 
    and any areas that seem unclear or require further explanation.\n
    Provide 5 shortest and probing questions.\n
    Document text: ```{text}```\n
    Response: a list of questions separated by commas.
    '''

    response = ollama_client.generate(
        model=model,
        prompt=prompt,
        options={
            'temperature': 0.7,
        }
    )
    output: str = response['response'].strip()
    return output.replace('\n\n', '\n')


def generate_summarized_text(text: str, model: str = OLLAMA_MODEL) -> str:
    prompt = f'''
    Analyze the text and shortly describe what it is about.
    Aim for a compact version that reflects the overall meaning and essence of the text.
    Maximum 1 sentences. As short as possible!
    Text:
    -----------------------------------
    {text}
    -----------------------------------
    '''

    response = ollama_client.generate(
        model=model,
        prompt=prompt,
        options={
            'temperature': 0.9,
            'max_tokens': 15
        }
    )
    output: str = response['response'].strip()
    output = output.replace('\n', '')
    return output


def clean_and_stem_tags(tags: List[str]) -> List[str]:
    cleaned_tags = []
    for tag in tags:
        tag = re.sub(r"[^a-zA-Z0-9]", "_", tag)
        if tag:
            cleaned_tags.append(tag)
    return cleaned_tags


def extract_headings(md_content, levels=(1, 2, 3)):
    pattern = re.compile(r"^(#{1,3})\s+(.*)$", re.MULTILINE)
    headings = set()
    for match in pattern.finditer(md_content):
        level = len(match.group(1))
        if level in levels:
            header = match.group(2).strip()
            cleaned_header = re.sub(r'^[\d\s\W_]+', '', header)
            headings.add(cleaned_header)
    return list(headings)


def make_doc_with_meta(content: str, filename: str, title: str, url: str, author: str):
    headings = extract_headings(content)
    summarized_text = generate_summarized_text(content)

    new_post = frontmatter.Post(
        content,
        title=title,
        summarized_text=summarized_text,
        tags=headings,
        url=url,
        author=author
    )

    with open(f'{DOCUMENTS_PATH}/{filename}', 'w', encoding='utf-8') as f:
        f.write(frontmatter.dumps(new_post))


def tag_documents(folder_path: str):
    md_files = load_markdown_files(folder_path)

    for filename, content in md_files.items():
        print(f"Processing: {filename}")
        title = filename
        source_url = f'https://{filename}.com'
        author = f'user_{filename}'
        make_doc_with_meta(content, filename, title, source_url, author)


if __name__ == "__main__":
    tag_documents(TEST_DOCUMENTS_PATH)
