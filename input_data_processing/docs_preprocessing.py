import os
import json
from typing import List, Dict
import re
from ollama import Client
import frontmatter

from source.backend.settings import *

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

    prompt = f'''
    You are an expert QA engineer creating question-answer pairs for a Retrieval-Augmented Generation (RAG) system. Your task is to generate concise, context-specific questions based exclusively on provided TestRail test case documentation. Follow these guidelines:

1. **Question Requirements**:
- Generate 15-20 questions per document
- Questions must be answerable solely using the test case content
- Focus on key elements: preconditions, test steps, expected results, and verification points
- Use exact terminology from the documents (e.g., "progression screen", "FEP hangar")
- Prioritize functional verification over theoretical concepts

2. **Question Structure**:
- Start with "What", "How", "When", "Where", or "Verify"
- Maximum 12 words per question
- Include specific screen names and UI elements
- Reference exact test step numbers where applicable

3. **Forbidden Content**:
- No hypothetical scenarios
- No questions requiring external knowledge
- No generalization (e.g., avoid "generally", "usually")

4. **Output Format**:
- Plain numbered list only
- No explanations or additional text

**Example Questions (based on sample test case)**:
1. What preconditions are needed for Progression Screen testing?
2. How to verify first entrance animation behavior?
3. What resolutions require validation in preconditions?
4. How does progression screen behave when reopened?
5. Where are FEP configuration files located?
6. Verify animation behavior after interrupted first entrance
7. What components comprise progression screen UI?
8. How to navigate to tier lists screen?
9. When does timer appear in progression screen?
10. Verify mouse wheel scrolling functionality

**Document Content**:
{text}

Generate questions now:
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


def generate_questions_and_save():
    md_files = load_markdown_files(DOCUMENTS_PATH)

    for filename, content in md_files.items():
        print(f"Generate questions for: {filename}")
        questions = generate_questions(content)
        with open(f"questions_{filename}.txt", "a", encoding="utf-8") as file:
            file.write(questions + "\n")


def extract_headers_as_tags():
    for filename in os.listdir(DOCUMENTS_PATH):
        if filename.endswith(".md"):
            unique_tags = set()
            with open(os.path.join(DOCUMENTS_PATH, filename), 'r', encoding='utf-8') as f:
                for line in f:
                    # Ищем строки с заголовком # Test Case:
                    match = re.match(r'^#\s*Test Case:\s*(.*)', line)
                    if match:
                        path_str = match.group(1)
                        # Разбиваем по ':' и '>'
                        parts = re.split(r'[:>]', path_str)
                        # Очищаем пробелы и фильтруем пустые
                        tags = [tag.strip() for tag in parts if tag.strip()]
                        unique_tags.update(tags)

            # Записываем уникальные теги в файл, каждый на отдельной строке
            with open(f'tags_{filename}.txt', 'w', encoding='utf-8') as out_f:
                for tag in sorted(unique_tags):
                    out_f.write('- ' + tag + '\n')


if __name__ == "__main__":
    extract_headers_as_tags()
    # generate_questions_and_save()
