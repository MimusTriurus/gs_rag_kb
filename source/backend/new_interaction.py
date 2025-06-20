import os
import json
from typing import List, Dict
import re
from nltk.stem import PorterStemmer
from settings import *
from ollama import Client

# === Настройки ===
FOLDER_PATH = 'documents_no_meta/'
OLLAMA_MODEL = LLM_MODEL
MAX_TAGS_PER_FILE = 5

ollama_client = Client(
    host=OLLAMA_BASE_URL
)

stemmer = PorterStemmer()

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
    Text:
    -----------------------------------
    {text}
    -----------------------------------
    '''

    response = ollama_client.generate(
        model=model,
        prompt=prompt,
        options={
            'temperature': 0.7,
            'max_tokens': 200
        }
    )
    output: str = response['response'].strip()
    return output

def clean_and_stem_tags(tags: List[str]) -> List[str]:
    cleaned_tags = []
    for tag in tags:
        tag = re.sub(r"[^a-zA-Z0-9]", "_", tag)
        if tag:
            cleaned_tags.append(tag)
    return cleaned_tags

def tag_documents(folder_path: str):
    md_files = load_markdown_files(folder_path)
    file_tags = {}
    file_questions = {}
    for filename, content in md_files.items():
        print(f"Processing: {filename}")

        summarized_text = generate_summarized_text(content)

        #questions = generate_questions(content)
        #file_questions[filename] = questions

        tags = generate_tags(summarized_text)

        file_tags[filename] = {
            "desc": summarized_text,
            "tags": tags
            #"questions": questions.split('\n')
        }
        #print(f"Tags for {filename}: {filtered_tags}")

    with open("file_tags.json", "w", encoding="utf-8") as f:
        json.dump(file_tags, f, ensure_ascii=False, indent=2)

    print("Tagging complete. Tags saved to file_tags.json")

if __name__ == "__main__":
    tag_documents(FOLDER_PATH)
