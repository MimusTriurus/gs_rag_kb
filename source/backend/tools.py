from typing import List
import re
from source.backend.settings import MISSING_INFO_TEXT

def clean_html_to_one_line(text: str) -> str:
    text_no_tags = re.sub(r'<[^>]+>', '', text)
    text_flat = re.sub(r'\s+', ' ', text_no_tags)
    return text_flat.strip()

def is_llm_answer_valid(answer: str) -> bool:
    cleaned_answer = clean_html_to_one_line(answer)
    return MISSING_INFO_TEXT.lower() in cleaned_answer.lower() or 'is not provided' in cleaned_answer.lower()


def make_answer_about_not_found_data_in_context(meta_infos: List[dict]) -> str:
    content = ''
    for meta in meta_infos:
        content += f'<li><a href="{meta["url"]}">{meta["title"]}</a> <br><br><strong>{meta["author"]}</strong></li><br>'
    return f'''
    <br>The requested information is not found in the context provided.<br>
    <br>Try searching for information on the following pages or ask the following people for help:<br><br>
    <ul>
    {content}
    </ul>
    '''