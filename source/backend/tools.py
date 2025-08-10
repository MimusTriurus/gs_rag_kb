from pathlib import Path
from typing import List
import re

from source.backend.llm_settings import Settings
from source.backend.llm_tools.llm_questions_generator import LlmQuestionsGenerator
from source.backend.settings import MISSING_INFO_TEXT

def clean_html_to_one_line(text: str) -> str:
    text_no_tags = re.sub(r'<[^>]+>', '', text)
    text_flat = re.sub(r'\s+', ' ', text_no_tags)
    return text_flat.strip()

def is_llm_answer_valid(answer: str) -> bool:
    cleaned_answer = clean_html_to_one_line(answer)
    return MISSING_INFO_TEXT.lower() in cleaned_answer.lower() or 'is not provided' in cleaned_answer.lower()


def make_answer_about_not_found_data_in_context(meta_infos: List[dict], files_context: dict, settings: Settings) -> str:
    possible_questions = ''

    if settings.GENERATE_QUESTIONS():
        questions_generator = LlmQuestionsGenerator(settings.LLM_MODEL())
        for meta in meta_infos:
            file_path: Path = meta['source_file']
            file_name = file_path.name
            if file_name:
                context = files_context.get(file_name, '')
                if context:
                    questions_data = questions_generator.generate(context)
                    if questions_data:
                        possible_questions = '<strong>Possible questions within the context:</strong><ul>'
                        for q in questions_data.get('questions', []):
                            question = q.get('question')
                            possible_questions += f'<li>{question}</li>'
                        possible_questions += '</ul>'
                    break

    content = ''
    for meta in meta_infos:
        content += f'<li><a href="{meta["url"]}">{meta.get("title", "Link")}</a> <br><br><strong>{meta["author"]}</strong></li><br>'
        break
    result = f'''
    <br>The requested information is not found in the context provided.<br>
    <br>Try searching for information on the <strong>following pages</strong> or ask the <strong>following people</strong> for help:<br>
    <ul>
    {content}
    </ul>
    '''
    if possible_questions:
        result = f'''
        {result}
        {possible_questions}
        '''
    return result