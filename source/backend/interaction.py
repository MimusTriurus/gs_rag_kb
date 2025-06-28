from ollama import Client
import json
from source.backend.settings import LLM_MODEL, MISSING_INFO_TEXT, OLLAMA_BASE_URL
from typing import List, Tuple, Dict
import logging
import re

ollama_client = Client(
    host=OLLAMA_BASE_URL
)

system_prompt_with_history = f"""
You are an information extraction assistant for a RAG system. 
Your role is to answer questions using ONLY the provided context.
OUTPUT FORMAT - HTML:
Structure your response as clean HTML with these elements:
1. Any text that represents headings should be wrapped in the appropriate <h1>–<h6> tags. If no heading level is specified, default to <h2>.
2. Each paragraph of text should be enclosed within a <p> tag.
3. If you encounter lists (numbered or bulleted), convert them into the appropriate <ol> (ordered list) or <ul> (unordered list) containing <li> elements.
4. The output should contain only HTML code without any additional explanations or comments.
CORE RULES:
1. Use STRICTLY ONLY information from the PROVIDED CONTEXT or information from the CHAT HISTORY.
2. Do not add knowledge from your training data
3. Be precise and factual
4. If information is not in context, say STRICTLY ONLY '{MISSING_INFO_TEXT}'
PROVIDED CONTEXT:
"""

system_prompt = f"""
You are an information extraction assistant for a RAG system. 
Your role is to answer questions using ONLY the provided context.
OUTPUT FORMAT - HTML:
Structure your response as clean HTML with these elements:
1. Any text that represents headings should be wrapped in the appropriate <h1>–<h6> tags. If no heading level is specified, default to <h2>.
2. Each paragraph of text should be enclosed within a <p> tag.
3. If you encounter lists (numbered or bulleted), convert them into the appropriate <ol> (ordered list) or <ul> (unordered list) containing <li> elements.
4. The output should contain only HTML code without any additional explanations or comments.
CORE RULES:
1. Use STRICTLY ONLY information from the PROVIDED CONTEXT.
2. Do not add knowledge from your training data
3. Be precise and factual
4. If information is not in context, say STRICTLY ONLY '{MISSING_INFO_TEXT}'
PROVIDED CONTEXT:
"""


def make_system_prompt_with_context(prompt: str, context: str):
    result = f"""
    {prompt}
    {context}
    """
    return result


def clean_html_to_one_line(text: str) -> str:
    text_no_tags = re.sub(r'<[^>]+>', '', text)
    text_flat = re.sub(r'\s+', ' ', text_no_tags)
    return text_flat.strip()


class OllamaChatSession:
    def __init__(self, model: str, sp: str, max_history: int = 2):
        self.model = model
        self.system_prompt = sp
        self.max_history = max_history
        self.messages: List[Dict[str, str]] = []

    def _prune_history(self):
        while len(self.messages) > 1 + self.max_history * 2:
            for _ in range(2):
                if len(self.messages) > 1:
                    self.messages.pop(1)

    def ask(self, context: str, query: str) -> str:
        user_message = f"""
        QUESTION:
        {query}
        """
        current_messages = [
            {"role": "system", "content": make_system_prompt_with_context(self.system_prompt, context)},
        ]
        current_messages.extend(self.messages)
        current_messages.append({"role": "user", "content": user_message.strip()})
        max_tokens = 4096
        response = ollama_client.chat(
            model=self.model,
            messages=current_messages,
            options={
                'temperature': 0.3,
                'max_tokens': max_tokens,
                "top_k": 20,
                "top_p": 0.8
            }
        )

        answer: str = response['message']['content'].strip()
        answer = answer.replace('```html', '').replace('```', '')

        cleaned_answer = clean_html_to_one_line(answer)
        if MISSING_INFO_TEXT.lower() in cleaned_answer.lower():
            return MISSING_INFO_TEXT

        return answer

    def update_history(self, question: str, answer: str):
        self._prune_history()
        self.messages.append({"role": "user",       "content": f'QUESTION: {clean_html_to_one_line(question)}'})
        self.messages.append({"role": "assistant",  "content": f'ANSWER: {clean_html_to_one_line(answer)}'})


session = OllamaChatSession(LLM_MODEL, system_prompt)


def answer_question(context: str, query: str, model: str = LLM_MODEL) -> str:
    return session.ask(context, query)
