from ollama import Client
import json
from source.backend.settings import LLM_MODEL, MISSING_INFO_TEXT, OLLAMA_BASE_URL
from typing import List, Tuple, Dict
import logging
import re

ollama_client = Client(
    host=OLLAMA_BASE_URL
)

system_prompt = """
You are an information extraction assistant for a RAG system. Your role is to answer questions using ONLY the provided context.
CORE RULES:
1. Use ONLY information from the provided context
2. Do not add knowledge from your training data
3. Be precise and factual
4. If information is not in context, say STRICTLY ONLY '{MISSING_INFO_TEXT}'
OUTPUT FORMAT - HTML:
Structure your response as clean HTML with these elements:
1. Any text that represents headings should be wrapped in the appropriate <h1>–<h6> tags. If no heading level is specified, default to <h2>.
2. Each paragraph of text should be enclosed within a <p> tag.
3. If you encounter lists (numbered or bulleted), convert them into the appropriate <ol> (ordered list) or <ul> (unordered list) containing <li> elements.
4. The output should contain only HTML code without any additional explanations or comments.
"""


def refine_user_prompt(user_query: str, model: str = LLM_MODEL) -> str:
    prompt = f"""
    You are an intelligent assistant whose task is to refine user queries for a Retrieval-Augmented Generation (RAG) system.
    The RAG system searches through technical documentation related to Git, TeamCity, infrastructure, and code.
    Your goal is to rephrase or expand the user's prompt to be more effective for keyword and semantic search.

    Guidelines:
    - Identify the core problem or request.
    - Extract key entities, tools, and error messages.
    - If needed, generate alternative phrasings or related keywords that would improve search relevance.
    - The output should be concise and contain ONLY the refined query string, without any conversational filler.

    Examples:
    User prompt: "My build is failing in TeamCity with a 'disk space' error."
    Refined query: "TeamCity build failure, disk space error, troubleshooting, fix, storage, agent, configuration."

    User prompt: "How do I revert a merge commit in Git?"
    Refined query: "Git revert merge commit, undo, rollback, fix."

    User prompt: "What is the best way to handle secrets in Jenkins pipelines?"
    Refined query: "Jenkins pipeline secrets, security, credentials management, environment variables."

    User prompt: {user_query}
    Refined query:"""

    try:
        response = ollama_client.generate(
            model=model,
            prompt=prompt,
            options={
                'temperature': 0.3,
                'max_tokens': 256,
            }
        )

        if 'response' in response and isinstance(response['response'], str):
            refined_query = response['response'].strip()
            logging.info(f"Successfully refined query from '{user_query}' to '{refined_query}'")
            return refined_query
        else:
            logging.error(
                f"Ollama response did not contain 'response' key or it was not a string for query: '{user_query}'. Response: {response}")
            return user_query

    except Exception as e:
        logging.error(f"Failed to refine user prompt '{user_query}' using LLM model '{model}': {e}", exc_info=True)
        return user_query


class OllamaChatSession:
    def __init__(self, model: str, system_prompt: str, max_history: int = 2):
        self.model = model
        self.system_prompt = system_prompt
        self.max_history = max_history  # Максимальное число сообщений в истории (пары запрос-ответ)
        self.messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_prompt}
        ]

    def _prune_history(self):
        # Оставляем system + последние max_history*2 сообщений (пары user/assistant)
        while len(self.messages) > 1 + self.max_history * 2:
            # Удаляем первую пару user + assistant
            for _ in range(2):
                if len(self.messages) > 1:
                    self.messages.pop(1)

    def clean_html_to_one_line(self, text: str) -> str:
        # Удаление всех HTML-тегов
        text_no_tags = re.sub(r'<[^>]+>', '', text)
        # Замена последовательностей пробелов, табуляций и переводов строк на один пробел
        text_flat = re.sub(r'\s+', ' ', text_no_tags)
        return text_flat.strip()

    def ask(self, context: str, query: str) -> str:
        user_message = f"""
        CONTEXT:
        {context}
        QUESTION:
        {query}
        Provide your response in HTML format following the system instructions.
        """
        self.messages.append({"role": "user", "content": user_message.strip()})

        self._prune_history()

        max_tokens = 4096
        response = ollama_client.chat(
            model=self.model,
            messages=self.messages,
            options={
                'temperature': 0.1,
                'max_tokens': max_tokens,
                "top_k": 20,
                "top_p": 0.8
            }
        )

        answer: str = response['message']['content'].strip()
        answer = answer.replace('```html', '').replace('```', '')

        self.messages.append({"role": "assistant", "content": self.clean_html_to_one_line(answer)})

        return answer


session = OllamaChatSession(LLM_MODEL, system_prompt)


def answer_question(context: str, query: str, model: str = LLM_MODEL) -> str:
    return session.ask(context, query)

    full_prompt = f"""
    CONTEXT:
    {context}
    QUESTION:
    {query}
    Provide your response in HTML format following the system instructions.
    """

    max_tokens = 4096
    response = ollama_client.generate(
        model=model,
        prompt=full_prompt,
        system=system_prompt,
        options={
            'temperature': 0.1,
            'max_tokens': max_tokens,
            "top_k": 20,
            "top_p": 0.8
        }
    )

    answer: str = response['response'].strip()
    # sanitize text answer
    answer = answer.replace('```html', '').replace('```', '')

    return answer.strip()


def format_answer(query: str, model: str = LLM_MODEL) -> str:
    system_prompt = f"""
    You are an experienced web developer specializing in HTML coding.
    Your task is to convert the given input text into a valid HTML document following HTML5 standards.
    Instructions:
    1. Any text that represents headings should be wrapped in the appropriate <h1>–<h6> tags. If no heading level is specified, default to <h2>.
    2. Each paragraph of text should be enclosed within a <p> tag.
    3. If you encounter lists (numbered or bulleted), convert them into the appropriate <ol> (ordered list) or <ul> (unordered list) containing <li> elements.
    4. The output should contain only HTML code without any additional explanations or comments.
    """

    print(f'===> Format answer:\n{query}')

    full_prompt = (
        "Input text:\n" 
        f"{query}"
    )
    max_tokens = 4096
    response = ollama_client.generate(
        model=model,
        prompt=full_prompt,
        system=system_prompt,
        options={
            'temperature': 0.1,
            'max_tokens': max_tokens,
            "top_k": 20,
            "top_p": 0.8
        }
    )

    answer: str = response['response'].strip()
    # sanitize text answer
    answer = answer.replace('```html', '').replace('```', '')

    return answer.strip()


# Глобальная история сообщений для текущего сеанса пользователя
conversation_history: List[Dict[str, str]] = []

def answer_question_history(context_parts: List[str], query: str, model: str = LLM_MODEL) -> str:
    global conversation_history

    context = '\n---\n'.join(context_parts)

    # Добавляем новый запрос в историю
    conversation_history.append({"role": "user", "content": f"Context: {context}\nQuestion: {query}"})

    # Собираем историю + новый запрос
    messages = [{"role": "system", "content": system_prompt}] + conversation_history

    max_tokens = 4096  # снизим до разумного лимита для Mistral-12B

    response = ollama_client.chat(
        model=model,
        messages=messages,
        options={
            'temperature': 0.1,
            'max_tokens': max_tokens,
            "top_k": 20,
            "top_p": 0.8
        }
    )

    answer: str = response['message']['content'].strip()
    answer = answer.replace('```html', '').replace('```', '')

    if MISSING_INFO_TEXT not in answer:
        conversation_history.append({"role": "assistant", "content": answer})

    return answer
