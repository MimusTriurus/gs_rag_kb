from typing import List, Dict, Optional

from langchain.chains.question_answering.map_reduce_prompt import messages
from ollama import Client
import json
from source.backend.settings import LLM_MODEL, MISSING_INFO_TEXT, OLLAMA_BASE_URL
from typing import List, Tuple, Dict
import logging
import re

ollama_client = Client(
    host=OLLAMA_BASE_URL
)

class QueryRefiner:
    def __init__(self, model: str, max_context_messages: int = 6):
        self.model = model
        self.max_context_messages = max_context_messages

    def refine(self, current_query: str, history: Optional[List[Dict[str, str]]] = None, tags: Optional[List[str]] = None) -> str:
        if history is None:
            history = []
        if tags is None:
            tags = []

        # Extract recent user queries and assistant replies
        recent_messages = history[-self.max_context_messages:]
        history_text = "\n".join(
            [f"{msg['role'].capitalize()}: {msg['content'].strip()}" for msg in recent_messages]
        )

        tags_text = ", ".join(tags) if tags else ""

        prompt = f"""
        Based on the following conversation history and provided tags, rewrite the user's current question
        to be more specific, clear, and helpful for a knowledge base search.
        If history is IRRELEVANT, return the ORIGINAL query verbatim.
        Only return the rewritten query. Do NOT include explanations or greetings.

        Conversation history:
        {history_text}

        User's current query:
        {current_query}

        Tags: {tags_text}
        """

        response = ollama_client.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt.strip()}],
            options={
                "temperature": 0.3,
                "max_tokens": 256,
                "top_k": 20,
                "top_p": 0.8
            }
        )

        refined = response['message']['content'].strip()
        return refined

if __name__ == '__main__':
    refiner = QueryRefiner(model=LLM_MODEL)
    messages = [
        {"role": "user", "content": 'What is the version of PR 11208?'},
        {"role": "assistant", "content": 'v0.42.1.3151-614ebe63'},
        {"role": "user", "content": 'When was this PR closed?'},
        {"role": "assistant", "content": '2025-05-21 15:33:28'},
    ]
    old_query = "Provide a 'player description' for this PR"
    new_query = refiner.refine(old_query, history=messages, tags=[])
    print(new_query)