import logging
import re
from typing import List, Dict, Optional, Tuple

import numpy as np
from ollama import Client
from sentence_transformers import SentenceTransformer, util
from source.backend.settings import OLLAMA_BASE_URL, EMBED_MODEL_NAME


class QueryRefiner:
    def __init__(self, model: str, temperature: float = 0.2, alternatives_count: int = 2):
        self.ollama_client = Client(
            host=OLLAMA_BASE_URL
        )
        self.model = model
        self.temperature = temperature
        self.alternatives_count = alternatives_count

    def refine(self, current_query: str) -> str:
        if not re.search(r'[.?!]\s*$', current_query):
            current_query = current_query.rstrip() + '?'

        prompt = f'''
        You are a search-query paraphraser. Produce {self.alternatives_count} concise alternatives
        that capture the same intent but use varied wording.
        Original query: "{current_query}"
        '''

        prompt = re.sub(r'\s+', ' ', prompt)

        response = self.ollama_client.generate(
            model=self.model,
            prompt=prompt,
            options={
                'temperature': self.temperature,
                'max_tokens': 256,
            }
        )

        refined_query = current_query
        if 'response' in response and isinstance(response['response'], str):
            refined_query = response['response'].strip()
            logging.info(f"Successfully refined query from '{current_query}' to '{refined_query}'")
            refined_query = re.sub(r'\d+\.\s*', '', refined_query)
            refined_query = re.sub(r'\s+', ' ', refined_query)
            refined_query = refined_query.replace('"', '')
            return f'{current_query} {refined_query}'
        return refined_query


class QueryRefinerBasedOnHistory:
    def __init__(self, model: str, max_context_messages: int = 6):
        self.ollama_client = Client(
            host=OLLAMA_BASE_URL
        )
        self.model = model
        self.max_context_messages = max_context_messages

    def refine(
            self,
            current_query: str,
            history: Optional[List[Dict[str, str]]] = None,
            tags: Optional[List[str]] = None
    ) -> str:
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
        Be brief and concise.

        Conversation history:
        {history_text}

        User's current query:
        {current_query}

        Tags: {tags_text}
        """

        prompt = f"""
        If the user's current query is unclear, try rewriting the user's current query.
        to make it more specific, understandable, and useful for searching the knowledge base based on the conversation history below.
        If the user's ORIGINAL query is clear and understandable, return the ORIGINAL user's query verbatim.
        If the history is IRRELEVANT, return the ORIGINAL user's query verbatim.
        DO NOT include explanations or greetings.
        Be brief and concise.

        Conversation history:
        {history_text}

        User's current query:
        {current_query}
        """

        prompt = re.sub(r'\s+', ' ', prompt)

        response = self.ollama_client.chat(
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


class QueryVariantsRefiner:
    def __init__(self, model: str, metadata_topics: List[str]):
        self.ollama_client = Client(
            host=OLLAMA_BASE_URL
        )
        self.model = model
        self.metadata_topics = metadata_topics
        self.metadata_topics_set = set(topic.lower() for topic in metadata_topics)
        self.embedding_model = SentenceTransformer(EMBED_MODEL_NAME)
        self.topic_embeddings = self.embedding_model.encode(metadata_topics, convert_to_numpy=True, normalize_embeddings=True)

    def is_relevant(self, query: str) -> bool:
        """
        Determine if the query is related to any known metadata topic.
        """
        response = self.ollama_client.chat(model=self.model, messages=[
            {
                "role": "system",
                "content": f"You are a classifier that checks if a user query relates to any of these topics: {', '.join(self.metadata_topics)}."
            },
            {
                "role": "user",
                "content": f"Query: {query}\nAnswer yes or no."
            }
        ])
        return 'yes' in response['message']['content'].lower()

    def is_relevant_embedding(self, query: str, threshold: float = 0.5) -> Tuple[Optional[str], Optional[float]]:
        """
        Determine the most relevant topic based on semantic similarity using numpy.
        Returns (best_topic, score) if above threshold, else None.
        """
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True, normalize_embeddings=True)  # shape: (1, D)
        scores = np.dot(self.topic_embeddings, query_embedding.T).squeeze()  # shape: (N,)
        max_index = np.argmax(scores)
        max_score = scores[max_index]

        if max_score >= threshold:
            return self.metadata_topics[max_index], float(max_score)
        else:
            return None, None

    def generate_clarifications(self, query: str, max_variants: int = 3) -> List[str]:
        """
        Generate refined versions of the query.
        """
        topics_list = ', '.join(self.metadata_topics)
        prompt = f"""
        The user asked: "{query}"
        This question did not return any results in the knowledge base.
        Based on the topics [{topics_list}], suggest {max_variants} more specific or alternative versions of this query
        that could help in getting better results from a knowledge base.
        Output only the list.
        """

        response = self.ollama_client.chat(model=self.model, messages=[
            {"role": "system", "content": "You are a helpful assistant that rewrites unclear or unanswerable questions based on a fixed topic set."},
            {"role": "user", "content": prompt}
        ])

        suggestions = response['message']['content'].strip().split('\n')
        return [s.strip('- ').strip() for s in suggestions if s.strip()]