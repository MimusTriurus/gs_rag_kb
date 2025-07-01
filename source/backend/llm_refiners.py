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
        Based ONLY on the following conversation history, rewrite the user's current question
        to be more specific, clear, and helpful for a knowledge base search.
        Do not add information from other sources.
        If history is IRRELEVANT or EMPTY, return the ORIGINAL query verbatim.
        Only return the rewritten query. Do NOT include explanations or greetings.
        Be brief and concise.

        Conversation history:
        {history_text}

        User's current query:
        {current_query}
        """

        prompt1 = f"""
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

        prompt = f'''
        You are an assistant that helps refine user questions.
        Your task is to rewrite the user's current query to make it more clear, specific, and suitable for knowledge base search.
        Guidelines:
        Use only the chat history between the user and the assistant to understand the user's intent.
        Do not invent or assume new information.
        Avoid creativity or speculation.
        Preserve the original meaning of the query.
        Use a neutral and concise tone.
        Format:
        Return only the revised version of the question, without commentary or explanation.
        Chat history:
        {history_text}
        User's question:
        {current_query}
        '''

        prompt = re.sub(r'\s+', ' ', prompt)

        response = self.ollama_client.generate(
            model=self.model,
            prompt=prompt.strip(),
            options={
                "temperature": 0.1,
                "max_tokens": 256,
                "top_k": 20,
                "top_p": 0.8
            }
        )

        refined = response['response'].strip()
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
        self.topic_embeddings = self.embedding_model.encode(metadata_topics, convert_to_numpy=True,
                                                            normalize_embeddings=True)

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
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True,
                                                      normalize_embeddings=True)  # shape: (1, D)
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
            {"role": "system",
             "content": "You are a helpful assistant that rewrites unclear or unanswerable questions based on a fixed topic set."},
            {"role": "user", "content": prompt}
        ])

        suggestions = response['message']['content'].strip().split('\n')
        return [s.strip('- ').strip() for s in suggestions if s.strip()]


class QueryChecker:
    def __init__(self, model: str):
        self.ollama_client = Client(
            host=OLLAMA_BASE_URL
        )
        self.model = model

    def is_relevant(self, query: str) -> bool:
        theme = '''
This text is a list of pull requests for updates in a video game
version, each containing changes related to technical adjustments (e.g., configuration
files, logging, language translations), bug fixes, and hotfixes. The changes aim
to improve the game performance, stability, and user experience across different
versions (master, 0.40-rc, 0.41-rc, etc.), with a focus on resolving issues that
may have impacted players' gameplay during version v42.
        '''

        system_prompt = f'''
        User query: "{query}"  
        Question: is this clear and comprehensive enough to search for information on the topic of "{theme}"?
        Don't be strict.  
        If not, ask 1-2 clarifying questions.
        Response format:  
        is_clear: <yes/no>  
        follow_up: <list of questions or empty>'''

        response = self.ollama_client.chat(model=self.model, messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": f"{query}"
            }
        ], options={
            'temperature': 0.7,
            "top_k": 20,
            "top_p": 0.8
        })
        return response['message']['content'].lower()
