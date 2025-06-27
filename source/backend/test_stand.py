from typing import List, Dict, Optional

from langchain.chains.question_answering.map_reduce_prompt import messages
from ollama import Client
import json

from source.backend.interaction import answer_question, clean_html_to_one_line
from source.backend.settings import LLM_MODEL, MISSING_INFO_TEXT, OLLAMA_BASE_URL
from typing import List, Tuple, Dict
import logging
import re

ollama_client = Client(
    host=OLLAMA_BASE_URL
)

context = '''
## Pull request: 11215
Close date: 2025-05-19 11:49:27
Version: v0.42.1.3148-ded852e9
## Description
* Nothing player facing.
R3-67981 EAC Deployments
## Changes
* Added config patch for EAC stage environment
## Player Description
* Nothing player facing.
## QA Description
* I've modified the WGC and STEAM config patch to point to prod EAC deploymentID and the rest of the configs to stage EAC deploymentID

## Pull request: 11208
Close date: 2025-05-21 15:33:28
Version: v0.42.1.3151-614ebe63
## Description
N/A: The issue appeared during v42 and shouldn't have reached the players
R3-67865 Removal of the skins unavailable in the Shipping from the bots' config.
## Changes
Hotfix version of
https://stash.wargaming.net/projects/R3/repos/game/pull-requests/11203/overview
* R3-66930 Removal of the skins unavailable in the Shipping from the bots' config.
* Improved the logging across the scenario.
* The code quality improvements in the related files
## Player Description
N/A: The issue appeared during v42 and shouldn't have reached the players
## QA Description
* Note: the original PR contains code style discussion not completely resolved by this moment. But it doesn't affect the code stability and quite minor in terms of code quality, so it doesn't stop, with @g_booker 's approval, this PR from being processed

## Pull request: 11234
Close date: 2025-05-22 09:15:39
Version: v0.42.1.3153-122948c8
## Description
* The game is back to 100% loc
* Some russian wording got fixed to be more idiotmatics
R3-68287 update all languages to 100%
## Changes
* Updated all languages and fixed some suboptimal russian translations
## Player Description
* The game is back to 100% loc
* Some russian wording got fixed to be more idiotmatics
## QA Description
* Amber task https://jira.wargaming.net/browse/R3-67988
* If release QA is faster, they are free to go ahead
'''

questions = [
    'What is the version of PR 11234?',
    'How to install Git on Linux?',
    'When was this PR closed?',
    'Provide me information about Changes for this PR',
    'Who is Putin?'
]


class QueryRefinerBasedOnTags:
    def __init__(self, model: str, tags: List[str]):
        self.model = model

        self.topics = '\\n- ' + '\\n- '.join(tags)

    def refine(self, current_query: str) -> str:
        prompt = f'''
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
    
        Use exclusively the provided TOPICS to answer the QUESTION.
        - If the TOPICS do not contain sufficient information, return USER'S QUESTION as it is
        - Do NOT make assumptions or use outside knowledge.
        - Do NOT add explanations, reasoning steps, or disclaimers.
        - Maintain the original order of information as found in the context.
        - Escape all special characters properly (&, <, >).
        
        TOPICS:
        {self.topics}
    
        User prompt: {current_query}
        Refined query:
        '''

        response = ollama_client.generate(
            model=LLM_MODEL,
            prompt=prompt,
            options={
                'temperature': 0.3,
                'max_tokens': 256,
            }
        )
        refined_query = ''
        if 'response' in response and isinstance(response['response'], str):
            refined_query = response['response'].strip()
            logging.info(f"Successfully refined query from '{current_query}' to '{refined_query}'")
            return refined_query
        return refined_query


class QueryRefinerBasedOnHistory:
    def __init__(self, model: str, max_context_messages: int = 6):
        self.model = model
        self.max_context_messages = max_context_messages

    def refine(self, current_query: str, history: Optional[List[Dict[str, str]]] = None,
               tags: Optional[List[str]] = None) -> str:
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


def test_refiner():
    refiner = QueryRefinerBasedOnHistory(model=LLM_MODEL)
    messages = [
        {"role": "user", "content": 'What is the version of PR 11208?'},
        {"role": "assistant", "content": 'v0.42.1.3151-614ebe63'},
        {"role": "user", "content": 'When was this PR closed?'},
        {"role": "assistant", "content": '2025-05-21 15:33:28'},
    ]
    # old_query = "Provide a 'player description' for this PR"
    old_query = "How to install Git on Linux?"
    new_query = refiner.refine(old_query, history=messages, tags=[])
    print(new_query)


def test_chat():
    for q in questions:
        answer = answer_question(context, q)
        print(f'==> {clean_html_to_one_line(answer)}\n\n')


def test_refiner_based_on_topics():
    topics = [
        '''
        This text is a list of pull requests for updates in a video game
          version, each containing changes related to technical adjustments (e.g., configuration
          files, logging, language translations), bug fixes, and hotfixes. The changes aim
          to improve the game performance, stability, and user experience across different
          versions (master, 0.40-rc, 0.41-rc, etc.), with a focus on resolving issues that
          may have impacted players' gameplay during version v42.\n
        ''',
        '''
        Here''s a summary of the systems listed:1. **Development and QA
        Build Agents** in the UK office (with i9-9900 8C16T, RTX 2070 GPUs, and 32GB RAM):
        These are for both DevOps and AutoQA teams located in the UK office. They have two
        variants with either 8 or 16 cores.2. **Production Build Agents** in the UK (with
        AMD Ryzen 9 5900X 16C32T): These are for the DevOps team in the UK and have 64GB
        of RAM and a large SSD + HDD storage.3. **Build Agent** in the dra4 data center:
        This is a G-Core VM located in the dra4 data center with 8 cores, 64GB of RAM, and
        1TB of storage. It runs Linux or Oracle 9.The table seems to be a mix of physical
        and virtual systems (PCs and VMs), with different specifications for various use
        cases such as development, testing, and production environments. The locations for
        the systems are either in the UK office or data center (dra4). The teams using these
        machines include DevOps and AutoQA.\n
        ''',
        '''
        This text explains how to connect a dedicated server locally without
        using a backend, including creating and configuring the necessary files, setting
        up connections for multiple clients, and various methods of connecting as a client.
        It also includes troubleshooting tips and instructions on starting the server from
        the editor.\n
        ''',
        '''
        This text outlines various performance-related resources for a game
        project, including documentation and server URLs for play test runner, exportana,
        unreal trace server, Perfana, and TraceObserver. The resources are divided into
        stages (production, staging) and tests, with associated storage locations, repositories,
        and logs for each component.\n
        '''
    ]
    refiner = QueryRefinerBasedOnTags(model=LLM_MODEL, tags=topics)
    for q in questions:
        answer = refiner.refine(q)
        print(f'old: {q}')
        print(f'new: {answer}')
        print('')


if __name__ == '__main__':
    test_refiner_based_on_topics()
    exit(0)
