import spacy
import neuralcoref
from ollama import ChatResponse, chat  # заменили openai

from source.backend.settings import LLM_MODEL

# 1. Настройка SpaCy + NeuralCoref
nlp = spacy.load("en_core_web_sm")
neuralcoref.add_to_pipe(nlp)


def is_query_ambiguous_via_coref(query: str) -> bool:
    doc = nlp(query)
    for cluster in doc._.coref_clusters:
        for mention in cluster.mentions:
            if mention.root.tag_ in ("PRP", "PRP$") and len(cluster) == 1:
                return True
    return False


def llm_check_and_clarify(query: str) -> dict:
    messages = [
        {"role": "system", "content": "You detect pronoun ambiguity."},
        {"role": "user", "content": (
            'Determine if the following user query is ambiguous—'
            'contains pronouns like "it", "they", "his" without clear reference. '
            'Response JSON: {"ambiguous": true/false, "clarifying_question": "<query>" or null}. '
            f'User query: "{query}"'
        )}
    ]
    resp: ChatResponse = chat(model=LLM_MODEL, messages=messages, stream=False)
    content = resp.message.content
    return content  # ожидаем JSON от модели


def process_user_query(query: str):
    print(f"> Проверка coref-неопределённости: '{query}'")
    coref_amb = is_query_ambiguous_via_coref(query)
    print(f"- По coref: {'неопределённый' if coref_amb else 'ясный'}")

    print("> Проверка через Ollama LLM...")
    llm_res = llm_check_and_clarify(query)
    print(f"- LLM ответ: {llm_res}")


if __name__ == "__main__":
    tests = [
        "What's the status of their proposal?",
        "How do I fix it?",
        "Show me the latest sales figures.",
        "Can you summarize it?"
    ]
    for q in tests:
        process_user_query(q)
