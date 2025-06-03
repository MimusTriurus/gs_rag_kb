from ollama import Client
from settings import LLM_MODEL, missing_info_text, OLLAMA_BASE_URL

ollama_client = Client(
    host=OLLAMA_BASE_URL
)


def refine_user_prompt(user_query: str, model: str = LLM_MODEL) -> str:
    prompt = f"""
    Your task is to refine the user prompt below, preserving its meaning.
    Steps to follow:
    1. Identify the main question or request.
    2. If there are multiple tasks, list them.
    3. Keep the text concise and clear.
    User prompt: {user_query}
    Refined user's prompt:"""

    response = ollama_client.generate(
        model=model,
        prompt=prompt,
        options={
            'temperature': 0.3,
            'max_tokens': 256,
        }
    )
    return response['response'].strip()


def answer_question(context: str, query: str, model: str = LLM_MODEL) -> str:
    system_prompt = f"""
    You are an assistant. Use only the context below to answer.
    Follow these rules STRICTLY:
    1. If the context doesn't contain needed information, respond ONLY with "{missing_info_text}"
    Format text to HTML using these rules:
    1. Highlight key terms with <strong>
    2. Convert lists to <ul> or <ol> with <li> items
    3. Wrap code in <pre><code> blocks
    4. Use <h2> for subtitles, <p> for paragraphs
    5. Add class="section" to wrapper divs
    6. Escape special characters (&, <, >)
    7. Never add explanations
    8. Preserve original order
    9. Use <a> tags for URLs
    """

    full_prompt = (
        f"Context: {context}\n"
        f"Based EXCLUSIVELY on this context.\n"
        f"Question: {query}\n"
        f"If answer is not in context, say STRICTLY ONLY '{missing_info_text}'"
        # f"Answer:"
    )
    max_tokens = 2048 * 2
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
