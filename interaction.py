import ollama
from settings import LLM_MODEL


def refine_user_prompt(user_query: str, model: str = LLM_MODEL) -> str:
    prompt = f"""
    Your task is to refine the user prompt below, preserving its meaning.
    Steps to follow:
    1. Identify the main question or request.
    2. If there are multiple tasks, list them.
    3. Keep the text concise and clear.
    User prompt: {user_query}
    Refined user's prompt:"""

    response = ollama.generate(
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

    full_prompt = f"Context: {context}\nQuestion: {query}\nAnswer:"
    max_tokens = 2048 * 2
    response = ollama.generate(
        model=model,
        prompt=full_prompt,
        system=system_prompt,
        options={
            'temperature': 0.7,
            'max_tokens': max_tokens,
        }
    )

    answer: str = response['response'].strip()
    # sanitize text answer
    answer = answer.replace('```html', '').replace('```', '')

    return answer.strip()
