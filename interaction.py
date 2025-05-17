from llama_cpp import Llama


def refine_user_prompt(llm, user_query):
    prompt = (
        "Your task is to refine the user prompt below, preserving its meaning."
        "Steps to follow:"
        "1. Identify the main question or request."
        "2. If there are multiple tasks, list them."
        "3. Keep the text concise and clear."
        f"User prompt:{user_query}"
        "Now, provide the improved prompt below:"
    )
    resp = llm(prompt, max_tokens=256, stop=[""])
    return resp["choices"][0]["text"].strip()


def answer_question(llm, context, query):
    prompt = (
        "You are an assistant. Use only the context below to answer."
        f"Context:{context}"
        f"Question: {query}Answer:"
    )
    resp = llm(prompt, max_tokens=256, stop=[""])
    return resp["choices"][0]["text"].strip()
