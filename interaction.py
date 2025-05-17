from llama_cpp import Llama


def refine_user_prompt(llm, user_query):
    prompt = (
        "Your task is to refine the user prompt below, preserving its meaning.\n"
        "Steps to follow:\n"
        "1. Identify the main question or request.\n"
        "2. If there are multiple tasks, list them.\n"
        "3. Keep the text concise and clear.\n"
        f"User prompt:{user_query}\n"
        "Now, provide the improved prompt below:"
    )
    resp = llm(prompt, max_tokens=256, stop=["\n\n"])
    return resp["choices"][0]["text"].strip()


END_OF_ANSWER = '\n\n'


def answer_question(llm, context, query):
    prompt = (
        f"You are an assistant. Use only the context below to answer."
        f"Format the Answer using html tags for better text perception in the browser."
        f"Highlighting significant words in bold, substitute the <a> tag for Internet addresses, etc."
        f"Use the {END_OF_ANSWER} character sequence only as a sign of the end of the answer."
        f"Context:{context}"
        f"Question: {query}Answer:"
    )
    resp = llm(prompt, max_tokens=1024, stop=[END_OF_ANSWER])
    return resp["choices"][0]["text"].strip()
