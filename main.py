import os
from pathlib import Path
from sentence_transformers import SentenceTransformer, CrossEncoder
from llama_cpp import Llama
from document_utils import parse_documents, build_or_load_index_for_file, select_best_files, retrieve_and_rerank
from interaction import refine_user_prompt, answer_question

from settings import (
    DOCUMENTS_PATH,
    CACHE_DIR,
    EMBED_MODEL_NAME,
    CROSS_ENCODER_NAME,
    GGUF_MODEL_PATH,
    need_2_refine_query
)

os.makedirs(CACHE_DIR, exist_ok=True)


def main():
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    cross_encoder = CrossEncoder(CROSS_ENCODER_NAME)
    llm = Llama(
        model_path=GGUF_MODEL_PATH,
        embedding=False,
        n_ctx=2048,
        n_threads=4,
        verbose=False
    )

    file_indices, file_titles, file_paths, file_meta = parse_documents(
        DOCUMENTS_PATH,
        embed_model
    )

    print("[main] Ready to answer questions. Type 'exit' to quit.")
    while True:
        query = input("Enter your question: ")
        if query.lower() in ("exit", "quit"):
            print("Exiting...")
            break

        refined = refine_user_prompt(llm, query) if need_2_refine_query else query
        selected = select_best_files(refined, file_titles, file_paths, embed_model)

        results = []
        for fname in selected:
            idx, chunks, sources = file_indices[fname]
            results.extend(
                retrieve_and_rerank(
                    embed_model,
                    cross_encoder,
                    idx,
                    chunks,
                    sources,
                    refined
                )
            )

        context = []
        for text, src in results:
            url, author = file_meta.get(src, ("", ""))
            context.append(f"Source: {src} (URL: {url}, Author: {author}){text}")
        answer = answer_question(llm, "---".join(context), refined)
        print(f"Answer: {answer}")


if __name__ == "__main__":
    main()