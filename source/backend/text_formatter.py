import ollama
import textwrap
import re

from settings import LLM_MODEL


class TextFormatter:
    def __init__(self, model_name=LLM_MODEL):
        self.model = model_name
        self.chunk_size = 3000  # Optimal chunk size for Mistral

    def process_large_text(self, input_text, output_file="documents/connect_2_local_dedic.md"):
        """Main text processing method"""
        chunks = self._split_text(input_text)
        processed_chunks = []

        for i, chunk in enumerate(chunks, 1):
            print(f"Processing chunk {i}/{len(chunks)}...")
            processed = self._process_chunk(chunk)
            processed_chunks.append(processed)

        self._save_output(processed_chunks, output_file)
        return output_file

    def _split_text(self, text):
        """Split text while preserving sentence boundaries"""
        sentences = re.split(r'(?<=[.!?]) +', text)
        chunks = []
        current_chunk = []

        for sentence in sentences:
            if len(' '.join(current_chunk + [sentence])) > self.chunk_size:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
            current_chunk.append(sentence)

        if current_chunk:
            chunks.append(' '.join(current_chunk))
        return chunks

    def _process_chunk(self, text_chunk):
        """Process text chunk with Ollama"""
        prompt = textwrap.dedent(f"""
            [INST] Format the following text into Markdown:
            1. Fix grammatical errors
            2. Add structure with headers
            3. Highlight lists, tables and code when needed
            4. Preserve original meaning
            5. Use proper punctuation
            6. Split into logical paragraphs

            Text:
            {text_chunk}

            Return only formatted text without additional comments.
            [/INST]
        """)

        response = ollama.generate(
            model=self.model,
            prompt=prompt,
            options={
                'temperature': 0.1,
                'num_predict': 2048,
                'repeat_penalty': 1.1
            }
        )
        return response['response'].strip()

    def _save_output(self, chunks, filename):
        """Assemble and save final result"""
        full_text = ''.join(chunks)
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(full_text)
        print(f"Document saved to {filename}")


if __name__ == "__main__":
    formatter = TextFormatter()
    with open('documents/build_agents_list.md', 'r', encoding='utf-8') as f:
        raw_text = f.read()
    formatter.process_large_text(raw_text)
