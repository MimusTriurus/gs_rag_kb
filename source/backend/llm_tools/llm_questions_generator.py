import json

from ollama import Client
import re
from source.backend.settings import OLLAMA_BASE_URL

class LlmQuestionsGenerator:
    def __init__(self, model: str):
        self.ollama_client = Client(
            host=OLLAMA_BASE_URL
        )
        self.model = model
        self.prompt = '''
        Identify the general topic of the Сontext and generate 3 short general questions.
        Format - JSON.
        Context:
        '''
        self.answer_format = '''
        Output format: JSON
        Example:
        {
            "general_topic": "<TOPIC TITLE IS HERE>",
            "questions": [
                {
                    "question": "<ANSWER IS HERE>"
                }
            ]
        }
        '''
        self.pattern = re.compile(r"```json\s*([\s\S]*?)```", re.MULTILINE)

    def generate(self, context: str):
        instruction = f'''
        {self.prompt}
        {context}
        {self.answer_format}
        '''


        response = self.ollama_client.generate(
            model=self.model,
            prompt=instruction,
            options={
                'temperature': 0.9,
                'max_tokens': 2048,
                "top_k": 20,
                "top_p": 0.8
            }
        )
        answer = response['response']
        json_blocks = self.pattern.findall(answer)
        generated_questions = None
        if json_blocks:
            generated_questions = json.loads(json_blocks[0])
        else:
            generated_questions = json.loads(answer)
        return generated_questions