import ollama
import json
from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime

from source.backend.settings import LLM_MODEL


@dataclass
class ChatMessage:
    """Structure for storing chat messages"""
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: datetime
    context_used: Optional[str] = None


class RAGChatSystem:
    def __init__(self, model_name: str = LLM_MODEL, max_history: int = 10):
        """
        Initialize RAG system with chat history support

        Args:
            model_name: Ollama model name
            max_history: Maximum number of messages in history
        """
        self.model_name = model_name
        self.max_history = max_history
        self.chat_history: List[ChatMessage] = []
        self.missing_info_text = "Information not available in the provided context"

    def _build_chat_history_context(self) -> str:
        """Builds context from chat history"""
        if not self.chat_history:
            return "Chat history is empty."

        history_context = "PREVIOUS MESSAGES HISTORY:\n"
        for i, msg in enumerate(self.chat_history[-self.max_history:], 1):
            role_name = "User" if msg.role == "user" else "Assistant"
            history_context += f"{i}. {role_name}: {msg.content}\n"
            if msg.context_used and msg.role == "assistant":
                history_context += f"   (Context used: {msg.context_used[:100]}...)\n"

        return history_context + "\n"

    def _create_system_prompt(self, context: str) -> str:
        """Creates system prompt with context and history"""
        chat_history_context = self._build_chat_history_context()

        system_prompt = f"""
You are an information extraction assistant for a RAG system.
Your role is to answer questions using ONLY the provided context and chat history.

IMPORTANT RULES FOR WORKING WITH HISTORY:
1. Analyze previous messages to understand the conversation context
2. If the current question is clarifying or refers to previous answers, use information from history
3. Pronouns and references ("this", "it", "he", "she", "mentioned above", "in the previous answer") should be interpreted based on chat history
4. If a question is incomplete or unclear, use context from previous messages for interpretation

OUTPUT FORMAT - HTML:
1. Wrap headings in appropriate <h1>–<h6> tags (default to <h2>)
2. Enclose each text paragraph within <p> tags
3. Convert lists to <ol> (ordered) or <ul> (unordered) with <li> elements
4. Output ONLY HTML code without additional explanations or comments
5. Ensure valid HTML structure

CORE RULES:
1. Use STRICTLY ONLY information from the PROVIDED CONTEXT and CHAT HISTORY
2. DO NOT add knowledge from your training data
3. Be precise and factual
4. If information is not available in context or history, respond STRICTLY ONLY: '{self.missing_info_text}'
5. When answering clarifying questions, always reference specific information from history or context

{chat_history_context}

PROVIDED CONTEXT:
{context}

INSTRUCTION: Answer the following user question, considering all available information from context and chat history."""

        return system_prompt

    def _add_to_history(self, role: str, content: str, context_used: Optional[str] = None):
        """Добавляет сообщение в историю чата"""
        message = ChatMessage(
            role=role,
            content=content,
            timestamp=datetime.now(),
            context_used=context_used
        )
        self.chat_history.append(message)

        # Ограничиваем размер истории
        if len(self.chat_history) > self.max_history * 2:  # *2 для user+assistant пар
            self.chat_history = self.chat_history[-self.max_history * 2:]

    def query(self, user_question: str, context: str) -> str:
        """
        Обрабатывает запрос пользователя с учетом контекста и истории

        Args:
            user_question: Вопрос пользователя
            context: Контекст для ответа

        Returns:
            Ответ в формате HTML
        """
        # Добавляем вопрос пользователя в историю
        self._add_to_history("user", user_question)

        # Создаем системный prompt
        system_prompt = self._create_system_prompt(context)

        try:
            # Отправляем запрос в Ollama
            response = ollama.generate(
                model=self.model_name,
                prompt=f"{system_prompt}\n\nUSER QUESTION: {user_question}",
                options={
                    "temperature": 0.2,  # Low temperature for accuracy
                    "top_p": 0.9,
                    "top_k": 40,
                    "num_predict": 2000,
                    "stop": ["USER QUESTION:", "PROVIDED CONTEXT:"]
                }
            )

            answer = response['response'].strip()

            # Добавляем ответ в историю
            self._add_to_history("assistant", answer, context[:200])

            return answer

        except Exception as e:
            error_msg = f"<p>Error processing request: {str(e)}</p>"
            self._add_to_history("assistant", error_msg)
            return error_msg

    def clear_history(self):
        """Clears chat history"""
        self.chat_history.clear()

    def get_history(self) -> List[Dict]:
        """Returns chat history in JSON format"""
        return [
            {
                "role": msg.role,
                "content": msg.content,
                "timestamp": msg.timestamp.isoformat(),
                "context_used": msg.context_used
            }
            for msg in self.chat_history
        ]

    def save_history(self, filename: str):
        """Saves chat history to file"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.get_history(), f, ensure_ascii=False, indent=2)

    def load_history(self, filename: str):
        """Loads chat history from file"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                history_data = json.load(f)

            self.chat_history = []
            for msg_data in history_data:
                msg = ChatMessage(
                    role=msg_data["role"],
                    content=msg_data["content"],
                    timestamp=datetime.fromisoformat(msg_data["timestamp"]),
                    context_used=msg_data.get("context_used")
                )
                self.chat_history.append(msg)

        except FileNotFoundError:
            print(f"File {filename} not found")
        except Exception as e:
            print(f"Error loading history: {e}")


# Test scenarios and example usage
def run_test_scenarios():
    """Runs automated test scenarios to demonstrate the system"""
    # Initialize system
    rag_system = RAGChatSystem(model_name=LLM_MODEL, max_history=5)

    # English context
    context = """
    Artificial Intelligence (AI) is a field of computer science that focuses on creating machines 
    capable of performing tasks that typically require human intelligence. The main areas of AI include:

    1. Machine Learning - a method of teaching computers without explicit programming
    2. Deep Learning - a subset of machine learning that uses neural networks
    3. Natural Language Processing - the ability of computers to understand and generate human language
    4. Computer Vision - the ability of computers to interpret visual information
    5. Robotics - the integration of AI with physical machines

    AI applications include autonomous vehicles, medical diagnosis, financial analysis, and much more.
    The field has seen rapid growth since 2010, with breakthrough developments in neural networks and 
    large language models. Major companies like Google, Microsoft, and OpenAI are leading research efforts.
    """

    # Test scenarios
    test_questions = [
        {
            "question": "What is Machine Learning?",
            "description": "Basic information extraction"
        },
        {
            "question": "What are the other areas mentioned?",
            "description": "Reference to previous context"
        },
        {
            "question": "Tell me more about the second one",
            "description": "Pronoun reference to previous list"
        },
        {
            "question": "What companies are mentioned in the text?",
            "description": "Specific information extraction"
        },
        {
            "question": "When did the field see rapid growth?",
            "description": "Timeline information"
        },
        {
            "question": "What applications does it have?",
            "description": "Pronoun reference to AI"
        },
        {
            "question": "What is quantum computing?",
            "description": "Information not in context"
        },
        {
            "question": "How does the first application work?",
            "description": "Reference to previous answer"
        }
    ]

    print("=" * 80)
    print("RAG SYSTEM TEST SCENARIOS")
    print("=" * 80)

    for i, test in enumerate(test_questions, 1):
        print(f"\n[TEST {i}] {test['description']}")
        print(f"Question: {test['question']}")
        print("-" * 40)

        answer = rag_system.query(test['question'], context)
        print(f"Answer: {answer}")
        print("=" * 80)

    # Show final history
    print("\nFINAL CHAT HISTORY:")
    history = rag_system.get_history()
    for msg in history:
        role = "👤 USER" if msg['role'] == 'user' else "🤖 ASSISTANT"
        content = msg['content'][:150] + "..." if len(msg['content']) > 150 else msg['content']
        print(f"{role}: {content}")

    return rag_system


# Example usage with interactive features
def interactive_demo():
    """Demonstrates interactive features of the system"""
    rag_system = RAGChatSystem(model_name="llama3.2", max_history=5)

    # Tech context
    context = """
    Cloud Computing is a technology that delivers computing services over the internet. 
    The main service models are:

    1. Infrastructure as a Service (IaaS) - provides virtualized computing resources
    2. Platform as a Service (PaaS) - provides development platforms
    3. Software as a Service (SaaS) - provides software applications

    Major cloud providers include Amazon Web Services (AWS), Microsoft Azure, and Google Cloud Platform.
    Benefits include scalability, cost-effectiveness, and accessibility from anywhere.
    """

    print("Interactive Demo - Cloud Computing Context")
    print("Available commands: 'history', 'clear', 'save', 'quit'")
    print("=" * 60)

    while True:
        user_input = input("\nYour question: ").strip()

        if user_input.lower() == 'quit':
            break
        elif user_input.lower() == 'clear':
            rag_system.clear_history()
            print("✅ Chat history cleared.")
            continue
        elif user_input.lower() == 'history':
            history = rag_system.get_history()
            print("\n📚 Chat History:")
            for msg in history:
                role = "👤" if msg['role'] == 'user' else "🤖"
                print(f"{role} {msg['content'][:100]}...")
            continue
        elif user_input.lower() == 'save':
            rag_system.save_history('chat_history.json')
            print("💾 History saved to chat_history.json")
            continue

        if not user_input:
            continue

        print("\n🤖 Answer:")
        answer = rag_system.query(user_input, context)
        print(answer)
        print("-" * 60)


if __name__ == "__main__":
    print("Choose mode:")
    print("1. Run automated test scenarios")
    print("2. Interactive demo")
    print("3. Both")

    choice = input("Enter choice (1-3): ").strip()

    if choice == '1':
        run_test_scenarios()
    elif choice == '2':
        interactive_demo()
    elif choice == '3':
        run_test_scenarios()
        print("\n" + "=" * 80)
        print("SWITCHING TO INTERACTIVE DEMO")
        print("=" * 80)
        interactive_demo()
    else:
        print("Running automated tests by default...")
        run_test_scenarios()