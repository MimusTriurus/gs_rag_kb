import os

DOCUMENTS_PATH = "documents/"
CACHE_DIR = "cache/"
EMBED_MODEL_NAME = "models/bge-large-en"
CROSS_ENCODER_NAME = "models/ms-marco-MiniLM-L6-v2"
LLM_MODEL = "mistral:7b-instruct"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
MAX_CHUNK_SIZE = 1500
OVERLAP_BLOCKS = 2
TOP_K_RETRIEVAL = 50
TOP_K_RERANK = 2
TOP_K_FILE_SELECT = 2

need_2_refine_query = False
clean_chunk_markdown = False

missing_info_text = "No information"
no_info_in_knowledge_base_message = '''
The information is missing from the knowledge base. We will analyze your request and update our knowledge base.
'''
