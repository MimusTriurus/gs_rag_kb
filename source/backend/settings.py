import os

DOCUMENTS_PATH = 'documents/'
TEST_DOCUMENTS_PATH = os.getenv('TEST_DOCUMENTS_PATH', 'test_docs/')
CACHE_DIR = 'cache/'
EMBED_MODEL_NAME = 'models/bge-large-en'
CROSS_ENCODER_NAME = 'models/ms-marco-MiniLM-L6-v2'
LLM_MODEL = os.getenv('LLM_MODEL', 'mistral:7b-instruct')
OLLAMA_BASE_URL = os.getenv('OLLAMA_HOST', 'http://127.0.0.1:11434')
MAX_CHUNK_SIZE = 1500
OVERLAP_BLOCKS = 2
TOP_K_RETRIEVAL = 50
TOP_K_RERANK = 5
TOP_K_FILE_SELECT = int(os.getenv('TOP_K_FILE_SELECT', 3))
THRESHOLD_FILE_SELECT = float(os.getenv('THRESHOLD_FILE_SELECT', 0.78))
THRESHOLD_CHUNKS_RETRIEVE = float(os.getenv('THRESHOLD_CHUNKS_RETRIEVE', 0.7))
HELP_LINK = f'<br><a href="{os.getenv("HELP_LINK", "")}">Try to specify context</a><br>'

NEED_2_REFINE_QUERY = bool(os.getenv('NEED_2_REFINE_QUERY', False))
clean_chunk_markdown = False

MISSING_INFO_TEXT = 'No information'
no_info_in_knowledge_base_message = f'''
The information is missing from the knowledge base.
{HELP_LINK}
We will analyze your request and update our knowledge base.
'''

DEFAULT_MAX_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 100
CLEAN_MARKDOWN_CONTENT = True

USE_OLLAMA_2_SELECT_KNOWLEDGE_BASE = bool(os.getenv('USE_OLLAMA_2_SELECT_KNOWLEDGE_BASE', False))

USE_CHAT_HISTORY_2_SEARCH = False

REFORMAT_ANSWER_USING_LLM = False
