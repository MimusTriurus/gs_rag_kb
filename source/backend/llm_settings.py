class Settings(dict):
    def to_string(self) -> str:
        return f'''
        LLM_MODEL: {self.LLM_MODEL()}
        TOP_K_RETRIEVAL: {self.TOP_K_RETRIEVAL()}
        TOP_K_RERANK: {self.TOP_K_RERANK()}
        TOP_K_FILE_SELECT: {self.TOP_K_FILE_SELECT()}
        THRESHOLD_FILE_SELECT: {self.THRESHOLD_FILE_SELECT()}
        THRESHOLD_CHUNKS_RETRIEVE: {self.THRESHOLD_CHUNKS_RETRIEVE()}
        NEED_2_REFINE_QUERY_USING_HISTORY: {self.NEED_2_REFINE_QUERY_USING_HISTORY()}
        NEED_2_REFINE_QUERY: {self.NEED_2_REFINE_QUERY()}
        HISTORY_LENGTH: {self.HISTORY_LENGTH()}
        LLM_CREATIVITY: {self.LLM_CREATIVITY()}
        QUESTION_REFINER_CREATIVITY: {self.QUESTION_REFINER_CREATIVITY()}
        '''

    def LLM_MODEL(self) -> str:
        return self.get('LLM_MODEL')

    def TOP_K_RETRIEVAL(self) -> int:
        return int(self.get('TOP_K_RETRIEVAL'))

    def TOP_K_RERANK(self) -> int:
        return int(self.get('TOP_K_RERANK'))

    def TOP_K_FILE_SELECT(self) -> int:
        return int(self.get('TOP_K_FILE_SELECT'))

    def THRESHOLD_FILE_SELECT(self) -> float:
        return float(self.get('THRESHOLD_FILE_SELECT'))

    def THRESHOLD_CHUNKS_RETRIEVE(self) -> float:
        return float(self.get('THRESHOLD_CHUNKS_RETRIEVE'))

    def NEED_2_REFINE_QUERY_USING_HISTORY(self) -> bool:
        return bool(self.get('NEED_2_REFINE_QUERY_USING_HISTORY'))#.lower() == 'on'

    def NEED_2_REFINE_QUERY(self) -> bool:
        return bool(self.get('NEED_2_REFINE_QUERY'))#.lower() == 'on'

    def HISTORY_LENGTH(self) -> int:
        return int(self.get('HISTORY_LENGTH'))

    def QUESTION_REFINER_CREATIVITY(self) -> float:
        return float(self.get('QUESTION_REFINER_CREATIVITY'))

    def LLM_CREATIVITY(self) -> float:
        return float(self.get('LLM_CREATIVITY'))

    def GENERATE_QUESTIONS(self):
        return bool(self.get('GENERATE_QUESTIONS', True))