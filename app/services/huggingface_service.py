from app.models.model_util import create_and_query_index

class HuggingFaceService:
    @staticmethod
    def process_query(question: str) -> str:
        return create_and_query_index(query=question)