from app.models.openai_util import query_gpt4

class DataOrganizeService:
    @staticmethod
    def process_query(question: str) -> str:
        return query_gpt4(question)