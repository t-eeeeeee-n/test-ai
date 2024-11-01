from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

from config import Config


def query_gpt4(prompt: str) -> str:

    documents = SimpleDirectoryReader("data").load_data()

    # set context window
    Settings.context_window = 4096
    # set number of output tokens
    Settings.num_output = 256
    # define LLM
    Settings.llm = OpenAI(
        temperature=0,
        model="gpt-4o-mini",
    )
    Settings.embed_model = OpenAIEmbedding()
    Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # インデックスの作成
    index = VectorStoreIndex.from_documents(documents, embed_model=Settings.embed_model)
    query_engine = index.as_query_engine(llm=Settings.llm)

    # クエリ実行
    response = query_engine.query(prompt)
    return response.response
