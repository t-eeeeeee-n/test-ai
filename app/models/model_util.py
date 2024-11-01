from app.utils.helpers import load_huggingface_model, load_embedding_model
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings

def create_and_query_index(model_name="gpt2", query=""):
    documents = SimpleDirectoryReader("data").load_data()
    llm = load_huggingface_model(model_name)
    Settings.llm = llm
    Settings.embed_model = load_embedding_model()

    index = VectorStoreIndex.from_documents(documents, embed_model=Settings.embed_model)
    query_engine = index.as_query_engine(llm=Settings.llm)
    response = query_engine.query(query)
    return response.response