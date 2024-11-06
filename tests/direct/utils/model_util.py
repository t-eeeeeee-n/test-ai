import os

import torch
from dotenv import load_dotenv
from langchain_huggingface import HuggingFacePipeline
from llama_index.core import SimpleDirectoryReader, Settings, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


# Hugging FaceのLLMの初期化
def load_huggingface_model(model_name: str):
    # モデルのロード
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16,
        offload_state_dict=True
    )
    # トークナイザーのロード
    tokenizer = AutoTokenizer.from_pretrained(model_name, legacy=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    # パイプラインの作成
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=100)
    return HuggingFacePipeline(pipeline=pipe)

# GPT-4を使ったクエリ処理の関数
def query_gpt4(prompt):
    api_key  = os.getenv("OPENAI_API_KEY")
    # OpenAIのクライアントをインスタンス化
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": "You are an assistant for organizing data."},
                  {"role": "user", "content": prompt}],
        max_tokens=300
    )
    return response.choices[0].message.content


# Hugging FaceまたはGPT-4を用いてインデックスを作成し、クエリを実行する
def create_and_query_index(model_type="huggingface", model_name="gpt2", query=""):
    # ドキュメントの読み込み
    documents = SimpleDirectoryReader("data").load_data()

    if model_type == "huggingface":
        # Hugging Faceモデルを読み込む
        llm = load_huggingface_model(model_name)

        Settings.llm = llm
        Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
        Settings.pad_token_id = AutoTokenizer.from_pretrained(model_name).eos_token_id
    elif model_type == "gpt-4":
        load_dotenv()

        # GPT-4のクエリのみを使用する（埋め込みやインデックス操作は省略）
        # return query_gpt4(query)

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
    else:
        raise ValueError("model_typeは 'huggingface' または 'gpt-4' である必要があります")

    # インデックスの作成
    index = VectorStoreIndex.from_documents(documents, embed_model=Settings.embed_model)
    query_engine = index.as_query_engine(llm=Settings.llm)

    # クエリ実行
    response = query_engine.query(query)
    return response.response