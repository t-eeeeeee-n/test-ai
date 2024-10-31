import torch
import utils
from langchain_huggingface import HuggingFacePipeline
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.langchain import LangChainLLM
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# モデルの準備
models = utils.load_models_from_config('config.yaml')
purpose = "gpt-neo"  # モデルの目的
model_name = utils.select_model_by_purpose(models, purpose)
# モデルのロード (効率的なロードのためにfloat16とstate_dictのオフロードを有効に)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16,
    offload_state_dict=True
)
# トークナイザーのロード
tokenizer = AutoTokenizer.from_pretrained(model_name, legacy=False)

# パイプラインの作成 (Hugging FaceをLangChainのLLMとして使用)
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=100)
llm = HuggingFacePipeline(pipeline=pipe)

# ローカル埋め込みモデルの指定
embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 非構造化データを読み込む
documents = SimpleDirectoryReader("data").load_data()

# Settingsの使用
Settings.llm = llm
Settings.embed_model = embed_model
Settings.pad_token_id = tokenizer.eos_token_id

# インデックスの作成
index = VectorStoreIndex.from_documents(documents, embed_model=Settings.embed_model)

# クエリエンジンの作成
query_engine = index.as_query_engine(llm=llm)

# クエリ実行
query = """
以下のデータを次のフォーマットに従って整理してください:

フォーマット:
日付: [ここに日付]
タイトル: [ここにタイトル]
内容: [ここに内容]

データを整理してください。
"""
response = query_engine.query(query)

# 結果の表示
print(response.response)