import os
from dotenv import load_dotenv
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer

# .envファイルの読み込み
load_dotenv()

# 環境変数からHugging Faceのトークンを取得
hugging_face_token = os.getenv('HUGGING_FACE_TOKEN')

# ログイン処理
if hugging_face_token:
    login(hugging_face_token)

# 日本語に対応したGPTモデル（例えばrinnaのモデルを使用）
model_name = "rinna/japanese-gpt2-medium"
model = AutoModelForCausalLM.from_pretrained(model_name)

# use_fast=False を指定して遅いトークナイザーを使う
tokenizer = AutoTokenizer.from_pretrained(model_name, legacy=False)

# より具体的なプロンプト
input_text = "あなたはだれですか？簡潔に「私は○○です」と一文で答えてください。"

# トークン化
inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=100, return_attention_mask=True)

# モデルが生成する応答を取得
output = model.generate(
    inputs["input_ids"],
    attention_mask=inputs['attention_mask'],
    max_new_tokens=5,  # さらに短い応答を強制
    num_return_sequences=1,
    do_sample=False,  # 決定論的な応答
    repetition_penalty=1.2,
    no_repeat_ngram_size=2,
    pad_token_id=tokenizer.pad_token_id
)

# トークンをデコードして日本語を表示
response = tokenizer.decode(output[0], skip_special_tokens=True)
print(response)
