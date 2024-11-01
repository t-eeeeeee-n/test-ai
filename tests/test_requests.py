import requests

# GPT-4 エンドポイントのテスト
# gpt4_response = requests.post(
#     "http://127.0.0.1:5000/api/gpt4-query/",
#     json={"question": "現在の日本の総理大臣は誰ですか？"}
# )
# print("GPT-4 Response:", gpt4_response.json())

# Hugging Face エンドポイントのテスト
huggingface_response = requests.post(
    "http://127.0.0.1:5000/api/huggingface-query/",
    json={"question": "AIの仕組みについて説明してください"}
)
print("Hugging Face Response:", huggingface_response.json())
