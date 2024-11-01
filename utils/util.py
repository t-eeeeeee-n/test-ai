import yaml
import os
from dotenv import load_dotenv
from huggingface_hub import login

# YAMLファイルを読み込んでモデルリストを取得
def load_models_from_config(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
        return config['models']

# モデルの用途に応じて切り替える関数
def select_model_by_purpose(models, purpose):
    for model in models:
        if model['purpose'] == purpose:
            return model['name']
    return None  # 見つからなかった場合

# モデルの用途に応じてローカルパスを取得する関数
def get_local_path_by_purpose(models, purpose):
    for model in models:
        if model['purpose'] == purpose:
            return model.get('local_path', None)  # ローカルパスがあれば返す
    return None

def hugging_face_login():
    # .envファイルの読み込み
    load_dotenv()

    # 環境変数からHugging Faceのトークンを取得
    hugging_face_token = os.getenv('HUGGING_FACE_TOKEN')

    # ログイン処理
    if hugging_face_token:
        login(hugging_face_token)
