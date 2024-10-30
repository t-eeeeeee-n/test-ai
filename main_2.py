import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import utils

# Hugging Faceログイン
utils.hugging_face_login()

# モデルを選択
models = utils.load_models_from_config('config.yaml')
purpose = "general-purpose"  # モデルの目的
model_name = utils.select_model_by_purpose(models, purpose)

if model_name:
    try:
        # モデルのロード (効率的なロードのためにfloat16とstate_dictのオフロードを有効に)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16,
            offload_state_dict=True
        )

        # トークナイザーのロード
        tokenizer = AutoTokenizer.from_pretrained(model_name, legacy=False)

        # パディングトークンがない場合の対応
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        # 入力プロンプト
        input_text = "Who are you?"

        # トークン化 (入力をパディングとトランケート)
        inputs = tokenizer(
            input_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=100,
            return_attention_mask=True
        )

        # モデルでテキスト生成
        output = model.generate(
            inputs["input_ids"],
            attention_mask=inputs['attention_mask'],
            max_new_tokens=50,  # 生成トークン数を50に
            num_return_sequences=1,
            do_sample=True,  # サンプリングを有効化
            top_k=50,  # 上位50の単語から選択
            top_p=0.9,  # 確率上位の単語で制限
            repetition_penalty=1.2,
            no_repeat_ngram_size=2,
            pad_token_id=tokenizer.eos_token_id
        )

        # 応答のデコード
        response = tokenizer.decode(output[0], skip_special_tokens=True)
        print(response)

    except Exception as e:
        print(f"モデルのロードや生成時にエラーが発生しました: {e}")
else:
    print(f"目的 '{purpose}' に適したモデルが見つかりませんでした。")
