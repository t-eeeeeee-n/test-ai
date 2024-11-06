from tests.direct.utils import model_util

# クエリ実行
query = """
以下のデータを次のフォーマットに従って整理してください。条件に従って必要に応じて情報を省略し、不要な記号や内容は含めないようにしてください。

フォーマット:
日付: [YYYY/MM/DDの形式で日付を記載]
タイトル: [ここにタイトルを記載]
内容: [ここに内容を記載、必要がない場合は空白にする]

条件:
1. 日付は必ずYYYY/MM/DDの形式にしてください。
2. タイトルが見当たらない場合、内容を要約してタイトルの代わりとして使用してください。
3. タイトルで十分に伝わる内容であれば、内容は省略し空白にしてください。
4. 内容が必要な場合、箇条書きや不要な記号（例えば「-」など）は使用せず、文章のみで記載してください。
5. 内容がない場合、そのデータは出力しなくても構いません。

以下のデータを整理してください。
"""

# 使用するモデルを指定してクエリを実行
# "huggingface" または "gpt-4" を指定して切り替え可能
response_huggingface = model_util.create_and_query_index(model_type="huggingface", model_name="gpt2", query=query)
print("Hugging Face Model Response:\n", response_huggingface)

# response_gpt4 = model_util.create_and_query_index(model_type="gpt-4", query=query)
# print("GPT-4 Model Response:\n", response_gpt4)