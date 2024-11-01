from flask import Blueprint, request, jsonify
from app.services.gpt4_service import Gpt4Service

gpt4_bp = Blueprint("gpt4", __name__)
gpt4_service = Gpt4Service()

@gpt4_bp.route("/", methods=["POST"])
def query_gpt4():
    data = request.get_json()
    question = data.get("question")
    answer = gpt4_service.process_query(question)
    return jsonify({"response": answer})