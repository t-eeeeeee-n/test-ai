from flask import Blueprint, request, jsonify
from app.services.huggingface_service import HuggingFaceService

huggingface_bp = Blueprint("huggingface", __name__)
huggingface_service = HuggingFaceService()

@huggingface_bp.route("/", methods=["POST"])
def query_huggingface():
    data = request.get_json()
    question = data.get("question")
    answer = huggingface_service.process_query(question)
    return jsonify({"response": answer})