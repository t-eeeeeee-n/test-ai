import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from huggingface_hub import login
import os

def load_huggingface_model(model_name: str):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16,
        offload_state_dict=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=100)
    return HuggingFacePipeline(pipeline=pipe)

def load_embedding_model():
    return HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

def hugging_face_login():
    token = os.getenv("HUGGING_FACE_TOKEN")
    if token:
        login(token)
