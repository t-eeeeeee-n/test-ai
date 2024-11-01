import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    HUGGING_FACE_TOKEN = os.getenv("HUGGING_FACE_TOKEN")