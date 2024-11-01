from flask import Flask
from app.routes.gpt4_routes import gpt4_bp
from app.routes.huggingface_routes import huggingface_bp
from dotenv import load_dotenv

load_dotenv()

def create_app():
    app = Flask(__name__)
    app.config.from_object("config.Config")
    # エンドポイントの登録
    app.register_blueprint(gpt4_bp, url_prefix="/api/gpt4-query")
    app.register_blueprint(huggingface_bp, url_prefix="/api/huggingface-query")
    return app

if __name__ == "__main__":
    app = create_app()
    app.run(debug=True)