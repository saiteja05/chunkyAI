import os
 # Use environment variables for production
class Config:
    SECRET_KEY="SOME_SECRET"
    ADOBE_SECRET_KEY ="SOME_SECRET" 
    ADOBE_CLIENT_ID="SOME_SECRET"
    UPLOAD_FOLDER = './uploads'
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # Limit uploads to 16 MB
    MongoDB_URI ="mongo srv"
    DB_NAME="ChunkyAI"
    APP_NAME = "ChunkyAI"
    OPEN_AI_KEY="SOME_SECRET"
    FIREWORKS_API_KEY="SOME_SECRET"
    LANGCHAIN_TRACING_V2=True
    LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
    LANGCHAIN_API_KEY="lSOME_SECRET"
    LANGCHAIN_PROJECT="CHUNKYAI"
    # Ollama API endpoint and model (adjust this based on your setup)
    OLLAMA_API_URL = "http://localhost:11434/v1/ask"  # Example URL, update it if needed
    OLLAMA_MODEL = "llama3"  # The model you want to use (change this if Ollama has specific models)

# TO USE CONFIG VARIABLE
    # from config import Config
    # app.config.from_object(Config)
#     @app.route('/')
# def home():
#     secret = app.config['SECRET_KEY']
#     return f"Your secret key is: {secret}"