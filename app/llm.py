import os
from dotenv import load_dotenv

# Carga las variables del archivo .env
load_dotenv()

# Accede a la API Key
api_key = os.getenv('OPENAI_API_KEY')
print(api_key)
#def generate_answer(question: str, context: str, mode: str):
