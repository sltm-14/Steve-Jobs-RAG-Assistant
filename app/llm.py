import os
from dotenv import load_dotenv
from openai import OpenAI

# Carga las variables del archivo .env
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY is not set")

# Accede a la API Key
client = OpenAI(
    api_key = os.getenv('OPENAI_API_KEY')
    
)

def generate_answer(question: str, matches: list[dict], mode: str) -> str:

    if mode == "evidence":
        prompt = """
                You are a source-grounded assistant.
                Answer the user's question using ONLY the provided context.
                If the answer is not supported by the context, say:
                "I don't have enough evidence in the provided context to answer that."
                Be concise and factual.
                Respond in the same language as the user's question.
                """

    elif mode == "persona":
        prompt = """
                You are a source-grounded assistant.
                Answer using ONLY the provided context.
                Respond in a Steve Jobs-inspired tone: clear, elegant, focused, concise.
                Do not claim to be Steve Jobs.
                Do not invent facts.
                Respond in the same language as the user's question.
                """
    else:
        return "Selected mode not supported"

    # Converts the matches in a string
    formatted_matches = "\n\n".join(
        item["text"] for item in matches if "text" in item
    )

    response = client.responses.create(
        model="gpt-5.4-mini",
        input=[
            {
                "role": "system",
                "content": prompt
            },

            {
                "role": "system",
                "content": f"CONTEXT:\n{formatted_matches}"
            },

            {
                "role": "user",
                "content": question
            }
        ]
    )

    return response.output_text