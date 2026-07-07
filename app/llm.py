import os
import logging
import anthropic
import openai

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

load_dotenv()

provider = os.getenv("LLM_PROVIDER", "openai").lower()

if provider == "anthropic":
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY is not set")
    client = anthropic.Anthropic(api_key=api_key)

elif provider == "openai":
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set")
    client = openai.OpenAI(api_key=api_key)

else:
    raise ValueError(
        f"Unsupported LLM_PROVIDER: '{provider}'. Must be 'anthropic' or 'openai'."
    )


def generate_answer(question: str, matches: list[dict], mode: str) -> str:
    """
    Generate an answer using the configured LLM provider.

    Args:
        question: The user's question.
        matches: Retrieved chunks, each expected to have a 'text' key.
        mode: Response mode — 'evidence' or 'persona'.

    Returns:
        Generated answer as a string.
    """

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
        logger.warning("Unsupported mode received. mode=%s", mode)
        return "Selected mode not supported"

    formatted_matches = "\n\n".join(
        item["text"] for item in matches if "text" in item
    )

    logger.info("Generating answer. provider=%s mode=%s chunks=%d", provider, mode, len(matches))

    if provider == "anthropic":
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1024,
            system=prompt,
            messages=[
                {
                    "role": "user",
                    "content": f"CONTEXT:\n{formatted_matches}\n\nQUESTION:\n{question}"
                }
            ]
        )
        answer = response.content[0].text
        logger.info(
            "Answer generated. stop_reason=%s output_tokens=%d",
            response.stop_reason,
            response.usage.output_tokens,
        )

    elif provider == "openai":
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=1024,
            messages=[
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": f"CONTEXT:\n{formatted_matches}\n\nQUESTION:\n{question}"
                }
            ]
        )
        answer = response.choices[0].message.content
        logger.info(
            "Answer generated. finish_reason=%s output_tokens=%d",
            response.choices[0].finish_reason,
            response.usage.completion_tokens,
        )

    return answer
