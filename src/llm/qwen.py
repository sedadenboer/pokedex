# qwen.py
#
# Description:
# Scipt to interact with the Qwen language model for RAG-based answering.

import os
import datetime

import ollama
from dotenv import load_dotenv

from llm.prompt import system_prompt

load_dotenv()


def generate_answer(prompt: str, verbose: bool = False) -> str:
    """
    Generate an answer with the LLM based on the provided prompt.

    Args:
        prompt (str): The complete prompt to send to the LLM.
        verbose (bool): If True, prints debug information.

    Returns:
        str: The generated answer from the LLM, or an error message if generation fails.
    """
    model = os.getenv("QWEN_MODEL", "qwen2.5:3b-instruct")

    try:
        if verbose:
            print(f"[{datetime.datetime.now()}] Generating answer using model '{model}'")

        response = ollama.chat(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt(),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            options={"temperature": 0.0},
        )

        answer = response.get("message", {}).get("content", "")
        if not answer:
            raise ValueError("Received empty response from the LLM.")
        else:
            if verbose:
                print(f"[{datetime.datetime.now()}] Successfully generated answer from LLM!\n")

        return answer
    except Exception as e:
        error_msg = f"Error generating answer with Qwen model: {e}"
        if verbose:
            print(f"[{datetime.datetime.now()}] {error_msg}")
        return error_msg
        
