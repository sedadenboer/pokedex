# pipeline.py
#
# Description:
# Pokédex RAG pipeline to load data, generate embeddings, run searches,
# and generate answers using the Qwen src.llm.

import datetime

from src.hybrid_search.search import search_pokemon
from src.llm.prompt import pokedex_prompt
from src.llm.qwen import generate_answer


def pipeline(
    query: str = "flower Pokémon",
    top_n: int = 5,
    search_method: str = 'hybrid',
    verbose: bool = False
) -> None:
    """
    RAG pipeline to process a query and generate an answer for searching Pokémon
    in the Pokédex database. 

    1. Retrieves top N Pokémon results using the specified search method.
    2. Builds the prompt and generates an answer using the Qwen src.llm.

    Args:
        query: Search query string.
        top_n: Number of top results to retrieve.
        search_method: Search method to use ('keyword', 'semantic', 'hybrid').
        verbose: Enable verbose output.
    """
    # Retrieve top N results and generate answer
    if verbose:
        print(f"[{datetime.datetime.now()}] Searching for '{query}' (top {top_n}) using {search_method} method...")
        
    documents = search_pokemon(query, limit=top_n, search_method=search_method, verbose=verbose)
    prompt = pokedex_prompt(query, documents, limit=top_n, verbose=verbose)
    answer = generate_answer(prompt, verbose=verbose)

    return answer