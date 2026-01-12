# pipeline.py
#
# Description:
# Pokédex RAG pipeline to load data, generate embeddings, run searches,
# and generate answers using the Qwen LLM.

import datetime

from hybrid_search.search import search_pokemon
from llm.prompt import pokedex_prompt
from llm.qwen import generate_answer


def pipeline(
    query: str = "flower Pokémon",
    top_n: int = 5,
    update_db: bool = False,
    search_method: str = 'hybrid',
    verbose: bool = False
) -> None:
    """
    RAG pipeline to process a query and generate an answer for searching Pokémon
    in the Pokédex database. 

    1. Optionally loads/updates the database with CSV data.
    2. Retrieves top N Pokémon results using the specified search method.
    3. Builds the prompt and generates an answer using the Qwen LLM.

    Args:
        query: Search query string.
        top_n: Number of top results to retrieve.
        update_db: If True, reloads the database with CSV data.
        search_method: Search method to use ('keyword', 'semantic', 'hybrid').
        verbose: Enable verbose output.
    """
    if update_db:
        if verbose:
            print(f"[{datetime.datetime.now()}] Updating database with CSV data...")
            
        # Create database tables
        Base.metadata.create_all(engine)

        # Load data from CSV into the database and generate embeddings
        load_csv("pokemon-dataset/pokedex.csv", verbose=verbose)
        generate_embeddings(verbose=verbose)

    # Retrieve top N results and generate answer
    if verbose:
        print(f"[{datetime.datetime.now()}] Searching for {query} (top {top_n}) using '{search_method}' method...")
        
    documents = search_pokemon(query, limit=top_n, search_method=search_method, verbose=verbose)
    prompt = pokedex_prompt(query, documents, limit=top_n, verbose=verbose)
    answer = generate_answer(prompt, verbose=verbose)

    return answer


if __name__ == "__main__":
    # Example usage of the pipeline
    response = pipeline(
        query="fire type Pokémon with high attack stats",
        top_n=5,
        update_db=False,
        search_method="hybrid",
        verbose=True
    )
    print(f"Generated Answer:\n{response}")