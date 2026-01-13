# search.py
#
# Description:
# Implements keyword, semantic, and hybrid search with cross-encoder reranking
# for querying Pokémon embeddings and metadata from the database.

from typing import List
import datetime
import itertools
import textwrap

from sqlalchemy import func
from sentence_transformers import CrossEncoder, SentenceTransformer

from src.hybrid_search.database import SessionLocal
from src.hybrid_search.models import Pokemon


def keyword_search(query: str, limit: int = 5, verbose: bool = False) -> List[Pokemon]:
    """
    Perform full-text keyword search on the Pokemon info field.

    Args:
        query: Search query string.
        limit: Maximum number of results to return.
        verbose: Print debug information.

    Returns:
        List of Pokemon objects ordered by text relevance.
    """
    session = SessionLocal()

    ts_query = func.plainto_tsquery("english", query)
    ts_vector = func.to_tsvector("english", Pokemon.info)
    rank = func.ts_rank_cd(ts_vector, ts_query)

    results = (
        session.query(Pokemon)
        .filter(ts_vector.op("@@")(ts_query))
        .order_by(rank.desc())
        .limit(limit)
        .all()
    )

    if verbose:
        print(f"[{datetime.datetime.now()}] Keyword search found {len(results)} results")

    session.close()
    return results

def semantic_search(query: str, limit: int = 5, verbose: bool = False) -> List[Pokemon]:
    """
    Perform vector similarity search using embeddings.

    Args:
        query: Search query string.
        limit: Maximum number of results to return.
        verbose: Print debug information.

    Returns:
        List of Pokemon objects ordered by embedding similarity.
    """
    session = SessionLocal()
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    query_embedding = embedding_model.encode(query).tolist()

    results = (
        session.query(Pokemon)
        .order_by(Pokemon.embedding.cosine_distance(query_embedding))
        .limit(limit)
        .all()
    )

    if verbose:
        print(f"[{datetime.datetime.now()}] Semantic search found {len(results)} results")

    session.close()
    return results

def rerank(query: str, result_sets: List[List[Pokemon]], verbose: bool = False) -> List[Pokemon]:
    """
    Rerank combined keyword and semantic results using a cross-encoder.

    Args:
        query: Original search query.
        result_sets: List containing keyword and semantic result lists.
        verbose: Print debug information.

    Returns:
        Reranked list of unique Pokemon objects.
    """
    combined_results = itertools.chain.from_iterable(result_sets)
    unique_results = {pokemon.id: pokemon for pokemon in combined_results}.values()

    if verbose:
        print(f"[{datetime.datetime.now()}] Reranking {len(unique_results)} results")

    rerank_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    scores = rerank_model.predict(
        [(query, pokemon.info) for pokemon in unique_results]
    )

    reranked = [
        pokemon
        for _, pokemon in sorted(
            zip(scores, unique_results), reverse=True
        )
    ]

    return reranked

def hybrid_search(query: str, limit: int = 5, verbose: bool = False) -> List[Pokemon]:
    """
    Perform hybrid search combining keyword, semantic search, and reranking.

    Args:
        query: Search query string.
        limit: Maximum number of results to return.
        verbose: Print debug information.

    Returns:
        List of top-ranked Pokemon objects.
    """
    if verbose:
        print(f"[{datetime.datetime.now()}] Performing hybrid search")

    keyword_results = keyword_search(query, limit, verbose)
    semantic_results = semantic_search(query, limit, verbose)

    reranked_results = rerank(query, [keyword_results, semantic_results], verbose)
    return reranked_results[:limit]

def search_pokemon(
    query: str,
    limit: int = 5,
    search_method: str = "hybrid",
    verbose: bool = False,
) -> List[Pokemon]:
    """
    Search for Pokemon using the specified method and print results.

    Args:
        query: Search query string.
        search_method: Either "keyword", "semantic", or "hybrid".
        verbose: Print debug information.
    """
    results: List[Pokemon] = []
    title: str = ""

    if search_method == "keyword":
        results = keyword_search(query, limit=limit, verbose=verbose)
        title = "Keyword Search Results"
    elif search_method == "semantic":
        results = semantic_search(query, limit=limit, verbose=verbose)
        title = "Semantic Search Results"
    elif search_method == "hybrid":
        results = hybrid_search(query, limit=limit, verbose=verbose)
        title = "Hybrid Search Results"
    else:
        raise ValueError(
            f"Invalid search method: {search_method}. "
            f"Choose 'keyword', 'semantic', or 'hybrid'."
        )

    if verbose:
        print(f"[{datetime.datetime.now()}] Displaying "
            f"{len(results)} results for '{search_method}' search")
        print("\n" + "=" * 80)
        print(title.center(80))
        print("=" * 80)
        print(f"QUERY: '{query}'\n")

        if not results:
            print("No results found.")
            return

        for index, pokemon in enumerate(results, start=1):
            print(f"{index}. {pokemon.name} (Type: {pokemon.type})")
            print("   Info:")

            wrapped_info = textwrap.fill(
                pokemon.info,
                width=80,
                initial_indent="      ",
                subsequent_indent="      "
            )
            print(wrapped_info)
            print()

        print("=" * 80 + "\n")
    
    return results