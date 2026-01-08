# search.py
#
# Description:
# Implements keyword, semantic, and hybrid search with cross-encoder reranking
# for querying Pokemon embeddings and metadata from the database.

from typing import List
import itertools
import textwrap

from sqlalchemy import func
from sentence_transformers import CrossEncoder, SentenceTransformer

from src.database import SessionLocal
from src.models import Pokemon

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
rerank_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def keyword_search(query: str, limit: int = 10, verbose: bool = False) -> List[Pokemon]:
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
        print(f"[keyword_search] Found {len(results)} results")

    session.close()
    return results

def semantic_search(query: str, limit: int = 10, verbose: bool = False) -> List[Pokemon]:
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

    query_embedding = embedding_model.encode(query).tolist()
    if verbose:
        print(f"[semantic_search] Query embedding shape: {len(query_embedding)}")

    results = (
        session.query(Pokemon)
        .order_by(Pokemon.embedding.cosine_distance(query_embedding))
        .limit(limit)
        .all()
    )

    if verbose:
        print(f"[semantic_search] Found {len(results)} results")

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
        print(f"[rerank] Reranking {len(unique_results)} unique results")

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

def hybrid_search(query: str, limit: int = 10, verbose: bool = False) -> List[Pokemon]:
    """
    Perform hybrid search combining keyword, semantic search, and reranking.

    Args:
        query: Search query string.
        limit: Maximum number of results to return.
        verbose: Print debug information.

    Returns:
        List of top-ranked Pokemon objects.
    """
    keyword_results = keyword_search(query, limit, verbose)
    semantic_results = semantic_search(query, limit, verbose)

    reranked_results = rerank(query, [keyword_results, semantic_results], verbose)
    return reranked_results[:limit]

def search_pokemon(
    query: str,
    search_method: str,
    verbose: bool = False,
) -> None:
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
        results = keyword_search(query, verbose=verbose)
        title = "Keyword Search Results"
    elif search_method == "semantic":
        results = semantic_search(query, verbose=verbose)
        title = "Semantic Search Results"
    elif search_method == "hybrid":
        results = hybrid_search(query, verbose=verbose)
        title = "Hybrid Search Results"
    else:
        raise ValueError(
            f"Invalid search method: {search_method}. "
            f"Choose 'keyword', 'semantic', or 'hybrid'."
        )

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
        print(f"      {pokemon.info}")
        print()

    print("=" * 80 + "\n")


if __name__ == "__main__":
    query = "grass pokemon with poison abilities"
    search_method = "hybrid"  # Options: "keyword", "semantic", "hybrid"
    search_pokemon(query, search_method)
