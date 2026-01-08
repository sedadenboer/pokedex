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

from database import SessionLocal
from models import Pokemon

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
rerank_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def keyword_search(query: str, limit: int = 10) -> List[Pokemon]:
    """
    Perform full-text keyword search on the Pokemon info field.

    Args:
        query: Search query string.
        limit: Maximum number of results to return.

    Returns:
        List of Pokemon objects ordered by text relevance.
    """
    session = SessionLocal()

    # Query and info field into tsvector and tsquery for full-text search
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

    session.close()
    return results

def semantic_search(query: str, limit: int = 10) -> List[Pokemon]:
    """
    Perform vector similarity search using embeddings.

    Args:
        query: Search query string.
        limit: Maximum number of results to return.

    Returns:
        List of Pokemon objects ordered by embedding similarity.
    """
    session = SessionLocal()

    query_embedding = embedding_model.encode(query).tolist()

    results = (
        session.query(Pokemon)
        .order_by(Pokemon.embedding.cosine_distance(query_embedding))
        .limit(limit)
        .all()
    )

    session.close()
    return results

def rerank(query: str, result_sets: List[List[Pokemon]]) -> List[Pokemon]:
    """
    Rerank combined keyword and semantic results using a cross-encoder
    on the info field of the Pokémon.

    Args:
        query: Original search query.
        result_sets: List containing keyword and semantic result lists.

    Returns:
        Reranked list of unique Pokemon objects.
    """
    # Combine results and remove duplicates
    combined_results = itertools.chain.from_iterable(result_sets)
    unique_results = {pokemon.id: pokemon for pokemon in combined_results}.values()

    # Rerank combined results based on info field
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

def hybrid_search(query: str, limit: int = 10) -> List[Pokemon]:
    """
    Perform hybrid search combining keyword, semantic search, and reranking.

    Args:
        query: Search query string.
        limit: Maximum number of results to return.

    Returns:
        List of top-ranked Pokemon objects.
    """
    keyword_results = keyword_search(query, limit)
    semantic_results = semantic_search(query, limit)

    reranked_results = rerank(query, [keyword_results, semantic_results])
    return reranked_results[:limit]

def print_results(
    query: str,
    results: List[Pokemon],
    title: str,
) -> None:
    """
    Print search results in a readable, formatted layout.

    Args:
        query: Search query string.
        results: List of Pokemon objects to display.
        title: Title shown above the results.
    """
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


if __name__ == "__main__":
    query = "grass pokemon with poison abilities"

    print_results(
        query,
        keyword_search(query),
        "Keyword Search Results",
    )
    print_results(
        query,
        semantic_search(query),
        "Semantic Search Results",
    )
    print_results(
        query,
        hybrid_search(query),
        "Hybrid Search Results",
    )
