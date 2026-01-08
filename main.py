# main.py
#
# Description:
# Main script to load data, generate embeddings, and run search queries.

import argparse

from src.database import Base, engine
from src.embeddings import generate_embeddings
from src.load_data import load_csv
from src.search import search_pokemon

QUERY = "grass pokemon with poison abilities"


def main(search_method: str = None, verbose: bool = False) -> None:
    """
    Load data, generate embeddings, and run search demonstration.

    Args:
        search_method: Search method to use ('keyword', 'semantic', 'hybrid', or 'all').
                      If None, skips search demonstration.
        verbose: Enable verbose output.
    """
    # Create database tables
    Base.metadata.create_all(engine)

    # Load data from CSV into the database and generate embeddings
    load_csv("pokemon-dataset/pokedex.csv", verbose=verbose)
    generate_embeddings(verbose=verbose)

    # Run search queries based on specified method
    if search_method:
        # Get query from user
        query = input("Enter your search query: ").strip()
        if not query:
            query = QUERY

        if search_method == "all":
            search_pokemon(query, search_method="keyword", verbose=verbose)
            search_pokemon(query, search_method="semantic", verbose=verbose)
            search_pokemon(query, search_method="hybrid", verbose=verbose)
        else:
            search_pokemon(query, search_method=search_method, verbose=verbose)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pokémon search system"
    )
    parser.add_argument(
        "--search",
        choices=["keyword", "semantic", "hybrid", "all"],
        help="Search method to use (keyword, semantic, hybrid, or all)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    args = parser.parse_args()

    main(search_method=args.search, verbose=args.verbose)