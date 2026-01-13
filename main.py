# main.py
#
# Description:
# Main script to load data, generate embeddings, and run search queries.

import argparse
import datetime

from src.hybrid_search.init_db import initialise_database
from src import pipeline


def args_parser() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Pokémon search system"
    )
    parser.add_argument(
        "--update",
        action="store_true",
        help="Reload the database with data from the CSV file"
    )
    parser.add_argument(
        "--search",
        choices=["keyword", "semantic", "hybrid"],
        help="Search method to use (keyword, semantic, or hybrid)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    return parser.parse_args()
 
def main() -> None:
    """
    Main function to handle database updates and search queries.

    1. Updates the database if --update flag is set.
    2. Runs search queries based on the specified method.
    3. Prints the generated answers.
    """
    args = args_parser()

    # Update database if specified
    if args.update:
        initialise_database(verbose=args.verbose)

    # Run search queries based on specified method
    if args.search:
        # Get query and top_n from user input
        query = input("Enter your search query: ").strip()

        while not query:
            print("Query cannot be empty. Please enter a valid search query.")
            query = input("Enter your search query: ").strip()
            
        top_n = input("Enter number of top results to retrieve (default 5): ").strip()
        top_n = int(top_n) if top_n.isdigit() else 5

        # Run the RAG pipeline
        response = pipeline.pipeline(
            query=query,
            top_n=top_n,
            search_method=args.search,
            verbose=args.verbose
        )
        print("\nGenerated Answer:")
        print(response)


if __name__ == "__main__":
    main()