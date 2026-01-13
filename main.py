# main.py
#
# Description:
# Main script to load data, generate embeddings, and run search queries.

import argparse
import datetime

from src import pipeline


def main() -> None:
    """
    #TODO: Add function description
    """
    # Parse command-line arguments
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
    args = parser.parse_args()
    
    # Update database if specified
    if args.update:
        if args.verbose:
            print(f"[{datetime.datetime.now()}] Updating database with CSV data...")
            
        # Create database tables
        Base.metadata.create_all(engine)

        # Load data from CSV into the database and generate embeddings
        load_csv("pokemon-dataset/pokedex.csv", verbose=args.verbose)
        generate_embeddings(verbose=args.verbose)

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
            update_db=args.update,
            search_method=args.search,
            verbose=args.verbose
        )
        print("\nGenerated Answer:")
        print(response)


if __name__ == "__main__":
    main()