# main.py
#
# Description:
# Main script to load data, generate embeddings, and run search queries.

from src.load_data import load_csv
from src.database import Base, engine
from src.embeddings import generate_embeddings
from src.search import search_pokemon

QUERY = "grass pokemon with poison abilities"

def main(demo: bool = True) -> None:
    """
    Load data, generate embeddings, and run search demonstration.
    
    Args:
        demo: If True, runs a demo search after setup.
    """
    # Create database tables
    Base.metadata.create_all(engine)

    # Load data from CSV into the database and generate embeddings
    load_csv("pokemon-dataset/pokedex.csv")
    generate_embeddings()

    # Run a search query to demonstrate functionality
    if demo:
        search_pokemon(QUERY, search_method="keyword")
        search_pokemon(QUERY, search_method="semantic")
        search_pokemon(QUERY, search_method="hybrid")


if __name__ == "__main__":
    main(demo=True)