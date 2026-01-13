# init_db.py
#
# Description:
# Initialise the Pokédex database by creating tables,
# loading data from CSV, and generating embeddings.

import datetime
from src.hybrid_search.setup_db import Base, engine
from src.hybrid_search.load_data import load_csv
from src.hybrid_search.embeddings import generate_embeddings

DATASET_CSV_PATH = "pokemon-dataset/pokedex.csv"


def initialise_database(verbose: bool = False):
    """
    Initialise the database by creating tables, loading data from CSV,
    and generating embeddings.

    Args:
        verbose: If True, print progress information.
    """
    if verbose:
        print(f"[{datetime.datetime.now()}] Updating database with CSV data...")

    Base.metadata.create_all(bind=engine)
    load_csv(DATASET_CSV_PATH, verbose=verbose)
    generate_embeddings(verbose=verbose)
