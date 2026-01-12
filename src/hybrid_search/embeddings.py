# embeddings.py
#
# Description:
# Generate and store vector embeddings for Pokemon data
# using SentenceTransformer.

import datetime
from sentence_transformers import SentenceTransformer

from hybrid_search.database import SessionLocal
from hybrid_search.models import Pokemon


def generate_embeddings(verbose: bool = True) -> None:
    """
    Generate and store embeddings for all Pokemon in the database.

    Encodes each Pokemon's name, type, and info into a vector embedding
    using the SentenceTransformer model and saves it to the database.
    
    Args:
        verbose: If True, print progress messages.
    """
    session = SessionLocal()
    model = SentenceTransformer("all-MiniLM-L6-v2")
    pokemons: list[Pokemon] = session.query(Pokemon).filter(
        Pokemon.embedding == None
        ).all()  # noqa: E711
    
    if verbose:
        if not pokemons:
            print(f"[{datetime.datetime.now()}] No new Pokémon to generate embeddings for.")
            session.close()
            return

        print(f"[{datetime.datetime.now()}] Generating embeddings for {len(pokemons)} Pokémon...")

    for i, pokemon in enumerate(pokemons):
        text: str = f"{pokemon.name}. Type: {pokemon.type}. {pokemon.info}"
        pokemon.embedding = model.encode(text).tolist()
        
        if verbose and (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(pokemons)} Pokemon")

    session.commit()
    session.close()
    
    if verbose:
        print(f"[{datetime.datetime.now()}] Embeddings generation complete!")
