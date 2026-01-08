# embeddings.py
#
# Description:
# Generate and store vector embeddings for Pokemon data using SentenceTransformer.

from sentence_transformers import SentenceTransformer

from src.database import SessionLocal
from src.models import Pokemon

model = SentenceTransformer("all-MiniLM-L6-v2")


def generate_embeddings() -> None:
    """
    Generate and store embeddings for all Pokemon in the database.

    Encodes each Pokemon's name, type, and info into a vector embedding
    using the SentenceTransformer model and saves it to the database.
    """
    session = SessionLocal()
    pokemons: list[Pokemon] = session.query(Pokemon).all()

    for pokemon in pokemons:
        text: str = f"{pokemon.name}. Type: {pokemon.type}. {pokemon.info}"
        pokemon.embedding = model.encode(text).tolist()

    session.commit()
    session.close()
