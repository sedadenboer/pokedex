# load_data.py
#
# Description:
# Loads data of pokemon into the database from a CSV file. Also see:
# https://www.kaggle.com/datasets/rzgiza/pokdex-for-all-1025-pokemon-w-text-description?

from src.database import SessionLocal
from src.models import Pokemon

import pandas as pd


def load_csv(path: str) -> None:
    """
    Load Pokemon data from a CSV file into the database.
    
    Args:
        path: Path to the CSV file containing Pokemon data.
    """
    df = pd.read_csv(path)
    session = SessionLocal()

    for _, row in df.iterrows():
        # Check if Pokemon already exists
        existing = session.query(Pokemon).filter(Pokemon.id == int(row["id"])).first()
        if existing:
            continue
        
        pokemon = Pokemon(
            id=int(row["id"]),
            name=row["name"],
            height=int(row["height"]),
            weight=int(row["weight"]),
            hp=int(row["hp"]),
            attack=int(row["attack"]),
            defense=int(row["defense"]),
            s_attack=int(row["s_attack"]),
            s_defense=int(row["s_defense"]),
            speed=int(row["speed"]),
            type=row["type"].strip("{}"),
            evo_set=int(row["evo_set"]),
            info=row["info"],
        )
        session.add(pokemon)

    session.commit()
    session.close()
