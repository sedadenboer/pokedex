# load_data.py
#
# Description:
# Loads data of pokemon into the database from a CSV file. Also see:
# https://www.kaggle.com/datasets/rzgiza/pokdex-for-all-1025-pokemon-w-text-description?

import datetime
import pandas as pd

from src.hybrid_search.database import SessionLocal
from src.hybrid_search.models import Pokemon


def load_csv(path: str, verbose: bool = False) -> None:
    """
    Load Pokémon data from a CSV file into the database.
    
    Args:
        path: Path to the CSV file containing Pokémon data.
        verbose: If True, print progress information.
    """
    df = pd.read_csv(path)
    session = SessionLocal()

    if verbose:
        print(f"[{datetime.datetime.now()}] Loading {len(df)} Pokémon from {path}...")

    new_pokemon = 0

    for idx, (_, row) in enumerate(df.iterrows(), 1):
        # Check if Pokémon already exists
        existing = session.query(Pokemon).filter(Pokemon.id == int(row["id"])).first()
        if existing:
            continue
        else:
            new_pokemon += 1
            if verbose:
                print(f"  [{datetime.datetime.now()}] [{idx}/{len(df)}] Adding {row['name']}...")
            
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

    if verbose:
        if new_pokemon == 0:
            print(f"[{datetime.datetime.now()}] No new Pokémon to add.")
        else:
            print(f"[{datetime.datetime.now()}] Loaded {new_pokemon} new Pokémon into the database.")