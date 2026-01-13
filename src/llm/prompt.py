# prompt.py
#
# Description:
# Prompt builder for Pokémon RAG queries.

import datetime
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from src.hybrid_search.models import Pokemon


def process_pokemon(pokemon: "Pokemon", position: int) -> str:
    """
    Processes a Pokémon object into a formatted string for LLM context.

    Args:
        pokemon ("Pokemon"): A Pokémon object with attributes from the database.
        position (int): The index of the Pokémon in the retrieval order (1 = first retrieved).

    Returns:
        str: A formatted string representation of the Pokémon.
    """
    return (
        f"{position}. {pokemon.name}\n"
        f"  id: {pokemon.id}\n"
        f"  height: {pokemon.height}\n"
        f"  weight: {pokemon.weight}\n"
        f"  hp: {pokemon.hp}\n"
        f"  attack: {pokemon.attack}\n"
        f"  defense: {pokemon.defense}\n"
        f"  s_attack: {pokemon.s_attack}\n"
        f"  s_defense: {pokemon.s_defense}\n"
        f"  speed: {pokemon.speed}\n"
        f"  type: {pokemon.type}\n"
        f"  evo_set: {pokemon.evo_set}\n"
        f"  info: {pokemon.info}"
    )

def build_pokemon_context(documents: List["Pokemon"]) -> str:
    """
    Builds a combined context string from a list of Pokémon objects.

    Args:
        documents (List["Pokemon"]): List of Pokémon objects retrieved.

    Returns:
        str: A formatted string representing all Pokémon in the context.
    """
    return "\n\n".join(
        process_pokemon(doc, i + 1) for i, doc in enumerate(documents)
    )

def system_prompt() -> str:
    """
    Returns the system prompt for the LLM instructing it on handling Pokémon RAG queries.

    Returns:
        str: The system prompt string.
    """
    return (
        "You are a RAG assistant for a Pokédex.\n"
        "Use only the Pokémon provided in the context. Do NOT invent any Pokémon, "
        "attributes, or lore.\n"
        "All retrieved Pokémon must be included in the top-N list in the exact "
        "order they are provided in the context (do not change the ranking).\n"
        "After listing all Pokémon, provide a clear, coherent, and concise descriptive "
        "summary synthesizing information from all listed Pokémon.\n"
        "- Include only attributes relevant to the question.\n"
        "- Compare or group Pokémon where appropriate.\n"
        "- Avoid listing every attribute in the summary.\n\n"
        "If the answer cannot be determined from the context, respond exactly:\n"
        "\"The answer is not available in the given context.\""
    )

def pokedex_prompt(query: str, documents: List["Pokemon"], limit: int, verbose: bool = False) -> str:
    """
    Build the full prompt for the LLM, ensuring all retrieved Pokémon are included
    in the ranked list in the exact order, followed by a descriptive summary.

    Args:
        query (str): The user's search query.
        documents (List["Pokemon"]): List of retrieved Pokémon objects.
        limit (int): Number of Pokémon requested by the user.
        verbose (bool): If True, prints debug information.

    Returns:
        str: Complete prompt for the src.llm.
    """
    count = len(documents)

    if count == 0:
        context = "No Pokémon found in the database."
    else:
        context = build_pokemon_context(documents)

    prompt = (
        f"The user searched for Pokémon with the query:\n"
        f"Query: {query}\n"
        f"The user requested {limit} Pokémon results.\n"
        f"Number of Pokémon retrieved: {count}.\n\n"
        "Below are all retrieved Pokémon, listed in the exact order provided by the "
        "retrieval system (do not change the ranking):\n\n"
        f"{context}\n\n"
        "Instruction:\n"
        "- First, present all retrieved Pokémon exactly in the order they appear above.\n"
        "- Then, provide a descriptive, coherent summary synthesizing information "
        "from all listed Pokémon.\n"
        "- Include only attributes relevant to answering the query.\n"
        "- Compare or group Pokémon if appropriate.\n"
        "- Do not invent Pokémon or details not present in the context.\n"
        "- If the answer cannot be determined from the context, respond exactly:\n"
        "\"The answer is not available in the given context.\"\n"
    )

    if verbose:
        print(f"[{datetime.datetime.now()}] Built prompt with {count} Pokémon for query: '{query}'")

    return prompt
