# models.py
#
# Description:
# This script defines the SQLAlchemy ORM model for storing Pokemon data,
# including their attributes and vector embeddings for similarity search.

from sqlalchemy import Column, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase

from pgvector.sqlalchemy import Vector

from src.database import Base


class Pokemon(Base):
    """SQLAlchemy ORM model for Pokemon data with embeddings.
    
    Stores Pokemon attributes and vector embeddings for similarity search.
    """
    __tablename__: str = "pokemon"

    id: int = Column(Integer, primary_key=True)
    name: str = Column(String)
    height: int = Column(Integer)
    weight: int = Column(Integer)
    hp: int = Column(Integer)
    attack: int = Column(Integer)
    defense: int = Column(Integer)
    s_attack: int = Column(Integer)
    s_defense: int = Column(Integer)
    speed: int = Column(Integer)
    type: str = Column(String)
    evo_set: int = Column(Integer)
    info: str = Column(Text)
    embedding: list = Column(Vector(384))
