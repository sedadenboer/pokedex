# setup_db.py
#
# Description:
# Database engine and session configuration for SQLAlchemy ORM.

import os

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, declarative_base, sessionmaker

load_dotenv()

# Create a database engine which the Session will use for connection resources
DATABASE_URL: str | None = os.getenv("DATABASE_URL")
engine: Engine | None = None
try:
    engine = create_engine(DATABASE_URL)
except Exception as e:
    print("Unable to access postgresql database", repr(e))

SessionLocal: sessionmaker[Session] = sessionmaker(bind=engine)
Base = declarative_base()