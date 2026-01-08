# Pokédex Hybrid Search with PostgreSQL and pgvector

This project demonstrates a complete hybrid search pipeline using only
open-source tools. It combines:

- PostgreSQL full-text search (keyword search)
- pgvector (semantic vector search)
- SentenceTransformers (embeddings + reranking)
- SQLAlchemy (ORM)

The example dataset is a Pokémon Pokédex CSV file from: [Pokédex For All 1025 Pokémon (+ text descriptions)](https://www.kaggle.com/datasets/rzgiza/pokdex-for-all-1025-pokemon-w-text-description?)

## Requirements

### System
- [Python](https://www.python.org/downloads/) 3.10+
- [PostgreSQL](https://www.postgresql.org/) 14+
- [pgvector](https://github.com/pgvector/pgvector) extension installed

## Project Structure

```text
.
├── database.py            # Database connection and session setup
├── models.py              # SQLAlchemy models (Pokemon table)
├── load_csv.py            # Load Pokémon CSV into PostgreSQL
├── embeddings.py          # Generate and store vector embeddings
├── search.py              # Keyword, semantic, and hybrid search
├── .env.example           # Example environment variables
├── requirements.txt       # Required packages
└── pokemon-dataset        # Directory containing Pokémon dataset file
    └── pokedex.csv        # Pokémon dataset
```

## Install packages

Create a virtual environment first:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

Install required packages with:

```bash
pip install -r requirements.txt
```

## Database Setup

Create the database and enable pgvector:

```sql
CREATE DATABASE pokedex_db;
\c pokedex_db
CREATE EXTENSION vector;
```

## Run Order

The scripts must be executed in the following order:

```text
1. python load_csv.py       # Load Pokémon data into database
2. python embeddings.py     # Generate and store embeddings
3. python search.py         # Execute search queries
```

### Step 1: Load CSV Data

```bash
python load_csv.py
```
* Reads `pokedex.csv`
* Inserts Pokémon metadata into PostgreSQL
* Does **not** generate embeddings yet

### Step 2: Generate Embeddings

```bash
python embeddings.py
```
* Loads Pokémon descriptions from the database
* Generates sentence embeddings using SentenceTransformers
* Stores embeddings in a `vector` column using pgvector

### Step 3: Run Searches

```bash
python search.py
```
* Keyword search (PostgreSQL full-text search)
* Semantic search (vector similarity)
* Hybrid search with cross-encoder reranking

## Search methods

### Keyword Search
Keyword search utilizes PostgreSQL's full-text search capabilities to find relevant Pokémon based on specific keywords. This method is efficient for retrieving exact matches and is particularly useful for users looking for specific terms.

Example usage:
```python
keyword_search("strange seed")
```

### Semantic Search
Semantic search leverages pgvector to compute vector similarity between the user's query and Pokémon descriptions. This approach allows for more nuanced searches, capturing the meaning behind the words rather than relying solely on exact matches.

Example usage:
```python
semantic_search("grass pokemon with poison abilities")
```

### Hybrid Search

1. Perform keyword search using PostgreSQL full-text search
2. Perform semantic search using pgvector similarity
3. Combine both result sets
4. Remove duplicates
5. Rerank results using a cross-encoder model
6. Return the most relevant documents

Example usage:
```python
hybrid_search("grass pokemon with poison abilities")
```