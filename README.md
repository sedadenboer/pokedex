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
├── src/
│   ├── database.py        # Database connection and session setup
│   ├── models.py          # SQLAlchemy models (Pokemon table)
│   ├── load_csv.py        # Load Pokémon CSV into PostgreSQL
│   ├── embeddings.py      # Generate and store vector embeddings
│   └── search.py          # Keyword, semantic, and hybrid search
├── main.py                # Entry point for all operations
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

## Quick Start

1. Ensure the database is available and dependencies are installed.
2. Run the script with a search method. The search query is **requested interactively** (or falls back to a default).

```bash
python main.py --search [search_method]
```

Available search methods:

* `keyword`
* `semantic`
* `hybrid`
* `all`

Optional verbose logging:

```bash
python main.py --search hybrid --verbose
```

After starting:

* Database tables are created
* The CSV file (`pokemon-dataset/pokedex.csv`) is loaded
* Embeddings are generated
* You are prompted for a search query
  (press Enter to use the default query)

## Search methods

### Keyword Search

Uses PostgreSQL full-text search.
Best for exact terms and fast filtering.

### Semantic Search

Uses pgvector for vector similarity on descriptions.
Best for meaning-based queries.

### Hybrid Search

Combination of keyword and semantic search. Uses a reranker to generate the final search results.

Pipeline:
1. Keyword search (PostgreSQL FTS)
2. Semantic search (pgvector)
3. Combine result sets
4. Remove duplicates
5. Rerank using a cross-encoder model
6. Return the most relevant results


## Examples

Run all search methods sequentially:

```bash
python main.py --search all --verbose
```

Only load data and generate embeddings (no search):

```bash
python main.py --verbose
```