# Pokédex RAG module

This project demonstrates a fully open-source Pokémon search system combining hybrid retrieval and LLM-based generation. It integrates:

- PostgreSQL full-text search (keyword search)
- pgvector (semantic vector search)
- SentenceTransformers (embeddings + reranking)
- SQLAlchemy (ORM)
- Ollama with Qwen 2.5 3B Instruct (local LLM)

The example dataset is a Pokémon Pokédex CSV file from: [Pokédex For All 1025 Pokémon (+ text descriptions)](https://www.kaggle.com/datasets/rzgiza/pokdex-for-all-1025-pokemon-w-text-description?)

## Requirements

### System
- [Python](https://www.python.org/downloads/) 3.10+
- [PostgreSQL](https://www.postgresql.org/) 14+
- [pgvector](https://github.com/pgvector/pgvector) extension installed
- [Ollama](https://ollama.com/) with [Qwen 2.5 3B Instruct](https://ollama.com/library/qwen2.5:3b)

## Project Structure

```text
.
├── pokemon-dataset/
│   └── pokedex.csv
├── src/
│   ├── hybrid_search/
│   │   ├── setup_db.py        # database setup
│   │   ├── init_db.py         # database initialisation
│   │   ├── models.py          # SQLAlchemy ORM models
│   │   ├── load_data.py       # CSV → database
│   │   ├── embeddings.py      # embedding generation
│   │   └── search.py          # keyword / semantic / hybrid search
│   └── llm/
│       ├── prompt.py          # prompt templates
│       └── qwen.py            # Ollama Qwen wrapper
├── main.py                    # Entry point
├── .env.example
└── requirements.txt

```

## Quick Start

### Install packages

Create a virtual environment first:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

Install required packages with:

```bash
pip install -r requirements.txt
```

Make sure you set the `.env` variables afterwards!

### Database setup

Create the database and enable pgvector:

```sql
CREATE DATABASE pokedex_db;
\c pokedex_db
CREATE EXTENSION vector;
```

### LLM setup
Before running the hybrid search with RAG, the local LLM must be available. This project uses Ollama to run the Qwen 2.5 3B Instruct model locally.

Pull the model with:

```bash
ollama pull qwen2.5:3b-instruct
```

### Start searching for Pokémon

1. Ensure the database is available and dependencies are installed.
2. Ensure LLM model is running locally by running:

```bash
ollama run qwen2.5:3b-instruct
```
3. Ensure environment variables for the database URL and LLM model are set and correct.
4. Initialize the database (create tables, load CSV data, generate embeddings):

```bash
python main.py --update
```

5. Run the script with a search method. The search query and number of Pokémon to display are **requested interactively**:

```bash
python main.py --search [keyword|semantic|hybrid]
```

6. For hybrid search, the RAG pipeline will combine results and rerank them before generating an answer using the LLM.

Optional verbose logging:

```bash
python main.py --search hybrid --verbose
```

It is also possible to combine flags and add verbose logging, like so:

```bash
python main.py --update --search hybrid --verbose
```

### After starting:

* Database tables are created
* The CSV file (`pokemon-dataset/pokedex.csv`) is loaded
* Embeddings are generated
* You are prompted for a search query and the number of Pokémon to give back

## RAG Pipeline steps

### 1. Search database

#### Keyword search

Uses PostgreSQL full-text search.
Best for exact terms and fast filtering.

####  Semantic search

Uses pgvector for vector similarity on descriptions.
Best for meaning-based queries.

#### Hybrid search

Combination of keyword and semantic search. Uses a reranker to generate the final search results.
Steps for hybrid search:

1. Keyword search (PostgreSQL FTS)
2. Semantic search (pgvector similarity)
3. Combine result sets and remove duplicates
4. Rerank using a cross-encoder model
5. Return the most relevant results

### 2. Generate LLM-based answer

After retrieval, the selected Pokémon records are passed to the LLM for answer generation:

1. Context construction
   * Pokémon metadata (name, type, description) is formatted for the prompt

2. Prompt injection
   * The user query and retrieved context are inserted into a prompt template

3. LLM model generates an answer based solely on the retrieved context

4. The generated answer is returned and printed




