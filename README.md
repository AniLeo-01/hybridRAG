# HybridRAG with Neo4j and LangChain

A hybrid retrieval-augmented generation (RAG) system that combines vector search, fulltext search, and semantic search using Neo4j as the backend database.

## Features

- **Hybrid Retrieval**: Combines vector similarity, fulltext search, and semantic analysis
- **Neo4j Integration**: Uses Neo4j for storing documents, embeddings, and relationships
- **Flexible Search Strategies**: Support for vector-only, fulltext-only, semantic-only, or hybrid search
- **LLM Pipeline**: Extensible pipeline for document retrieval and response generation
- **Embedding Generation**: Support for OpenAI embeddings with fallback to random embeddings for testing

## Project Structure

```
hybridrag-neo4j-langchain/
├── README.md
├── requirements.txt
├── .env
├── config.yaml
├── data/
│   └── dataset.json
├── db/
│   ├── create_indexes.cypher
│   ├── load_data.cypher
│   └── reset_db.cypher
├── scripts/
│   ├── run_ingest.py
│   └── run_demo_query.py
├── src/
│   ├── __init__.py
│   ├── config/
│   │   └── settings.py
│   ├── db/
│   │   ├── connection.py
│   │   └── ingestion.py
│   ├── embeddings/
│   │   ├── generator.py
│   │   └── utils.py
│   ├── retrieval/
│   │   ├── neo4j_vector_store.py
│   │   ├── neo4j_fulltext_retriever.py
│   │   └── hybrid_retriever.py
│   ├── llm/
│   │   └── pipeline.py
│   └── utils/
│       └── logging.py
└── notebooks/
    └── demo.ipynb
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd hybridrag-neo4j-langchain
```

2. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your Neo4j credentials and OpenAI API key
```

## Configuration

The system uses a `.env` file for configuration:

```env
# Neo4j Configuration
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password

# OpenAI Configuration (optional)
OPENAI_API_KEY=your_openai_api_key_here

# Other Configuration
LOG_LEVEL=INFO
```

## Usage

### 1. Data Ingestion

```bash
python scripts/run_ingest.py
```

### 2. Demo Queries

```bash
python scripts/run_demo_query.py
```

### 3. Jupyter Notebook

```bash
jupyter notebook notebooks/demo.ipynb
```

### 4. Programmatic Usage

```python
from src.retrieval.hybrid_retriever import HybridRetriever
from src.llm.pipeline import LLMPipeline

# Initialize components
retriever = HybridRetriever()
pipeline = LLMPipeline()

# Perform hybrid search
results = retriever.retrieve("What is machine learning?", top_k=5)

# Process through LLM pipeline
response = pipeline.process_query("What is machine learning?")
```

## Search Strategies

The system supports multiple search strategies:

- **Vector Search**: Uses embedding similarity for semantic search
- **Fulltext Search**: Traditional text-based search with Neo4j fulltext indexes
- **Semantic Search**: Word overlap and text analysis
- **Hybrid Search**: Combines all strategies with configurable weights

## Architecture

- **Retrieval Layer**: Handles document retrieval using multiple strategies
- **Embedding Layer**: Generates and manages document embeddings
- **LLM Layer**: Orchestrates retrieval and generation
- **Database Layer**: Neo4j integration for data storage and indexing

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the MIT License.