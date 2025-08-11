# HybridRAG: Hybrid Retrieval-Augmented Generation System

A comprehensive hybrid retrieval-augmented generation (RAG) system that combines vector search, fulltext search, and semantic search strategies to provide accurate and contextually relevant responses.

## Features

- **Hybrid Retrieval**: Combines vector similarity, fulltext search, and semantic search
- **Neo4j Integration**: Uses Neo4j as the knowledge graph and vector database
- **OpenAI Integration**: Leverages OpenAI's GPT models for response generation
- **Configurable Weights**: Adjustable strategy weights for different search approaches
- **Multiple Retrieval Strategies**: Support for vector-only, fulltext-only, semantic-only, and hybrid approaches
- **Comprehensive Pipeline**: Complete RAG pipeline from retrieval to response generation

## Architecture

```
Query → Hybrid Retriever → Context Preparation → LLM Generation → Response
                ↓
    [Vector Search + Fulltext Search + Semantic Search]
                ↓
    [Neo4j Database with Vector + Fulltext Indexes]
```

## Prerequisites

- Python 3.8+
- Neo4j 5.15+ (for vector search support)
- OpenAI API key (optional, for LLM generation)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd hybridRAG-1
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Configuration

### Environment Variables

Create a `.env` file in the root directory:

```env
# Neo4j Configuration
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password

# OpenAI Configuration (optional)
OPENAI_API_KEY=your_openai_api_key
```

### Configuration File

The system uses `config.yaml` for advanced configuration:

```yaml
# Database Configuration
database:
  neo4j:
    uri: "bolt://localhost:7687"
    user: "neo4j"
    password: "password"
    database: "neo4j"

# Retrieval Configuration
retrieval:
  default_top_k: 5
  vector_similarity_threshold: 0.7
  strategy_weights:
    vector: 0.4
    fulltext: 0.3
    semantic: 0.3

# LLM Configuration
llm:
  max_context_length: 4000
  temperature: 0.7
  max_tokens: 500
```

## Quick Start

### 1. Start Neo4j Database

Ensure Neo4j is running and accessible at the configured URI.

### 2. Ingest Data

```bash
python scripts/run_ingest.py
```

This will:
- Create necessary indexes and constraints
- Load sample data from `data/dataset.json`

### 3. Run Demo Queries

```bash
# Basic demo with hybrid strategy
python scripts/run_demo_query.py

# Compare different retrieval strategies
python scripts/run_strategy_comparison.py
```

## Usage

### Basic Usage

```python
from src.llm.pipeline import LLMPipeline

# Initialize pipeline
pipeline = LLMPipeline(
    uri="bolt://localhost:7687",
    user="neo4j",
    password="password"
)

# Process a query
result = pipeline.process_query(
    query="What is machine learning?",
    strategy="hybrid",
    top_k=5,
    include_explanation=True
)

print(result['response'])
```

### Available Strategies

- **`vector`**: Vector similarity search using embeddings
- **`fulltext`**: Fulltext search using Neo4j fulltext indexes
- **`semantic`**: Semantic search using text analysis
- **`hybrid`**: Combines all strategies with configurable weights

### Customizing Retrieval Weights

```python
# Set custom weights for hybrid strategy
pipeline.retriever.set_weights(
    vector_weight=0.6,
    fulltext_weight=0.3,
    semantic_weight=0.1
)
```

### Batch Processing

```python
queries = [
    "What is machine learning?",
    "Explain neural networks",
    "How does deep learning work?"
]

results = pipeline.batch_process(queries, strategy="hybrid", top_k=3)
```

## API Reference

### LLMPipeline

Main class for orchestrating the complete RAG pipeline.

#### Methods

- `process_query(query, strategy, top_k, include_explanation)`: Process a single query
- `batch_process(queries, strategy, top_k)`: Process multiple queries
- `set_pipeline_config(max_context_length, temperature, max_tokens)`: Configure pipeline parameters
- `get_pipeline_stats()`: Get pipeline statistics and configuration

### HybridRetriever

Handles document retrieval using multiple strategies.

#### Methods

- `retrieve(query, top_k, strategy)`: Retrieve documents using specified strategy
- `set_weights(vector_weight, fulltext_weight, semantic_weight)`: Set strategy weights
- `explain_retrieval(query, top_k)`: Get detailed retrieval explanation

## Data Format

The system expects data in the following JSON format:

```json
{
  "documents": [
    {
      "id": "doc1",
      "title": "Document Title",
      "source": "http://example.com",
      "created_at": "2024-01-01",
      "lang": "en",
      "passages": [
        {
          "id": "p1_0",
          "text": "Document content text...",
          "embedding": [0.1, 0.2, ...],
          "chunk_idx": 0
        }
      ]
    }
  ]
}
```

## Indexes

The system creates the following Neo4j indexes:

- **Vector Index**: `documentEmbeddingIndex` for ANN search on document embeddings
- **Fulltext Index**: `documentFulltextIndex` for fulltext search on document properties
- **Entity Index**: `entity_name_fulltext_index` for entity search
- **Constraints**: Unique constraints on document IDs and entity names

## Troubleshooting

### Common Issues

1. **Vector Index Not Found**: Ensure Neo4j version 5.15+ is installed
2. **OpenAI API Errors**: Check your API key configuration
3. **Connection Issues**: Verify Neo4j is running and accessible

### Debug Mode

Enable debug output by modifying the ingestion script or adding logging configuration.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

[Add your license information here]

## Support

For issues and questions, please open a GitHub issue or contact the maintainers.