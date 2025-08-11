import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from db.ingestion import Neo4jIngestor
from config.settings import get_settings

def main():
    """Main function to run ingestion."""
    try:
        settings = get_settings()
        ingestor = Neo4jIngestor(
            uri=settings.neo4j_uri,
            user=settings.neo4j_user,
            password=settings.neo4j_password
        )
        print("Starting data ingestion...")
        ingestor.ingest()
        print("Data ingestion completed successfully!")
        
    except Exception as e:
        print(f"Error during ingestion: {e}")
        sys.exit(1)
    finally:
        if 'ingestor' in locals():
            ingestor.close()

if __name__ == "__main__":
    main()
