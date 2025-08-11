# ingest.py
import os
from neo4j import GraphDatabase
from config.settings import get_settings

class Neo4jIngestor:
    # Paths to Cypher scripts
    DROP_SCRIPT_PATH = os.path.join("db", "drop_indexes.cypher")
    INDEX_SCRIPT_PATH = os.path.join("db", "create_indexes.cypher")
    LOAD_SCRIPT_PATH = os.path.join("db", "load_data.cypher")

    def __init__(self, uri=None, user=None, password=None):
        settings = get_settings()
        self.uri = uri or settings.neo4j_uri
        self.user = user or settings.neo4j_user
        self.password = password or settings.neo4j_password
        self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))

    def close(self):
        if self.driver:
            self.driver.close()

    def run_cypher_file(self, session, file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            cypher_commands = f.read().split(";")
            for command in cypher_commands:
                command = command.strip()
                if command:
                    session.run(command)
                    print(f"Executed: {command[:50]}...")

    def ingest(self):
        with self.driver.session() as session:
            print("Dropping indexes...")
            self.run_cypher_file(session, self.DROP_SCRIPT_PATH)

            print("Creating indexes...")
            self.run_cypher_file(session, self.INDEX_SCRIPT_PATH)

            print("Loading data...")
            self.run_cypher_file(session, self.LOAD_SCRIPT_PATH)

        print("Data ingestion complete.")

if __name__ == "__main__":
    ingestor = Neo4jIngestor()
    try:
        ingestor.ingest()
    finally:
        ingestor.close()
