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
            lines = f.readlines()
            
            # Process lines to build commands
            current_command = ""
            for line in lines:
                line = line.strip()
                
                # Skip empty lines and pure comment lines
                if not line or line.startswith("//") or line.startswith("--"):
                    continue
                
                # Add line to current command
                current_command += line + " "
                
                # If line ends with semicolon, execute the command
                if line.endswith(";"):
                    command = current_command.strip()
                    if command:
                        try:
                            result = session.run(command)
                            print(f"Executed: {command[:50]}...")
                            # Print summary if available
                            if hasattr(result, 'consume'):
                                summary = result.consume()
                                if summary.counters.indexes_added > 0:
                                    print(f"  -> Created {summary.counters.indexes_added} indexes")
                        except Exception as e:
                            print(f"Error executing: {command[:50]}...")
                            print(f"  Error: {e}")
                            # Continue with other commands instead of failing completely
                    
                    # Reset for next command
                    current_command = ""

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
