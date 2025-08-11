from neo4j import GraphDatabase
from config.settings import get_settings

class Neo4jConnection:
    def __init__(self, uri=None, user=None, password=None):
        settings = get_settings()
        self.uri = uri or settings.neo4j_uri
        self.user = user or settings.neo4j_user
        self.password = password or settings.neo4j_password
        self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))

    def close(self):
        if self.driver:
            self.driver.close()

    def query(self, query, parameters=None, db=None):
        with self.driver.session(database=db) as session:
            result = session.run(query, parameters)
            return [r.data() for r in result]

# Example usage
if __name__ == "__main__":
    conn = Neo4jConnection()
    print(conn.query("RETURN 'Neo4j connection successful' AS message"))
    conn.close()
