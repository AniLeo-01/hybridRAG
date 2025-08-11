// Reset database script for hybridRAG system
// This script removes all data and indexes to start fresh

// Drop all vector indexes
CALL db.indexes() YIELD name, type
WHERE type = 'VECTOR'
CALL db.index.drop(name) YIELD name
RETURN name;

// Drop all fulltext indexes
CALL db.indexes() YIELD name, type
WHERE type = 'FULLTEXT'
CALL db.index.drop(name) YIELD name
RETURN name;

// Delete all nodes
MATCH (n)
DETACH DELETE n;

// Delete all relationships
MATCH ()-[r]-()
DELETE r;

// Clear database statistics
CALL db.clearQueryCaches();
CALL db.clearQueryStatistics();

// Verify database is empty
MATCH (n) RETURN count(n) as node_count;
MATCH ()-[r]-() RETURN count(r) as relationship_count;
