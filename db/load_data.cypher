// Load JSON dataset and insert into Neo4j
WITH 'file:///dataset/data.json' AS file
CALL apoc.load.json(file) YIELD value AS row
MERGE (d:Document {id: row.id})
SET d.text = row.text,
    d.embedding = row.embedding

FOREACH (entityName IN row.entities |
    MERGE (e:Entity {name: entityName})
    MERGE (d)-[:MENTIONS]->(e)
)

FOREACH (rel IN row.relationships |
    MERGE (d2:Document {id: rel.target})
    MERGE (d)-[:RELATED {type: rel.type}]->(d2)
);
