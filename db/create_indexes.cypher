//create constraints
CREATE CONSTRAINT IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE;


// Index for fast document ID lookups
CREATE INDEX doc_id_index FOR (d:Document) ON (d.id);


CREATE FULLTEXT INDEX entity_name_index IF NOT EXISTS FOR (e:Entity) ON EACH [e.name];

// Vector index for embeddings (for ANN search)
CREATE VECTOR INDEX document_embedding_index IF NOT EXISTS
    FOR (d:Document) ON (d.embedding)
    OPTIONS { indexConfig: {
        `vector.dimensions`: 1536,
        `vector.similarity_function`: 'cosine'
    }};

// Index for entity lookups
CREATE INDEX entity_name_index FOR (e:Entity) ON (e.name);
