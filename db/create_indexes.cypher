CREATE CONSTRAINT doc_id_constraint IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE;
CREATE CONSTRAINT entity_name_constraint IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE;
CREATE FULLTEXT INDEX entity_name_fulltext_index IF NOT EXISTS FOR (e:Entity) ON EACH [e.name];
CREATE FULLTEXT INDEX documentFulltextIndex IF NOT EXISTS FOR (d:Document) ON EACH [d.title, d.source];
CREATE VECTOR INDEX documentEmbeddingIndex IF NOT EXISTS FOR (d:Document) ON (d.embedding) OPTIONS { indexConfig: { `vector.dimensions`: 1536, `vector.similarity_function`: 'cosine' }};