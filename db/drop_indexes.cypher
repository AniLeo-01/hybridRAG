// Drop existing indexes/constraints to avoid conflicts
DROP CONSTRAINT doc_id_constraint IF EXISTS;
DROP CONSTRAINT entity_name_constraint IF EXISTS;
DROP INDEX entity_name_fulltext_index IF EXISTS;
DROP INDEX documentFulltextIndex IF EXISTS;
DROP INDEX documentEmbeddingIndex IF EXISTS;
