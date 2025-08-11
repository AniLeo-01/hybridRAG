// Drop existing indexes/constraints to avoid conflicts
DROP INDEX doc_id_index IF EXISTS;
DROP INDEX entity_name_index IF EXISTS;
DROP INDEX doc_embedding_index IF EXISTS;
