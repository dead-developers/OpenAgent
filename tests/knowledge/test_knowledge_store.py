# /home/ubuntu/OpenAgent/tests/knowledge/test_knowledge_store.py

import pytest
import os
import shutil
import json
import numpy as np

from app.knowledge.embedding import EmbeddingGenerator
from app.knowledge.vector_db import VectorDatabase
from app.knowledge.store import VectorStore
from app.logger import logger # Assuming logger is configured

# --- Fixtures ---

@pytest.fixture(scope="module")
def embedding_generator():
    """Provides an EmbeddingGenerator instance."""
    return EmbeddingGenerator(model_name="all-MiniLM-L6-v2") # Use a known fast model

@pytest.fixture
def temp_workspace_path(tmp_path):
    """Creates a temporary workspace directory for testing store persistence."""
    ws_path = tmp_path / "test_knowledge_store_ws"
    os.makedirs(ws_path, exist_ok=True)
    return ws_path

@pytest.fixture
def vector_store_instance(temp_workspace_path, embedding_generator):
    """Provides a VectorStore instance with a temporary persistence path."""
    index_path = str(temp_workspace_path / "test_faiss.idx")
    data_path = str(temp_workspace_path / "test_metadata.json")
    # Ensure clean state for each test using this fixture if it modifies files
    if os.path.exists(index_path):
        os.remove(index_path)
    if os.path.exists(data_path):
        os.remove(data_path)
    
    store = VectorStore(
        embedding_model_name="all-MiniLM-L6-v2", 
        index_path=index_path, 
        data_path=data_path
    )
    # Override the embed_generator to use the fixture one for consistency if needed, though VectorStore creates its own.
    # store.embed_generator = embedding_generator 
    # store.dimension = embedding_generator.get_embedding_dimension()
    # store.vector_db = VectorDatabase(dimension=store.dimension) # Re-init with consistent dimension
    return store

# --- Tests for EmbeddingGenerator ---

def test_embedding_generator_init(embedding_generator):
    assert embedding_generator.model is not None
    assert embedding_generator.get_embedding_dimension() > 0

def test_embedding_generator_generate_single(embedding_generator):
    text = "This is a test sentence."
    embedding = embedding_generator.generate(text)
    assert isinstance(embedding, list)
    assert len(embedding) == embedding_generator.get_embedding_dimension()
    assert all(isinstance(x, float) for x in embedding)

def test_embedding_generator_generate_multiple(embedding_generator):
    texts = ["First sentence.", "Second sentence."]
    embeddings = embedding_generator.generate(texts)
    assert isinstance(embeddings, list)
    assert len(embeddings) == 2
    assert all(isinstance(emb, list) and len(emb) == embedding_generator.get_embedding_dimension() for emb in embeddings)

def test_embedding_generator_empty_input(embedding_generator):
    assert embedding_generator.generate("") == []
    assert embedding_generator.generate([]) == []

# --- Tests for VectorDatabase ---

@pytest.fixture
def vector_db_instance(embedding_generator):
    dim = embedding_generator.get_embedding_dimension()
    return VectorDatabase(dimension=dim)

def test_vector_db_init(vector_db_instance, embedding_generator):
    assert vector_db_instance.index is not None
    assert vector_db_instance.dimension == embedding_generator.get_embedding_dimension()
    assert vector_db_instance.get_total_vectors() == 0

def test_vector_db_add_and_search(vector_db_instance, embedding_generator):
    dim = embedding_generator.get_embedding_dimension()
    embeddings_to_add = [
        np.random.rand(dim).astype(np.float32).tolist(),
        np.random.rand(dim).astype(np.float32).tolist()
    ]
    vector_db_instance.add(embeddings_to_add, ids=["doc1", "doc2"])
    assert vector_db_instance.get_total_vectors() == 2

    query_vec = embeddings_to_add[0]
    distances, indices = vector_db_instance.search(query_vec, k=1)
    assert len(indices) == 1
    assert indices[0] == 0 # Should find itself
    assert distances[0] < 1e-5 # Distance to self should be close to 0

def test_vector_db_add_empty(vector_db_instance):
    vector_db_instance.add([])
    assert vector_db_instance.get_total_vectors() == 0

def test_vector_db_search_empty(vector_db_instance, embedding_generator):
    query_vec = np.random.rand(embedding_generator.get_embedding_dimension()).astype(np.float32).tolist()
    distances, indices = vector_db_instance.search(query_vec, k=1)
    assert distances == []
    assert indices == []

def test_vector_db_dimension_mismatch_add(vector_db_instance):
    # Initialized with dim, try adding embedding with different dim
    wrong_dim_embedding = [np.random.rand(vector_db_instance.dimension + 1).astype(np.float32).tolist()]
    # This should log an error and not add, or raise an error depending on implementation
    # For now, we check that total vectors don't change
    initial_count = vector_db_instance.get_total_vectors()
    vector_db_instance.add(wrong_dim_embedding)
    assert vector_db_instance.get_total_vectors() == initial_count

def test_vector_db_save_load_reset(vector_db_instance, embedding_generator, tmp_path):
    dim = embedding_generator.get_embedding_dimension()
    db_path = tmp_path / "test_db.index"
    
    embeddings_to_add = [np.random.rand(dim).astype(np.float32).tolist()]
    vector_db_instance.add(embeddings_to_add, ids=["doc_save"])
    assert vector_db_instance.get_total_vectors() == 1
    vector_db_instance.save_index(str(db_path))
    assert os.path.exists(db_path)

    loaded_db = VectorDatabase(dimension=dim) # Or init without dim and let load set it
    loaded_db.load_index(str(db_path))
    assert loaded_db.get_total_vectors() == 1
    assert loaded_db.dimension == dim

    vector_db_instance.reset()
    assert vector_db_instance.get_total_vectors() == 0

# --- Tests for VectorStore ---

DOCS_FOR_TESTING = [
    {"id": "vs_doc1", "text": "The quick brown fox jumps over the lazy dog.", "source": "classic"},
    {"id": "vs_doc2", "text": "Apples are a type of fruit, often red or green.", "source": "common"},
    {"id": "vs_doc3", "text": "The capital of France is Paris.", "source": "geo"},
]

def test_vector_store_init(vector_store_instance):
    assert vector_store_instance.embed_generator is not None
    assert vector_store_instance.vector_db is not None
    assert vector_store_instance.dimension > 0
    assert os.path.exists(vector_store_instance.index_path) # Index is created/loaded
    assert os.path.exists(vector_store_instance.data_path)   # Metadata file is created/loaded

def test_vector_store_store_knowledge_single_batch(vector_store_instance):
    vector_store_instance.store_knowledge(DOCS_FOR_TESTING)
    assert vector_store_instance.vector_db.get_total_vectors() == len(DOCS_FOR_TESTING)
    assert len(vector_store_instance.get_all_document_ids()) == len(DOCS_FOR_TESTING)
    for doc in DOCS_FOR_TESTING:
        assert doc["id"] in vector_store_instance.metadata_store
        assert vector_store_instance.metadata_store[doc["id"]]["text"] == doc["text"]

def test_vector_store_store_knowledge_batching(vector_store_instance):
    vector_store_instance.store_knowledge(DOCS_FOR_TESTING, batch_size=1)
    assert vector_store_instance.vector_db.get_total_vectors() == len(DOCS_FOR_TESTING)
    assert len(vector_store_instance.get_all_document_ids()) == len(DOCS_FOR_TESTING)

def test_vector_store_retrieve_context(vector_store_instance):
    vector_store_instance.store_knowledge(DOCS_FOR_TESTING)
    query = "What is the capital city of France?"
    results = vector_store_instance.retrieve_context(query, k=1)
    assert len(results) == 1
    assert results[0]["id"] == "vs_doc3"
    assert "score" in results[0]

def test_vector_store_retrieve_context_empty_store(vector_store_instance):
    query = "Any information?"
    results = vector_store_instance.retrieve_context(query, k=1)
    assert len(results) == 0

def test_vector_store_duplicate_ids(vector_store_instance):
    vector_store_instance.store_knowledge([DOCS_FOR_TESTING[0]])
    initial_count = vector_store_instance.vector_db.get_total_vectors()
    # Store the same doc again, should be skipped
    vector_store_instance.store_knowledge([DOCS_FOR_TESTING[0]])
    assert vector_store_instance.vector_db.get_total_vectors() == initial_count

def test_vector_store_persistence(temp_workspace_path, embedding_generator):
    index_path = str(temp_workspace_path / "persistent_faiss.idx")
    data_path = str(temp_workspace_path / "persistent_metadata.json")

    store1 = VectorStore(embedding_model_name="all-MiniLM-L6-v2", index_path=index_path, data_path=data_path)
    store1.store_knowledge(DOCS_FOR_TESTING)
    assert store1.vector_db.get_total_vectors() == len(DOCS_FOR_TESTING)
    # store1._save_all() # store_knowledge calls save_all if new data added

    # Create a new store instance, it should load the persisted data
    store2 = VectorStore(embedding_model_name="all-MiniLM-L6-v2", index_path=index_path, data_path=data_path)
    assert store2.vector_db.get_total_vectors() == len(DOCS_FOR_TESTING)
    assert len(store2.get_all_document_ids()) == len(DOCS_FOR_TESTING)
    retrieved_doc = store2.get_document_by_id("vs_doc1")
    assert retrieved_doc is not None
    assert retrieved_doc["text"] == DOCS_FOR_TESTING[0]["text"]

def test_vector_store_reset(vector_store_instance):
    vector_store_instance.store_knowledge(DOCS_FOR_TESTING)
    assert vector_store_instance.vector_db.get_total_vectors() > 0
    index_file = vector_store_instance.index_path
    data_file = vector_store_instance.data_path
    
    vector_store_instance.reset_store()
    assert vector_store_instance.vector_db.get_total_vectors() == 0
    assert len(vector_store_instance.get_all_document_ids()) == 0
    assert not os.path.exists(index_file)
    assert not os.path.exists(data_file)

# To run these tests: pytest path/to/test_knowledge_store.py

