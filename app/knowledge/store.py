# app/knowledge/store.py

from typing import List, Dict, Any, Optional, Tuple
from app.knowledge.embedding import EmbeddingGenerator
from app.knowledge.vector_db import VectorDatabase
from app.logger import logger
import os

class VectorStore:
    """Manages the storage and retrieval of knowledge using embeddings and a vector database."""

    def __init__(self, 
                 embedding_model_name: str = "all-MiniLM-L6-v2", 
                 index_path: Optional[str] = None, 
                 data_path: Optional[str] = None):
        """
        Initializes the VectorStore.

        Args:
            embedding_model_name (str): Name of the SentenceTransformer model for embeddings.
            index_path (Optional[str]): Path to save/load the FAISS index. 
                                        If None, defaults to a path in the workspace.
            data_path (Optional[str]): Path to save/load associated metadata (e.g., text content).
                                      If None, defaults to a path in the workspace.
        """
        self.embed_generator = EmbeddingGenerator(model_name=embedding_model_name)
        self.dimension = self.embed_generator.get_embedding_dimension()
        if self.dimension == 0:
            logger.error("Failed to get embedding dimension. VectorStore cannot be initialized.")
            raise ValueError("Could not determine embedding dimension for VectorStore.")

        self.vector_db = VectorDatabase(dimension=self.dimension)
        
        # Default paths if not provided
        # Assuming a workspace directory exists. This should align with config.py WORKSPACE_ROOT
        # For simplicity, using a fixed path here. In a real app, use config.
        self.workspace_root = os.path.join(os.path.expanduser("~"), "OpenAgent", "workspace", "knowledge_store")
        os.makedirs(self.workspace_root, exist_ok=True)

        self.index_path = index_path or os.path.join(self.workspace_root, "faiss_index.idx")
        self.data_path = data_path or os.path.join(self.workspace_root, "metadata.json") # Simple JSON for metadata

        self.metadata_store: Dict[str, Dict[str, Any]] = {} # Stores {doc_id: {text: "...", source: "..."}}

        self._load_all()

    def _load_all(self):
        """Loads the index and metadata from disk if they exist."""
        if os.path.exists(self.index_path):
            try:
                self.vector_db.load_index(self.index_path)
                logger.info(f"Successfully loaded FAISS index from {self.index_path}")
            except Exception as e:
                logger.error(f"Failed to load FAISS index from {self.index_path}: {e}. Initializing new index.")
                # Ensure dimension is set if index load fails but store was initialized with one
                if self.dimension and self.vector_db.dimension is None:
                    self.vector_db._initialize_index(self.dimension)
        else:
            logger.info(f"No FAISS index found at {self.index_path}. Initializing new index.")
            # Ensure dimension is set if no index file exists
            if self.dimension and self.vector_db.index is None:
                 self.vector_db._initialize_index(self.dimension)

        if os.path.exists(self.data_path):
            import json
            try:
                with open(self.data_path, "r") as f:
                    self.metadata_store = json.load(f)
                logger.info(f"Successfully loaded metadata from {self.data_path}")
            except Exception as e:
                logger.error(f"Failed to load metadata from {self.data_path}: {e}. Initializing empty metadata store.")
                self.metadata_store = {}
        else:
            logger.info(f"No metadata file found at {self.data_path}. Initializing empty metadata store.")
            self.metadata_store = {}

    def _save_all(self):
        """Saves the index and metadata to disk."""
        try:
            self.vector_db.save_index(self.index_path)
        except Exception as e:
            logger.error(f"Failed to save FAISS index to {self.index_path}: {e}")
        
        import json
        try:
            with open(self.data_path, "w") as f:
                json.dump(self.metadata_store, f, indent=4)
            logger.info(f"Successfully saved metadata to {self.data_path}")
        except Exception as e:
            logger.error(f"Failed to save metadata to {self.data_path}: {e}")

    def store_knowledge(self, documents: List[Dict[str, Any]], batch_size: int = 32):
        """
        Stores knowledge by embedding documents and adding them to the vector database.
        Each document is a dictionary, expected to have at least a "text" key and an "id" key.
        Other keys like "source" will be stored as metadata.

        Args:
            documents (List[Dict[str, Any]]): A list of documents to store.
                                              Each dict must have "id" and "text".
            batch_size (int): Number of documents to process in a batch for embedding.
        """
        if not documents:
            logger.warning("store_knowledge called with no documents.")
            return

        texts_to_embed = []
        doc_ids_for_batch = []
        new_metadata = {}

        for doc in documents:
            if not isinstance(doc, dict) or "text" not in doc or "id" not in doc:
                logger.warning(f"Skipping invalid document: {doc}")
                continue
            if doc["id"] in self.metadata_store:
                logger.info(f"Document with ID {doc['id']} already exists. Skipping.") # Or update logic
                continue
            
            texts_to_embed.append(doc["text"])
            doc_ids_for_batch.append(doc["id"])
            new_metadata[doc["id"]] = {k: v for k, v in doc.items() if k != "id"}

            if len(texts_to_embed) >= batch_size:
                self._process_batch(texts_to_embed, doc_ids_for_batch, new_metadata)
                texts_to_embed = []
                doc_ids_for_batch = []
                new_metadata = {} # Reset for next batch, already processed
        
        # Process any remaining documents
        if texts_to_embed:
            self._process_batch(texts_to_embed, doc_ids_for_batch, new_metadata)

        if any(doc["id"] not in self.metadata_store for doc in documents if isinstance(doc, dict) and "id" in doc): # Check if any new data was actually added
             self._save_all() # Save if new data was added

    def _process_batch(self, texts: List[str], ids: List[str], metadata_batch: Dict[str, Any]):
        """Helper to process a batch of texts for embedding and storage."""
        if not texts:
            return
        
        embeddings = self.embed_generator.generate(texts)
        if not embeddings or len(embeddings) != len(texts):
            logger.error("Failed to generate embeddings for a batch or mismatch in count.")
            return

        valid_embeddings = []
        valid_ids_for_db = [] # These are the string IDs for the vector_db
        
        for i, emb in enumerate(embeddings):
            if emb: # Ensure embedding is not empty
                valid_embeddings.append(emb)
                valid_ids_for_db.append(ids[i])
            else:
                logger.warning(f"Skipping empty embedding for document ID: {ids[i]}")

        if valid_embeddings:
            # The FAISS index itself uses numerical indices. We map these to our string IDs via metadata_store
            # and the vector_db.id_map (if the FAISS index type supports it or we build a mapping).
            # For simplicity here, vector_db.add can take string IDs which it might map internally or store.
            self.vector_db.add(valid_embeddings, ids=valid_ids_for_db) 
            
            # Update metadata store for successfully added embeddings
            for doc_id in valid_ids_for_db:
                if doc_id in metadata_batch:
                    self.metadata_store[doc_id] = metadata_batch[doc_id]
            logger.info(f"Processed and stored batch of {len(valid_embeddings)} documents.")

    def retrieve_context(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieves relevant context for a given query.

        Args:
            query (str): The query string.
            k (int): The number of relevant documents to retrieve.

        Returns:
            List[Dict[str, Any]]: A list of retrieved documents with their metadata.
        """
        if not query:
            logger.warning("retrieve_context called with empty query.")
            return []
        
        if self.vector_db.get_total_vectors() == 0:
            logger.info("Knowledge store is empty. Cannot retrieve context.")
            return []

        query_embedding = self.embed_generator.generate(query)
        if not query_embedding:
            logger.error("Failed to generate embedding for the query.")
            return []

        distances, indices = self.vector_db.search(query_embedding, k=k)
        
        results = []
        # The indices from FAISS are numerical. We need to map them back to our document IDs.
        # The current vector_db.id_map is a simple list mirroring FAISS internal additions.
        # This assumes that the order of addition to vector_db.id_map matches FAISS internal indices.
        # This is a simplification; a robust system might use faiss.IndexIDMap or a DB.
        
        # Correct mapping: vector_db.id_map stores the string IDs at the FAISS index position.
        # So, if FAISS returns index `idx`, the string ID is `self.vector_db.id_map[idx]`
        
        for i, faiss_idx in enumerate(indices):
            if 0 <= faiss_idx < len(self.vector_db.id_map):
                doc_id = self.vector_db.id_map[faiss_idx]
                if doc_id in self.metadata_store:
                    retrieved_doc = {
                        "id": doc_id,
                        "text": self.metadata_store[doc_id].get("text", ""), # Ensure text is present
                        "source": self.metadata_store[doc_id].get("source", "N/A"),
                        "score": 1 - distances[i] # Example: convert L2 distance to a similarity score (0-1)
                                                # This is a simplistic conversion, actual score depends on distance metric
                    }
                    results.append(retrieved_doc)
                else:
                    logger.warning(f"Metadata not found for document ID: {doc_id} at FAISS index {faiss_idx}")
            else:
                logger.warning(f"FAISS index {faiss_idx} out of bounds for id_map (len: {len(self.vector_db.id_map)}).")
        
        return results

    def get_document_by_id(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves a document by its ID."""
        return self.metadata_store.get(doc_id)

    def get_all_document_ids(self) -> List[str]:
        """Returns a list of all document IDs in the store."""
        return list(self.metadata_store.keys())

    def reset_store(self):
        """Resets the entire knowledge store, clearing the index and metadata."""
        self.vector_db.reset()
        self.metadata_store = {}
        # Delete files from disk
        if os.path.exists(self.index_path):
            try:
                os.remove(self.index_path)
                logger.info(f"Deleted FAISS index file: {self.index_path}")
            except OSError as e:
                logger.error(f"Error deleting FAISS index file {self.index_path}: {e}")
        if os.path.exists(self.data_path):
            try:
                os.remove(self.data_path)
                logger.info(f"Deleted metadata file: {self.data_path}")
            except OSError as e:
                logger.error(f"Error deleting metadata file {self.data_path}: {e}")
        logger.info("VectorStore has been reset.")

# Example Usage
if __name__ == "__main__":
    # Create a dummy workspace for the example
    example_workspace = os.path.join(os.path.expanduser("~"), "OpenAgent_example_ws", "knowledge_store")
    os.makedirs(example_workspace, exist_ok=True)
    example_index_path = os.path.join(example_workspace, "example_faiss.idx")
    example_data_path = os.path.join(example_workspace, "example_metadata.json")

    # Clean up previous example files if they exist
    if os.path.exists(example_index_path):
        os.remove(example_index_path)
    if os.path.exists(example_data_path):
        os.remove(example_data_path)

    try:
        store = VectorStore(index_path=example_index_path, data_path=example_data_path)
        print(f"Store initialized. Index path: {store.index_path}, Data path: {store.data_path}")
        print(f"Initial document count: {len(store.get_all_document_ids())}")

        docs_to_add = [
            {"id": "doc1", "text": "The quick brown fox jumps over the lazy dog.", "source": "classic"},
            {"id": "doc2", "text": "Apples are a type of fruit, often red or green.", "source": "common knowledge"},
            {"id": "doc3", "text": "The capital of France is Paris.", "source": "geography"},
            {"id": "doc4", "text": "A lazy dog sat under a tree, while a quick fox ran by.", "source": "variation"}
        ]
        store.store_knowledge(docs_to_add)
        print(f"Document count after adding: {len(store.get_all_document_ids())}")
        print(f"Total vectors in FAISS: {store.vector_db.get_total_vectors()}")

        # Test retrieval
        query1 = "What is the capital city of France?"
        results1 = store.retrieve_context(query1, k=2)
        print(f"\nResults for query 	'{query1}	':")
        for res in results1:
            print(f"  ID: {res['id']}, Score: {res['score']:.4f}, Text: {res['text'][:50]}..., Source: {res['source']}")

        query2 = "Information about quick animals and lazy animals"
        results2 = store.retrieve_context(query2, k=3)
        print(f"\nResults for query 	'{query2}	':")
        for res in results2:
            print(f"  ID: {res['id']}, Score: {res['score']:.4f}, Text: {res['text'][:50]}..., Source: {res['source']}")
        
        # Test loading from disk
        print("\nReloading store from disk...")
        store_reloaded = VectorStore(index_path=example_index_path, data_path=example_data_path)
        print(f"Document count in reloaded store: {len(store_reloaded.get_all_document_ids())}")
        results_reloaded = store_reloaded.retrieve_context(query2, k=3)
        assert len(results_reloaded) == len(results2), "Data mismatch after reload"
        print("Reload successful and data matches.")

        # Test reset
        # store.reset_store()
        # print(f"\nDocument count after reset: {len(store.get_all_document_ids())}")
        # print(f"Total vectors in FAISS after reset: {store.vector_db.get_total_vectors()}")

    except ValueError as ve:
        print(f"ValueError during example: {ve}")
    except RuntimeError as re_:
        print(f"RuntimeError during example: {re_}")
    except ImportError:
        print("A required library (e.g., sentence-transformers, faiss) might not be installed.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        # Clean up example files
        # if os.path.exists(example_index_path):
        #     os.remove(example_index_path)
        # if os.path.exists(example_data_path):
        #     os.remove(example_data_path)
        pass # Keep files for inspection if needed


