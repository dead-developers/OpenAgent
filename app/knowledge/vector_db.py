# app/knowledge/vector_db.py

import faiss
import numpy as np
from typing import List, Optional, Tuple
from app.logger import logger

class VectorDatabase:
    """Manages a FAISS index for storing and searching vector embeddings."""

    def __init__(self, dimension: Optional[int] = None, index_type: str = "IndexFlatL2"):
        """
        Initializes the VectorDatabase.

        Args:
            dimension (Optional[int]): The dimension of the vectors to be stored.
                                       If None, the index will be initialized upon first data addition.
            index_type (str): The type of FAISS index to use (e.g., "IndexFlatL2", "IndexIVFFlat").
                              Defaults to "IndexFlatL2".
        """
        self.dimension = dimension
        self.index_type = index_type
        self.index: Optional[faiss.Index] = None
        self.is_trained: bool = False
        self.id_map: List[str] = [] # To map FAISS internal IDs to external string IDs if needed

        if self.dimension:
            self._initialize_index(self.dimension)

    def _initialize_index(self, dimension: int):
        """Initializes the FAISS index with the given dimension."""
        self.dimension = dimension
        try:
            if hasattr(faiss, self.index_type):
                self.index = getattr(faiss, self.index_type)(self.dimension)
                logger.info(f"FAISS index {self.index_type} initialized with dimension {self.dimension}.")
                # For some index types, training is required before adding vectors.
                # IndexFlatL2 does not require training, but others like IndexIVFFlat do.
                if not self.index_type.startswith("IndexFlat") and not self.index_type.startswith("IndexIDMap") :
                    self.is_trained = False # Requires training
                else:
                    self.is_trained = True # Does not require explicit training
            else:
                logger.error(f"FAISS index type {self.index_type} not found. Falling back to IndexFlatL2.")
                self.index = faiss.IndexFlatL2(self.dimension)
                self.index_type = "IndexFlatL2"
                self.is_trained = True
        except Exception as e:
            logger.error(f"Error initializing FAISS index: {e}")
            raise RuntimeError(f"Could not initialize FAISS index: {e}")

    def add(self, embeddings: List[List[float]], ids: Optional[List[str]] = None):
        """
        Adds embeddings to the index.

        Args:
            embeddings (List[List[float]]): A list of embeddings (each embedding is a list of floats).
            ids (Optional[List[str]]): Optional list of string IDs corresponding to the embeddings.
                                       If provided, an IndexIDMap might be used or a manual mapping maintained.
        """
        if not embeddings:
            logger.warning("Attempted to add empty list of embeddings.")
            return

        embeddings_np = np.array(embeddings, dtype=np.float32)

        if self.index is None:
            current_dimension = embeddings_np.shape[1]
            if current_dimension == 0:
                logger.error("Cannot initialize index with 0-dimension embeddings.")
                return
            self._initialize_index(current_dimension)
        
        if embeddings_np.shape[1] != self.dimension:
            logger.error(f"Embedding dimension mismatch. Expected {self.dimension}, got {embeddings_np.shape[1]}.")
            return

        if not self.is_trained:
            if hasattr(self.index, "train") and embeddings_np.shape[0] > 0:
                logger.info(f"Training FAISS index {self.index_type} with {embeddings_np.shape[0]} vectors.")
                self.index.train(embeddings_np)
                self.is_trained = True
                logger.info("FAISS index training complete.")
            else:
                logger.warning("Index requires training, but no data provided or train method not available.")
                return
        
        try:
            self.index.add(embeddings_np)
            # If using IDs, store them. For simplicity, this example just appends to a list.
            # A more robust solution would use faiss.IndexIDMap or a proper database for ID mapping.
            if ids:
                start_index = len(self.id_map)
                self.id_map.extend(ids)
                # If you need to map FAISS index to these string IDs, you might need IndexIDMap
                # or handle it externally if your index doesn't support direct ID mapping.
            logger.info(f"Added {embeddings_np.shape[0]} embeddings to the index. Total: {self.index.ntotal}")
        except Exception as e:
            logger.error(f"Error adding embeddings to FAISS index: {e}")

    def search(self, query_embedding: List[float], k: int = 5) -> Tuple[List[float], List[int]]:
        """
        Searches the index for the k nearest neighbors to the query_embedding.

        Args:
            query_embedding (List[float]): The query embedding (a list of floats).
            k (int): The number of nearest neighbors to retrieve.

        Returns:
            Tuple[List[float], List[int]]: A tuple containing:
                - distances (List[float]): List of distances to the k nearest neighbors.
                - indices (List[int]): List of indices of the k nearest neighbors.
        """
        if self.index is None or self.index.ntotal == 0:
            logger.warning("Search attempted on an empty or uninitialized index.")
            return [], []

        query_np = np.array([query_embedding], dtype=np.float32)
        if query_np.shape[1] != self.dimension:
            logger.error(f"Query embedding dimension mismatch. Expected {self.dimension}, got {query_np.shape[1]}.")
            return [], []

        try:
            # Adjust k if it's larger than the number of items in the index
            actual_k = min(k, self.index.ntotal)
            if actual_k == 0:
                 return [], []
            distances, indices = self.index.search(query_np, actual_k)
            return distances[0].tolist(), indices[0].tolist()
        except Exception as e:
            logger.error(f"Error searching FAISS index: {e}")
            return [], []

    def get_total_vectors(self) -> int:
        """Returns the total number of vectors in the index."""
        return self.index.ntotal if self.index else 0

    def reset(self):
        """Resets the index, clearing all stored vectors."""
        if self.dimension:
            self._initialize_index(self.dimension) # Re-initialize with the same dimension
        else:
            self.index = None # If dimension was never set, just clear the index
        self.id_map = []
        logger.info("FAISS index has been reset.")

    def save_index(self, path: str):
        """Saves the FAISS index to a file."""
        if self.index:
            try:
                faiss.write_index(self.index, path)
                logger.info(f"FAISS index saved to {path}")
            except Exception as e:
                logger.error(f"Error saving FAISS index to {path}: {e}")
        else:
            logger.warning("Attempted to save an uninitialized FAISS index.")

    def load_index(self, path: str):
        """Loads a FAISS index from a file."""
        try:
            self.index = faiss.read_index(path)
            self.dimension = self.index.d
            self.is_trained = self.index.is_trained # is_trained is an attribute of the index itself
            # Note: id_map would need to be saved/loaded separately if it's critical.
            self.id_map = [] # Reset id_map, assuming it's not part of the FAISS index file
            logger.info(f"FAISS index loaded from {path}. Dimension: {self.dimension}, Total vectors: {self.index.ntotal}")
        except Exception as e:
            logger.error(f"Error loading FAISS index from {path}: {e}")
            # Potentially re-initialize to a clean state or raise
            self.index = None
            self.dimension = None # Reset dimension if load fails

# Example usage (for testing purposes)
if __name__ == "__main__":
    try:
        # Initialize with a dimension (e.g., from an embedding model)
        dim = 384 # Example dimension for all-MiniLM-L6-v2
        db = VectorDatabase(dimension=dim)

        # Add some embeddings
        embeddings_to_add = [
            list(np.random.rand(dim).astype(np.float32)),
            list(np.random.rand(dim).astype(np.float32)),
            list(np.random.rand(dim).astype(np.float32))
        ]
        db.add(embeddings_to_add, ids=["doc1", "doc2", "doc3"])
        print(f"Total vectors in DB: {db.get_total_vectors()}")

        # Search for similar embeddings
        query_vec = list(np.random.rand(dim).astype(np.float32))
        distances, indices = db.search(query_vec, k=2)
        print(f"Search results for query:")
        for i, idx in enumerate(indices):
            print(f"  Neighbor {i+1}: Index {idx}, Distance {distances[i]:.4f}, Mapped ID (if used): {db.id_map[idx] if idx < len(db.id_map) else 'N/A'}")

        # Save and load index
        index_file = "/tmp/my_faiss_index.index"
        db.save_index(index_file)
        
        db_loaded = VectorDatabase() # Initialize without dimension, will be set on load
        db_loaded.load_index(index_file)
        print(f"Total vectors in loaded DB: {db_loaded.get_total_vectors()}")
        
        distances_loaded, indices_loaded = db_loaded.search(query_vec, k=2)
        print(f"Search results from loaded DB:")
        for i, idx in enumerate(indices_loaded):
            # Note: id_map is not saved with faiss.write_index, so it won't be available here unless handled separately
            print(f"  Neighbor {i+1}: Index {idx}, Distance {distances_loaded[i]:.4f}")

        db.reset()
        print(f"Total vectors after reset: {db.get_total_vectors()}")

    except RuntimeError as e:
        print(f"Could not run example: {e}")
    except ImportError:
        print("FAISS library not found. Please install it to run this example (pip install faiss-cpu or faiss-gpu).")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


