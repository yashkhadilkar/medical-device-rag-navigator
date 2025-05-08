import os 
import logging
import numpy as np
import json
import time
from typing import List, Dict, Any, Optional, Union, Tuple
from pathlib import Path
import requests
from datetime import datetime

try:
    from datasets import Dataset, load_dataset
    from huggingface_hub import HfApi, HfFolder, Repository, create_repo
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False
    logging.warning("Hugging Face libraries not available. Install with: pip install datasets huggingface_hub")
    
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CloudVectorStore:
    """A vector store that uses Hugging Face Datasets for storage. 
    Provides a cost-effective alternative to storing vectors locally.
    """
    
    def __init__(self, dataset_name: str = "medical-device-regs", username: Optional[str] = None,embedding_dim: int = 384, local_cache_dir: str = ".cache/vector_store", use_auth_token: Optional[str] = None):
        """
        Initialize cloud vector store.
        
        Args:
            dataset_name: Name of the Hugging Face dataset
            username: Hugging Face username (if None, uses env var HF_USERNAME)
            embedding_dim: Dimension of the embeddings
            local_cache_dir: Directory for local caching
            use_auth_token: Hugging Face token (if None, uses env var HF_TOKEN)
        """
        if not HUGGINGFACE_AVAILABLE:
            raise ImportError("Hugging Face libraries required. Install with: pip install datasets huggingface_hub")
            
        self.dataset_name = dataset_name
        self.username = username or os.environ.get("HF_USERNAME")
        self.embedding_dim = embedding_dim
        self.local_cache_dir = local_cache_dir
        self.token = use_auth_token or os.environ.get("HF_TOKEN") or HfFolder.get_token()
        
        if self.username:
            self.full_dataset_name = f"{self.username}/{self.dataset_name}"
        else:
            logger.warning("No username provided. Will use local mode only.")
            self.full_dataset_name = None

        os.makedirs(local_cache_dir, exist_ok=True)
        self.metadata_cache_path = os.path.join(local_cache_dir, "metadata.json")
        self.vectors_cache_path = os.path.join(local_cache_dir, "vectors.npz")

        self.doc_metadata = {}
        self.vectors = {}
        self.id_to_index = {}
        self.doc_count = 0

        self._load_local_cache()

        self.api = HfApi(token=self.token)

        self._initialize_dataset()
        
        logger.info(f"Initialized cloud vector store with {self.doc_count} documents")
        
    def _initialize_dataset(self):
        """Initialize the dataset on HF or locally."""
        if self.doc_count > 0:
            logger.info(f"Using local cache with {self.doc_count} documents")
            return
            
        if self.full_dataset_name:
            try:
                logger.info(f"Trying to load dataset: {self.full_dataset_name}")
                self.dataset = load_dataset(self.full_dataset_name)
                if isinstance(self.dataset, dict) and "train" in self.dataset:
                    self.dataset = self.dataset["train"]
                
                logger.info(f"Loaded dataset with {len(self.dataset)} rows")

                self._update_local_from_dataset()
                return
            except Exception as e:
                logger.warning(f"Could not load dataset from Hugging Face: {e}")
                logger.info("Will create a new dataset")

        self._create_new_dataset()

    def _create_new_dataset(self):
        """Create a new empty dataset."""
        try:
            empty_data = {
                "id": [],
                "embedding": [],
                "metadata": []
            }
            
            self.dataset = Dataset.from_dict(empty_data)
            
            if self.full_dataset_name and self.token:
                try:
                    create_repo(
                        repo_id=self.full_dataset_name,
                        token=self.token,
                        repo_type="dataset",
                        exist_ok=True
                    )
                    logger.info(f"Created dataset repository: {self.full_dataset_name}")
                except Exception as e:
                    logger.warning(f"Could not create dataset repository: {e}")
            
            logger.info("Created new empty dataset")
        except Exception as e:
            logger.error(f"Error creating new dataset: {e}")
            self.dataset = None
            
    def _load_local_cache(self):
        """Load cached vectors and metadata from disk."""
        if os.path.exists(self.metadata_cache_path):
            try:
                with open(self.metadata_cache_path, 'r') as f:
                    self.doc_metadata = json.load(f)
                logger.info(f"Loaded metadata for {len(self.doc_metadata)} documents from cache")
            except Exception as e:
                logger.warning(f"Error loading metadata cache: {e}")

        if os.path.exists(self.vectors_cache_path):
            try:
                data = np.load(self.vectors_cache_path, allow_pickle=True)
                self.vectors = data['vectors'].item() if 'vectors' in data else {}
                self.id_to_index = data['id_to_index'].item() if 'id_to_index' in data else {}
                self.doc_count = len(self.vectors)
                logger.info(f"Loaded {self.doc_count} vectors from cache")
            except Exception as e:
                logger.warning(f"Error loading vectors cache: {e}")
            
    def _save_local_cache(self):
        """Save vectors and metadata to local cache."""
        try:
            with open(self.metadata_cache_path, 'w') as f:
                json.dump(self.doc_metadata, f)
        except Exception as e:
            logger.warning(f"Error saving metadata cache: {e}")
        
        # Save vectors
        try:
            np.savez_compressed(
                self.vectors_cache_path,
                vectors=self.vectors,
                id_to_index=self.id_to_index
            )
        except Exception as e:
            logger.warning(f"Error saving vectors cache: {e}")
            
    def _update_local_from_dataset(self):
        """Update local cache from HF dataset."""
        if not hasattr(self, 'dataset') or self.dataset is None:
            return
            
        try:
            data_dict = self.dataset.to_dict()
            
            self.doc_metadata = {}
            self.vectors = {}
            self.id_to_index = {}
            
            for i, doc_id in enumerate(data_dict["id"]):
                embedding_list = data_dict["embedding"][i]
                embedding = np.array(embedding_list)

                metadata = data_dict["metadata"][i]

                self.vectors[doc_id] = embedding
                self.doc_metadata[doc_id] = metadata
                self.id_to_index[doc_id] = i
            
            self.doc_count = len(self.vectors)

            self._save_local_cache()
            
            logger.info(f"Updated local cache with {self.doc_count} documents from dataset")
        except Exception as e:
            logger.error(f"Error updating local cache from dataset: {e}")

    def _push_to_hub(self):
        """Push the dataset to Hugging Face Hub."""
        if not self.full_dataset_name or not self.token:
            logger.warning("Cannot push to Hub: missing username or token")
            return False
            
        if not hasattr(self, 'dataset') or self.dataset is None:
            logger.warning("Cannot push to Hub: no dataset loaded")
            return False
            
        try:
            self.dataset.push_to_hub(
                self.full_dataset_name,
                token=self.token,
                private=False
            )
            logger.info(f"Successfully pushed dataset to {self.full_dataset_name}")
            return True
        except Exception as e:
            logger.error(f"Error pushing dataset to Hub: {e}")
            return False

    def add(self, doc_id: str, embedding: np.ndarray, metadata: Dict[str, Any]) -> bool:
        """
        Add a document embedding to the vector store.
        
        Args:
            doc_id: Document ID
            embedding: Document embedding
            metadata: Document metadata
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if isinstance(embedding, list):
                embedding = np.array(embedding, dtype=np.float32)

            if embedding.ndim == 2 and embedding.shape[0] == 1:
                embedding = embedding.flatten()

            self.vectors[doc_id] = embedding
            self.doc_metadata[doc_id] = metadata
            self.id_to_index[doc_id] = self.doc_count
            self.doc_count += 1

            self._save_local_cache()

            if hasattr(self, 'dataset') and self.dataset is not None:
                embedding_list = embedding.tolist()

                new_data = {
                    "id": [doc_id],
                    "embedding": [embedding_list],
                    "metadata": [metadata]
                }
                
                new_row = Dataset.from_dict(new_data)
                self.dataset = self.dataset.add_item(new_data)

                if self.full_dataset_name and self.token:
                    self._push_to_hub()
            
            return True
        except Exception as e:
            logger.error(f"Error adding document: {e}")
            return False
        
    def add_many(self, docs: List[Dict[str, Any]], id_key: str = "id", 
                 embedding_key: str = "embedding", metadata_key: str = "metadata") -> int:
        """
        Add multiple document embeddings to the vector store.
        
        Args:
            docs: List of document dictionaries
            id_key: Key for document ID
            embedding_key: Key for embedding
            metadata_key: Key for metadata
            
        Returns:
            Number of documents successfully added
        """
        if not docs:
            return 0
            
        try:
            ids = []
            embeddings = []
            metadatas = []
            
            for doc in docs:
                if id_key not in doc or embedding_key not in doc:
                    logger.warning(f"Document missing required keys ({id_key} or {embedding_key})")
                    continue
                
                doc_id = doc[id_key]
                embedding = doc[embedding_key]

                if metadata_key in doc:
                    metadata = doc[metadata_key]
                else:
                    metadata = {k: v for k, v in doc.items() if k != embedding_key}

                if isinstance(embedding, list):
                    embedding = np.array(embedding)
                
                self.vectors[doc_id] = embedding
                self.doc_metadata[doc_id] = metadata
                self.id_to_index[doc_id] = self.doc_count + len(ids)

                ids.append(doc_id)
                embeddings.append(embedding.tolist())  
                metadatas.append(metadata)

            self.doc_count += len(ids)

            self._save_local_cache()

            if hasattr(self, 'dataset') and self.dataset is not None and ids:
                new_data = {
                    "id": ids,
                    "embedding": embeddings,
                    "metadata": metadatas
                }
                
                self.dataset = self.dataset.add_items(new_data)

                if self.full_dataset_name and self.token:
                    self._push_to_hub()
            
            return len(ids)
        except Exception as e:
            logger.error(f"Error adding documents in batch: {e}")
            return 0
        
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar documents.
        
        Args:
            query_embedding: Query embedding
            top_k: Number of results to return
            
        Returns:
            List of dictionaries with document IDs, scores, and metadata
        """
        if self.doc_count == 0:
            logger.warning("Vector store is empty")
            return []
        
        try:
            if isinstance(query_embedding, list):
                query_embedding = np.array(query_embedding)
            
            if query_embedding.ndim == 2 and query_embedding.shape[0] == 1:
                query_embedding = query_embedding.flatten()

            similarities = {}
            for doc_id, vector in self.vectors.items():
                query_norm = np.linalg.norm(query_embedding)
                vector_norm = np.linalg.norm(vector)
                
                if query_norm == 0 or vector_norm == 0:
                    similarity = 0
                else:
                    similarity = np.dot(query_embedding, vector) / (query_norm * vector_norm)
                
                similarities[doc_id] = similarity

            sorted_ids = sorted(similarities.keys(), key=lambda x: similarities[x], reverse=True)

            results = []
            for doc_id in sorted_ids[:top_k]:
                metadata = self.doc_metadata.get(doc_id, {})
                
                results.append({
                    "id": doc_id,
                    "score": float(similarities[doc_id]),
                    "metadata": metadata
                })
            
            return results
        except Exception as e:
            logger.error(f"Error searching vector store: {e}")
            return []
        
    def get(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a document by ID.
        
        Args:
            doc_id: Document ID
            
        Returns:
            Document metadata or None if not found
        """
        return self.doc_metadata.get(doc_id)
    
    def delete(self, doc_id: str) -> bool:
        """
        Delete a document from the vector store.
        
        Args:
            doc_id: Document ID
            
        Returns:
            True if successful, False otherwise
        """
        if doc_id in self.vectors:
            del self.vectors[doc_id]
            del self.doc_metadata[doc_id]
            if doc_id in self.id_to_index:
                del self.id_to_index[doc_id]
            
            self.doc_count = len(self.vectors)
            self._save_local_cache()
            
            logger.info(f"Deleted document {doc_id} (from local cache only)")
            return True
        
        return False
    
    def stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store.
        
        Returns:
            Dictionary of statistics
        """
        return {
            "document_count": self.doc_count,
            "embedding_dimension": self.embedding_dim,
            "storage_mode": "cloud" if self.full_dataset_name else "local",
            "dataset_name": self.full_dataset_name or "local-only",
            "local_cache_size_mb": self._get_cache_size_mb()
        }
        
    def _get_cache_size_mb(self) -> float:
        """Get the size of the local cache in MB."""
        total_size = 0
        
        if os.path.exists(self.metadata_cache_path):
            total_size += os.path.getsize(self.metadata_cache_path)
        
        if os.path.exists(self.vectors_cache_path):
            total_size += os.path.getsize(self.vectors_cache_path)
        
        return total_size / (1024 * 1024)
    
    def clear_cache(self) -> bool:
        """
        Clear the local cache.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            if os.path.exists(self.metadata_cache_path):
                os.remove(self.metadata_cache_path)
            
            if os.path.exists(self.vectors_cache_path):
                os.remove(self.vectors_cache_path)

            self.doc_metadata = {}
            self.vectors = {}
            self.id_to_index = {}
            self.doc_count = 0
            
            logger.info("Cleared local cache")
            return True
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            return False
        
    @classmethod
    def from_existing(cls, dataset_name: str, username: str, local_cache_dir: str = None) -> 'CloudVectorStore':
        """
        Create a CloudVectorStore from an existing Hugging Face dataset.
        
        Args:
            dataset_name: Name of the existing dataset
            username: Hugging Face username
            local_cache_dir: Directory for local caching
            
        Returns:
            CloudVectorStore instance
        """
        store = cls(
            dataset_name=dataset_name,
            username=username,
            local_cache_dir=local_cache_dir or f".cache/vector_store/{dataset_name}"
        )
        return store
    
    def synchronize(self) -> bool:
        """
        Synchronize local cache with the cloud dataset.
        
        Returns:
            True if successful, False otherwise
        """
        if not self.full_dataset_name:
            logger.warning("Cannot synchronize: no dataset name")
            return False
            
        try:
            self.dataset = load_dataset(self.full_dataset_name, force_download=True)
            if isinstance(self.dataset, dict) and "train" in self.dataset:
                self.dataset = self.dataset["train"]

            self._update_local_from_dataset()
            return True
        except Exception as e:
            logger.error(f"Error synchronizing with dataset: {e}")
            return False