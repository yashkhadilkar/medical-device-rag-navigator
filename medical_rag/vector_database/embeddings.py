import os 
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import json 
import time
import hashlib

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logging.warning("sentence-transformers not available. Install with: pip install sentence-transformers")

# Importing transformers for fallback
try:
    import torch
    from transformers import AutoTokenizer, AutoModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("transformers not available. Install with: pip install transformers torch")


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Generate embeddings for text chunks using various embedding models.
    Optimized for minimal resource usage and cloud compatibility.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", cache_dir: Optional[str] = "data/embeddings_cache", device: str = "cpu", use_api: bool = False, api_key: Optional[str] = None):
        """
        Initialize the embedding generator.
        
        Args:
            model_name: Name of the embedding model to use
            cache_dir: Directory to cache embeddings
            device: Device to run the model on ('cpu' or 'cuda')
            use_api: Whether to use remote API for embeddings (not yet implemented)
            api_key: API key for remote embedding service (not yet implemented)
        """
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.device = device
        self.use_api = use_api
        self.api_key = api_key
        
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

        self.model_loaded = False

        self.embedding_dim = 384 

        self.model = None
        
        logger.info(f"Initialized embedding generator with model {model_name} (lazy loading)")
        
        
    def _load_model(self):
        """Load the embedding model (called on first use)."""
        if self.model_loaded:
            return
            
        try: 
            logger.info(f"Loading embedding model: {self.model_name}")
            
            if SENTENCE_TRANSFORMERS_AVAILABLE:
                self.model = SentenceTransformer(self.model_name, device=self.device)
                self.embedding_dim = self.model.get_sentence_embedding_dimension()
                logger.info(f"Loaded sentence-transformers model. Embedding dimension: {self.embedding_dim}")
            elif TRANSFORMERS_AVAILABLE:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
                with torch.no_grad():
                    test_input = self.tokenizer("test", return_tensors="pt").to(self.device)
                    test_output = self.model(**test_input)
                    self.embedding_dim = test_output.last_hidden_state.shape[-1]
                logger.info(f"Loaded transformers model. Embedding dimension: {self.embedding_dim}")
            else:
                logger.error("No embedding libraries available. Install sentence-transformers or transformers.")
                return
                
            self.model_loaded = True
            
        except Exception as e:
            logger.error(f"Error loading embedding model: {e}")
            raise
        
    def _get_cache_path(self, text_hash: str) -> str:
        """
        Get path for cached embedding.
        
        Args:
            text_hash: Hash of the text
            
        Returns:
            Path to the cached embedding file
        """
        if not self.cache_dir:
            return None
            
        return os.path.join(self.cache_dir, f"{text_hash}_{self.model_name}.npy")
    
    def _compute_text_hash(self, text: str) -> str:
        """
        Compute hash for text to use as cache key.
        
        Args:
            text: Input text
            
        Returns:
            Hash string
        """
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def _mean_pooling(self, model_output, attention_mask):
        """
        Perform mean pooling for transformers model output.
        
        Args:
            model_output: Output from the model
            attention_mask: Attention mask from tokenizer
            
        Returns:
            Pooled embeddings
        """
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def _generate_embedding_with_transformers(self, text: str) -> np.ndarray:
        """
        Generate embedding using transformers library.
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector
        """
        encoded_input = self.tokenizer(text, padding=True, truncation=True, 
                                     max_length=512, return_tensors='pt').to(self.device)
        
        with torch.no_grad():
            model_output = self.model(**encoded_input)

        sentence_embeddings = self._mean_pooling(model_output, encoded_input['attention_mask'])
        sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
        
        return sentence_embeddings.cpu().numpy()[0]
    
    def embed_text(self, text: str, use_cache: bool = True) -> np.ndarray:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            use_cache: Whether to use cached embeddings
            
        Returns:
            Embedding vector as numpy array
        """
        if not text.strip():
            logger.warning("Empty text provided for embedding")
            return np.zeros(self.embedding_dim)

        text_hash = self._compute_text_hash(text)
        cache_path = self._get_cache_path(text_hash) if use_cache and self.cache_dir else None

        if use_cache and cache_path and os.path.exists(cache_path):
            try:
                embedding = np.load(cache_path)
                return embedding
            except Exception as e:
                logger.warning(f"Error loading cached embedding: {e}")

        if not self.model_loaded:
            self._load_model()

        try:
            if hasattr(self, 'model') and self.model is not None:
                if SENTENCE_TRANSFORMERS_AVAILABLE and isinstance(self.model, SentenceTransformer):
                    embedding = self.model.encode(text, normalize_embeddings=True)
                elif TRANSFORMERS_AVAILABLE:
                    embedding = self._generate_embedding_with_transformers(text)
                else:
                    logger.error("No valid model available for embedding generation")
                    return np.zeros(self.embedding_dim)

                if cache_path:
                    np.save(cache_path, embedding)
                    
                return embedding
            else:
                logger.error("Model not properly initialized")
                return np.zeros(self.embedding_dim)
                
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return np.zeros(self.embedding_dim)
        
    def embed_texts(self, texts: List[str], batch_size: int = 8, 
                   use_cache: bool = True, show_progress: bool = True) -> List[np.ndarray]:
        """
        Generate embeddings for multiple texts.
        Uses smaller batch size to reduce memory requirements.
        
        Args:
            texts: List of texts to embed
            batch_size: Batch size for embedding generation (smaller for RAM efficiency)
            use_cache: Whether to use cached embeddings
            show_progress: Whether to show progress
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
            
        embeddings = []
        texts_to_embed = []
        texts_to_embed_indices = []

        if use_cache and self.cache_dir:
            for i, text in enumerate(texts):
                if not text.strip():
                    embeddings.append(np.zeros(self.embedding_dim))
                    continue
                    
                text_hash = self._compute_text_hash(text)
                cache_path = self._get_cache_path(text_hash)
                
                if os.path.exists(cache_path):
                    try:
                        embedding = np.load(cache_path)
                        embeddings.append(embedding)
                    except Exception as e:
                        logger.warning(f"Error loading cached embedding: {e}")
                        texts_to_embed.append(text)
                        texts_to_embed_indices.append(i)
                else:
                    texts_to_embed.append(text)
                    texts_to_embed_indices.append(i)
        else:
            texts_to_embed = texts
            texts_to_embed_indices = list(range(len(texts)))

        if texts_to_embed and not self.model_loaded:
            self._load_model()

        if texts_to_embed:
            logger.info(f"Generating embeddings for {len(texts_to_embed)} texts")

            if len(embeddings) < len(texts):
                embeddings = [None] * len(texts)

            for i in range(0, len(texts_to_embed), batch_size):
                batch_texts = texts_to_embed[i:i+batch_size]
                
                try:
                    if SENTENCE_TRANSFORMERS_AVAILABLE and isinstance(self.model, SentenceTransformer):
                        batch_embeddings = self.model.encode(batch_texts, normalize_embeddings=True)
                    elif TRANSFORMERS_AVAILABLE:
                        batch_embeddings = []
                        for text in batch_texts:
                            embedding = self._generate_embedding_with_transformers(text)
                            batch_embeddings.append(embedding)
                        batch_embeddings = np.array(batch_embeddings)
                    else:
                        logger.error("No valid model available for embedding generation")
                        batch_embeddings = [np.zeros(self.embedding_dim) for _ in batch_texts]

                    for j, embedding in enumerate(batch_embeddings):
                        idx = texts_to_embed_indices[i+j]
                        embeddings[idx] = embedding
                        
                        if self.cache_dir:
                            text_hash = self._compute_text_hash(texts_to_embed[i+j])
                            cache_path = self._get_cache_path(text_hash)
                            np.save(cache_path, embedding)
                    
                    if show_progress and i % (batch_size * 2) == 0:
                        logger.info(f"Processed {i}/{len(texts_to_embed)} texts")
                        
                except Exception as e:
                    logger.error(f"Error generating batch embeddings: {e}")
                    for j in range(len(batch_texts)):
                        idx = texts_to_embed_indices[i+j]
                        if embeddings[idx] is None:
                            embeddings[idx] = np.zeros(self.embedding_dim)
        
        return embeddings
    
    def embed_chunks(self, chunks: List[Dict[str, Any]], text_key: str = "text", 
                     embedding_key: str = "embedding", batch_size: int = 8,
                     use_cache: bool = True) -> List[Dict[str, Any]]:
        """
        Generate embeddings for a list of text chunks.
        
        Args:
            chunks: List of chunk dictionaries
            text_key: Key in each chunk that contains the text
            embedding_key: Key to store the embedding in each chunk
            batch_size: Batch size for embedding generation
            use_cache: Whether to use cached embeddings
            
        Returns:
            List of chunks with embeddings added
        """
        texts = [chunk[text_key] for chunk in chunks if text_key in chunk]
        
        if len(texts) != len(chunks):
            logger.warning(f"Not all chunks contain the text key '{text_key}'")
        
        embeddings = self.embed_texts(texts, batch_size, use_cache)
        
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            chunk[embedding_key] = embedding
        
        return chunks
    
    def save_embeddings_to_file(self, chunks: List[Dict[str, Any]], output_path: str,
                              embedding_key: str = "embedding") -> bool:
        """
        Save embedded chunks to a file.
        
        Args:
            chunks: List of chunks with embeddings
            output_path: Path to save the file
            embedding_key: Key that contains the embedding in each chunk
            
        Returns:
            True if successful, False otherwise
        """
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            serializable_chunks = []
            for chunk in chunks:
                chunk_copy = chunk.copy()
                if embedding_key in chunk_copy and isinstance(chunk_copy[embedding_key], np.ndarray):
                    chunk_copy[embedding_key] = chunk_copy[embedding_key].tolist()
                serializable_chunks.append(chunk_copy)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(serializable_chunks, f, ensure_ascii=False, indent=2)
                
            logger.info(f"Saved {len(chunks)} embedded chunks to {output_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving embeddings to {output_path}: {e}")
            return False
        
    def load_embeddings_from_file(self, input_path: str, embedding_key: str = "embedding") -> List[Dict[str, Any]]:
        """
        Load embedded chunks from a file.
        
        Args:
            input_path: Path to the file
            embedding_key: Key that contains the embedding in each chunk
            
        Returns:
            List of chunks with embeddings
        """
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                chunks = json.load(f)

            for chunk in chunks:
                if embedding_key in chunk:
                    chunk[embedding_key] = np.array(chunk[embedding_key])
            
            logger.info(f"Loaded {len(chunks)} embedded chunks from {input_path}")
            return chunks
        except Exception as e:
            logger.error(f"Error loading embeddings from {input_path}: {e}")
            return []
        
    def unload_model(self):
        """
        Unload the model to free memory when not needed.
        """
        if self.model_loaded:
            self.model = None
            self.model_loaded = False

            if self.device.startswith('cuda'):
                try:
                    import torch
                    import sys
                    if 'torch' in sys.modules:
                        torch.cuda.empty_cache()
                except:
                    pass
                
            logger.info("Unloaded embedding model to free memory")
            
    def clear_cache(self):
        """
        Clear the embedding cache to free disk space.
        """
        if not self.cache_dir or not os.path.exists(self.cache_dir):
            logger.warning("No cache directory to clear")
            return
            
        try:
            cache_files = list(Path(self.cache_dir).glob("*.npy"))
            file_count = len(cache_files)

            for file_path in cache_files:
                try:
                    os.remove(file_path)
                except Exception as e:
                    logger.warning(f"Error removing cache file {file_path}: {e}")
            
            logger.info(f"Cleared {file_count} files from embedding cache")
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            
if __name__ == "__main__":
    # Initialize the embedding generator
    embedding_generator = EmbeddingGenerator(model_name="all-MiniLM-L6-v2")
    
    # Sample text chunks
    sample_chunks = [
        {"id": "1", "text": "The FDA regulates medical devices through the 510(k) premarket notification process."},
        {"id": "2", "text": "Class III medical devices require Premarket Approval (PMA) from the FDA."}
    ]
    
    # Generate embeddings for the chunks
    embedded_chunks = embedding_generator.embed_chunks(sample_chunks)
    
    # Print embedding dimensions
    for chunk in embedded_chunks:
        print(f"Chunk {chunk['id']} embedding shape: {chunk['embedding'].shape}")
        
    # Clean up memory when done
    embedding_generator.unload_model()