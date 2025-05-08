import os 
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple, Callable
import json
import re
from pathlib import Path
from datetime import datetime
import time
import hashlib
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))  # rag_implementation dir
medical_rag_dir = os.path.dirname(script_dir)            # medical_rag dir
project_root = os.path.dirname(medical_rag_dir)          # project root
sys.path.append(project_root)     

from medical_rag.vector_database.embeddings import EmbeddingGenerator
from medical_rag.vector_database.cloud_store import CloudVectorStore

# Set this before importing any Hugging Face libraries
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DocumentRetriever:
    """Retrieve relevant documents for a given query. 
    Optimized for cloud storage and minimal resource usage.
    """
    
    def __init__(self, vector_store: CloudVectorStore, embedding_generator: EmbeddingGenerator, top_k: int = 5, min_score: float = 0.5, cache_dir: Optional[str] = ".cache/retriever"):
        """Initialize the document retriever.

        Args:
            vector_store: Vector store containing document embeddings
            embedding_generator: Generator for query embeddings
            top_k: Default number of documents to retrieve
            min_score: Minimum similarity score for retrieved documents
            cache_dir: Directory for caching query results
        """
        self.vector_store = vector_store
        self.embedding_generator = embedding_generator
        self.top_k = top_k
        self.min_score = min_score
        self.cache_dir = cache_dir

        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

        self.query_cache = {}
        self.cache_ttl = 3600 
        
        logger.info(f"Initialized document retriever with top_k={top_k}, min_score={min_score}")
        
    def _get_query_hash(self, query: str, filter_str: str = "") -> str:
        """
        Compute a hash for the query and filters to use as cache key.
        
        Args:
            query: Query string
            filter_str: String representation of filters
            
        Returns:
            Hash string
        """
        combined = query.lower() + "|" + filter_str
        return hashlib.md5(combined.encode()).hexdigest()
        
    def _check_cache(self, query: str, filters: Optional[Dict[str, Any]] = None) -> Optional[List[Dict[str, Any]]]:
        """
        Check if query results are in cache.
        
        Args:
            query: Query string
            filters: Filters to apply
            
        Returns:
            Cached results or None
        """
        if not self.cache_dir:
            return None

        filter_str = json.dumps(filters) if filters else ""
        query_hash = self._get_query_hash(query, filter_str)

        if query_hash in self.query_cache:
            cache_time, results = self.query_cache[query_hash]
            if time.time() - cache_time < self.cache_ttl:
                logger.info(f"Retrieved results from in-memory cache for query: '{query}'")
                return results

        cache_path = os.path.join(self.cache_dir, f"{query_hash}.json")
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r') as f:
                    cache_data = json.load(f)

                if time.time() - cache_data.get("timestamp", 0) < self.cache_ttl:
                    logger.info(f"Retrieved results from disk cache for query: '{query}'")
                    self.query_cache[query_hash] = (cache_data["timestamp"], cache_data["results"])
                    return cache_data["results"]
            except Exception as e:
                logger.warning(f"Error reading cache: {e}")
                
        return None
    
    def _save_to_cache(self, query: str, results: List[Dict[str, Any]], filters: Optional[Dict[str, Any]] = None) -> None:
        """
        Save query results to cache.
        
        Args:
            query: Query string
            results: Retrieved results
            filters: Filters applied
        """
        if not self.cache_dir:
            return

        filter_str = json.dumps(filters) if filters else ""
        query_hash = self._get_query_hash(query, filter_str)

        timestamp = time.time()
        self.query_cache[query_hash] = (timestamp, results)
        cache_path = os.path.join(self.cache_dir, f"{query_hash}.json")
        
        try:
            cache_data = {
                "query": query,
                "filters": filters,
                "timestamp": timestamp,
                "results": results
            }
            
            with open(cache_path, 'w') as f:
                json.dump(cache_data, f)
        except Exception as e:
            logger.warning(f"Error writing to cache: {e}")
            
    def retrieve(self, query: str, top_k: Optional[int] = None, 
                min_score: Optional[float] = None, filters: Optional[Dict[str, Any]] = None,
                use_cache: bool = True) -> List[Dict[str, Any]]:
        """
        Retrieve documents for a query.
        
        Args:
            query: Query string
            top_k: Number of documents to retrieve (overrides default)
            min_score: Minimum similarity score (overrides default)
            filters: Filters to apply to retrieved documents
            use_cache: Whether to use cached results
            
        Returns:
            List of retrieved documents with scores and metadata
        """
        if not query.strip():
            logger.warning("Empty query provided")
            return []

        top_k = top_k if top_k is not None else self.top_k
        min_score = min_score if min_score is not None else self.min_score

        if use_cache:
            cached_results = self._check_cache(query, filters)
            if cached_results:
                filtered_results = [r for r in cached_results if r["score"] >= min_score]
                return filtered_results[:top_k]

        query_embedding = self.embedding_generator.embed_text(query)

        results = self.vector_store.search(query_embedding, top_k=min(top_k * 2, 20)) 

        if filters:
            results = self._apply_filters(results, filters)

        results = [r for r in results if r["score"] >= min_score]

        # Debug: check if results have actual text content
        for i, result in enumerate(results):
            text = result.get("text", "")
            logger.info(f"Retrieved document {i+1} text length: {len(text)}")
            if len(text) < 50:  # If text is suspiciously short
                logger.warning(f"Document has very short text: {text}")
                
                # Check if metadata contains text
                metadata = result.get("metadata", {})
                if "text" in metadata and len(metadata["text"]) > len(text):
                    logger.info(f"Found longer text in metadata ({len(metadata['text'])} chars). Using that instead.")
                    result["text"] = metadata["text"]

        results = results[:top_k]

        for result in results:
            result["query"] = query

        if use_cache:
            self._save_to_cache(query, results, filters)
        
        logger.info(f"Retrieved {len(results)} documents for query: '{query}'")
        return results
    
    def _apply_filters(self, results: List[Dict[str, Any]], filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Apply filters to retrieved documents.
        
        Args:
            results: List of retrieval results
            filters: Filters to apply
            
        Returns:
            Filtered results
        """
        filtered_results = []
        
        for result in results:
            metadata = result.get("metadata", {})
            include = True
            
            for key, value in filters.items():
                if key not in metadata:
                    include = False
                    break
                
                if isinstance(value, list):
                    if metadata[key] not in value:
                        include = False
                        break
                elif isinstance(value, dict) and "range" in value:
                    range_min = value["range"].get("min")
                    range_max = value["range"].get("max")
                    
                    if range_min is not None and metadata[key] < range_min:
                        include = False
                        break
                    
                    if range_max is not None and metadata[key] > range_max:
                        include = False
                        break
                elif isinstance(value, dict) and "regex" in value:
                    if not re.search(value["regex"], str(metadata[key])):
                        include = False
                        break
                elif metadata[key] != value:
                    include = False
                    break
            
            if include:
                filtered_results.append(result)
        
        return filtered_results
    
    def _extract_keywords(self, query: str) -> List[str]:
        """
        Extract keywords from a query.
        
        Args:
            query: Query string
            
        Returns:
            List of keywords
        """
        stopwords = {'a', 'an', 'the', 'and', 'or', 'but', 'is', 'are', 'be', 'was', 'were', 
                    'in', 'of', 'to', 'for', 'with', 'by', 'at', 'on', 'from', 'about'}
        
        words = re.findall(r'\b\w+\b', query.lower())
        keywords = [word for word in words if word not in stopwords and len(word) > 2]
        
        return keywords
    
    def hybrid_retrieve(self, query: str, keyword_weight: float = 0.3, 
                       top_k: Optional[int] = None, filters: Optional[Dict[str, Any]] = None,
                       use_cache: bool = True) -> List[Dict[str, Any]]:
        """
        Hybrid retrieval combining vector similarity with keyword matching.
        
        Args:
            query: Query string
            keyword_weight: Weight for keyword matching (0-1)
            top_k: Number of documents to retrieve
            filters: Filters to apply
            use_cache: Whether to use cached results
            
        Returns:
            List of retrieved documents with scores
        """
        if use_cache:
            cache_key = f"hybrid_{keyword_weight}_{query}"
            cached_results = self._check_cache(cache_key, filters)
            if cached_results:
                return cached_results[:top_k if top_k else self.top_k]

        vector_results = self.retrieve(query, top_k=min(top_k * 2 if top_k else self.top_k * 2, 20), 
                                    filters=filters, use_cache=use_cache)

        keywords = self._extract_keywords(query)

        if keywords:
            for result in vector_results:
                metadata = result.get("metadata", {})
                text = result.get("text", "")
                
                # Make sure text has content
                if len(text) < 50 and "text" in metadata:
                    text = metadata["text"]
                
                keyword_score = self._calculate_keyword_score(text, keywords)

                vector_score = result["score"]
                result["vector_score"] = vector_score
                result["keyword_score"] = keyword_score
                result["score"] = (1 - keyword_weight) * vector_score + keyword_weight * keyword_score

        vector_results.sort(key=lambda x: x["score"], reverse=True)
        vector_results = vector_results[:top_k if top_k else self.top_k]

        if use_cache:
            cache_key = f"hybrid_{keyword_weight}_{query}"
            self._save_to_cache(cache_key, vector_results, filters)
        
        return vector_results
    
    def _calculate_keyword_score(self, text: str, keywords: List[str]) -> float:
        """
        Calculate keyword match score for a text.
        
        Args:
            text: Text to score
            keywords: List of keywords
            
        Returns:
            Score between 0 and 1
        """
        if not text or not keywords:
            return 0.0
        
        text_lower = text.lower()

        matches = 0
        for keyword in keywords:
            matches += len(re.findall(r'\b' + re.escape(keyword) + r'\b', text_lower))

        word_count = len(re.findall(r'\b\w+\b', text_lower))
        if word_count == 0:
            return 0.0
        
        density = matches / word_count
        score = min(1.0, density * 10)
        
        return score
    
    def clear_cache(self):
        """
        Clear the query cache to free memory and disk space.
        """
        self.query_cache = {}

        if self.cache_dir and os.path.exists(self.cache_dir):
            try:
                cache_files = list(Path(self.cache_dir).glob("*.json"))
                for file_path in cache_files:
                    try:
                        os.remove(file_path)
                    except Exception as e:
                        logger.warning(f"Error removing cache file {file_path}: {e}")
                
                logger.info(f"Cleared {len(cache_files)} files from query cache")
            except Exception as e:
                logger.error(f"Error clearing cache: {e}")
                
    def save_query_history(self, query: str, results: List[Dict[str, Any]], history_file: str = "data/query_history.jsonl") -> bool:
        """
        Save query and results to history file.
        Keeps history file small by storing minimal information.
        
        Args:
            query: Query string
            results: Retrieved results
            history_file: Path to history file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            os.makedirs(os.path.dirname(history_file), exist_ok=True)
            entry = {
                "timestamp": datetime.now().isoformat(),
                "query": query,
                "result_count": len(results),
                "top_ids": [r.get("id", "unknown") for r in results[:3]]
            }

            with open(history_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(entry) + '\n')
            
            return True

        except Exception as e:
            logger.error(f"Error saving query history: {e}")
            return False
        
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get retriever statistics.
        
        Returns:
            Dictionary of statistics
        """
        cache_count = len(self.query_cache)
        cache_size = 0
        if self.cache_dir and os.path.exists(self.cache_dir):
            cache_files = list(Path(self.cache_dir).glob("*.json"))
            cache_size = sum(os.path.getsize(f) for f in cache_files) / (1024 * 1024)  # MB
        
        return {
            "vector_store_stats": self.vector_store.stats(),
            "embedding_model": self.embedding_generator.model_name,
            "embedding_dim": self.embedding_generator.embedding_dim,
            "top_k": self.top_k,
            "min_score": self.min_score,
            "cache_entries": cache_count,
            "cache_size_mb": round(cache_size, 2)
        }
        
    def retrieve_by_id(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a specific document by ID.
        
        Args:
            doc_id: Document ID
            
        Returns:
            Document metadata or None if not found
        """
        return self.vector_store.get(doc_id)