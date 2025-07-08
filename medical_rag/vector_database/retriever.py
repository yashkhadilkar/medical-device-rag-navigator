#!/usr/bin/env python
"""
Clean, working DocumentRetriever that fixes all the variable scope and indentation issues.
Simplified version focusing on core retrieval functionality.
"""

import os
import json
import logging
import time
from typing import Dict, List, Any, Optional
import numpy as np

logger = logging.getLogger(__name__)

class DocumentRetriever:
    """Document retriever for finding relevant content from vector store."""
    
    def __init__(self, 
                 vector_store,
                 embedding_generator,
                 top_k: int = 5,
                 min_score: float = 0.1,
                 max_chars_per_doc: int = 8000,
                 cache_dir: str = ".cache/retriever"):
        """
        Initialize the document retriever.
        
        Args:
            vector_store: Vector store for document storage
            embedding_generator: Embedding generator for query embeddings
            top_k: Number of top documents to retrieve
            min_score: Minimum similarity score threshold
            max_chars_per_doc: Maximum characters per document
            cache_dir: Directory for caching results
        """
        self.vector_store = vector_store
        self.embedding_generator = embedding_generator
        self.top_k = top_k
        self.min_score = min_score
        self.max_chars_per_doc = max_chars_per_doc
        self.cache_dir = cache_dir
        
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        
        logger.info(f"Initialized document retriever with top_k={top_k}, min_score={min_score}, max_chars_per_doc={max_chars_per_doc}")
    
    def retrieve(self, query: str, top_k: Optional[int] = None, min_score: Optional[float] = None, 
                 filters: Optional[Dict[str, Any]] = None, use_cache: bool = True) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: Search query
            top_k: Number of documents to retrieve (overrides default)
            min_score: Minimum similarity score (overrides default)
            filters: Additional filters to apply
            use_cache: Whether to use caching
            
        Returns:
            List of relevant documents with metadata
        """
        if top_k is None:
            top_k = self.top_k
        if min_score is None:
            min_score = self.min_score
        
        logger.info(f"🎯 Query Type: testing")
        logger.info(f"🔍 Keywords: {self._extract_keywords(query)}")
        
        try:
            search_results = self.vector_store.search(query, top_k=top_k * 2) 
            
            logger.info(f"🔍 Initial search returned {len(search_results)} results for query: '{query}'")

            results = []
            for i, result in enumerate(search_results):
                score = result.get('score', result.get('similarity', 0))
                
                if score < min_score:
                    continue

                processed_result = {
                    'id': result.get('id', f'doc_{i}'),
                    'title': result.get('title', 'No title'),
                    'text': result.get('text', ''),
                    'category': result.get('category', 'unknown'),
                    'score': score,
                    'similarity': score,  
                    'metadata': {
                        'title': result.get('title', 'No title'),
                        'category': result.get('category', 'unknown'),
                        'file_path': result.get('file_path', ''),
                        'text_length': result.get('text_length', len(result.get('text', ''))),
                        'extraction_method': result.get('extraction_method', 'unknown')
                    }
                }

                if len(processed_result['text']) > self.max_chars_per_doc:
                    processed_result['text'] = processed_result['text'][:self.max_chars_per_doc] + '...'
                
                results.append(processed_result)
                
                logger.info(f"  {i+1}. {processed_result['id']}: {processed_result['title']} (score: {score:.3f})")
                
                if len(results) >= top_k:
                    break
            
            logger.info(f"Retrieved {len(results)} documents for query: '{query}'")
            return results
            
        except Exception as e:
            logger.error(f"Error during retrieval: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Extract keywords from query for logging."""
        keywords = []
        query_lower = query.lower()

        medical_keywords = [
            'biocompatibility', 'iso 10993', 'implant', 'sterilization', 
            'cytotoxicity', 'sensitization', '510k', 'pma', 'classification',
            'software', 'cybersecurity', 'validation', 'testing'
        ]
        
        for keyword in medical_keywords:
            if keyword in query_lower:
                keywords.append(keyword)
        
        return keywords[:5]  
    
    def get_stats(self) -> Dict[str, Any]:
        """Get retriever statistics."""
        return {
            'top_k': self.top_k,
            'min_score': self.min_score,
            'max_chars_per_doc': self.max_chars_per_doc,
            'cache_dir': self.cache_dir,
            'vector_store_docs': getattr(self.vector_store, 'doc_count', 0)
        }