import os
import logging
import json
import time
import sys
from typing import List, Dict, Any, Optional, Union, Tuple
from pathlib import Path
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))  
parent_dir = os.path.dirname(current_dir)                 
grandparent_dir = os.path.dirname(parent_dir)             
sys.path.append(grandparent_dir)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from medical_rag.rag_implementation.remote_llm import LLMInterface, LLMProvider
from medical_rag.rag_implementation.query_processor import QueryProcessor, QueryType
from medical_rag.vector_database.retriever import DocumentRetriever
from medical_rag.vector_database.cloud_store import CloudVectorStore
from medical_rag.vector_database.embeddings import EmbeddingGenerator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RAGPipeline:
    """Main RAG pipeline for the Medical Device Regulation Navigator.
    Integrates retrieval, document processing, and LLM generation.
    """
    
    def __init__(self, 
                 retriever: DocumentRetriever,
                 llm_interface: LLMInterface,
                 query_processor: QueryProcessor,
                 cache_dir: str = ".cache/rag_pipeline",
                 top_k: int = 5,
                 citation_style: str = "numbered"):
        """
        Initialize the RAG pipeline.
        
        Args:
            retriever: Document retriever for finding relevant content
            llm_interface: LLM interface for generating responses
            query_processor: Query processor for optimizing queries
            cache_dir: Directory for caching responses
            top_k: Number of documents to retrieve
            citation_style: Citation style (numbered, bracketed, or footnote)
        """
        self.retriever = retriever
        self.llm_interface = llm_interface
        self.query_processor = query_processor
        self.top_k = top_k
        self.citation_style = citation_style
        
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            self.cache_dir = cache_dir
        else:
            self.cache_dir = None
            
        self.query_cache = {}
        self.max_cache_age = 24 * 60 * 60  # 24 hours
        
        logger.info(f"Initialized RAG pipeline with {self.top_k} documents per query")
        
    @classmethod
    def create_default(cls, 
                      vector_store_path: str = "data/vector_store",
                      llm_provider: str = "anthropic",
                      model_name: Optional[str] = None,
                      api_key: Optional[str] = None,
                      embedding_model: str = "all-MiniLM-L6-v2",
                      cache_dir: str = ".cache") -> 'RAGPipeline':
        """
        Create a default RAG pipeline with standard components.
        
        Args:
            vector_store_path: Path to vector store
            llm_provider: LLM provider (openai, anthropic, or huggingface)
            model_name: Model name (if None, uses provider default)
            api_key: API key (if None, reads from environment variable)
            embedding_model: Embedding model name
            cache_dir: Directory for caching
            
        Returns:
            Configured RAG pipeline
        """
        os.makedirs(cache_dir, exist_ok=True)
        vector_cache = os.path.join(cache_dir, "vectors")
        llm_cache = os.path.join(cache_dir, "llm")
        query_cache = os.path.join(cache_dir, "queries")
        rag_cache = os.path.join(cache_dir, "rag")

        embedding_generator = EmbeddingGenerator(
            model_name=embedding_model,
            cache_dir=vector_cache
        )

        vector_store = CloudVectorStore(
            dataset_name="medical-device-regs",
            local_cache_dir=vector_cache,
            embedding_dim=embedding_generator.embedding_dim
        )

        retriever = DocumentRetriever(
            vector_store=vector_store,
            embedding_generator=embedding_generator,
            cache_dir=vector_cache
        )

        llm_interface = LLMInterface(
            provider=llm_provider,
            model_name=model_name,
            api_key=api_key,
            cache_dir=llm_cache
        )

        query_processor = QueryProcessor(
            cache_dir=query_cache,
            enable_advanced_nlp=False 
        )

        return cls(
            retriever=retriever,
            llm_interface=llm_interface,
            query_processor=query_processor,
            cache_dir=rag_cache
        )
    
    def _get_cache_key(self, query: str) -> str:
        """Generate a cache key for a query."""
        import hashlib
        return hashlib.md5(query.encode()).hexdigest()
    
    def _check_cache(self, query: str) -> Optional[Dict[str, Any]]:
        """Check if a query result is in cache."""
        if not self.cache_dir:
            return None
            
        cache_key = self._get_cache_key(query)

        if cache_key in self.query_cache:
            timestamp, result = self.query_cache[cache_key]
            age = time.time() - timestamp
            if age < self.max_cache_age:
                logger.info(f"Using in-memory cache for query: '{query}'")
                return result

        cache_path = os.path.join(self.cache_dir, f"{cache_key}.json")
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    cached_data = json.load(f)
                    
                timestamp = cached_data.get("timestamp", 0)
                age = time.time() - timestamp
                
                if age < self.max_cache_age:
                    logger.info(f"Using disk cache for query: '{query}'")
                    self.query_cache[cache_key] = (timestamp, cached_data)
                    return cached_data
                    
            except Exception as e:
                logger.warning(f"Error reading cache: {e}")
                
        return None
    
    def _save_to_cache(self, query: str, result: Dict[str, Any]) -> None:
        """Save a query result to cache."""
        if not self.cache_dir:
            return
            
        cache_key = self._get_cache_key(query)
        timestamp = time.time()
        
        self.query_cache[cache_key] = (timestamp, result)

        cache_path = os.path.join(self.cache_dir, f"{cache_key}.json")
        try:
            result_copy = result.copy()
            result_copy["timestamp"] = timestamp
            result_copy["query"] = query
            
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(result_copy, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            logger.warning(f"Error saving to cache: {e}")
    
    def process_query(self, query: str, use_cache: bool = True) -> Dict[str, Any]:
        """
        Process a query through the full RAG pipeline.
        
        Args:
            query: User query
            use_cache: Whether to use cached results
            
        Returns:
            Response with answer and supporting documents
        """
        if use_cache:
            cached_result = self._check_cache(query)
            if cached_result:
                return cached_result

        query_info = self.query_processor.process_query(query)
        search_queries = query_info["search_queries"]
        
        logger.info(f"Processing query: '{query}'")
        logger.info(f"Query type: {query_info['query_type']}")

        all_results = []
        for search_query in search_queries:
            logger.info(f"Searching with query: '{search_query}'")
            results = self.retriever.retrieve(search_query, top_k=self.top_k)
            all_results.extend(results)

        unique_results = self._deduplicate_results(all_results)
        
        logger.info(f"Retrieved {len(unique_results)} unique documents")

        if not unique_results:
            empty_response = {
                "query": query,
                "answer": "I couldn't find any relevant information in the medical device regulations database. Please try a different query or provide more context.",
                "documents": [],
                "generated": time.time()
            }
            return empty_response

        unique_results.sort(key=lambda x: x["score"], reverse=True)
        top_results = unique_results[:self.top_k]

        # Debug information about document content
        for i, doc in enumerate(top_results):
            text = doc.get("text", "")
            logger.info(f"Top result {i+1} text length: {len(text)}")
            if len(text) < 50:
                logger.warning(f"Document has suspiciously short text: {text}")

        system_prompt = self._generate_system_prompt(query_info)
        answer = self.llm_interface.generate_with_context(
            question=query,
            context_docs=top_results,
            system_prompt=system_prompt
        )

        response = {
            "query": query,
            "answer": answer,
            "documents": self._format_documents(top_results),
            "generated": time.time()
        }

        if use_cache:
            self._save_to_cache(query, response)

        self.query_processor.save_query_stats(query_info, len(unique_results))
        
        return response
    
    def _deduplicate_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """De-duplicate results by document ID."""
        seen_ids = set()
        unique_results = []
        
        for result in results:
            doc_id = result.get("id", "")
            if doc_id and doc_id not in seen_ids:
                seen_ids.add(doc_id)
                unique_results.append(result)
                
        return unique_results
    
    def _generate_system_prompt(self, query_info: Dict[str, Any]) -> str:
        """Generate a system prompt based on the query type."""
        query_type = query_info["query_type"]
        
        base_prompt = """
You are a Medical Device Regulatory Assistant that helps people understand FDA regulations.
Answer questions based on the provided context information from regulatory documents.
If the context doesn't contain enough information to fully answer the question, say so clearly.
Use a helpful, professional tone and cite your sources.
"""
        
        if query_type == QueryType.CLASSIFICATION.value:
            base_prompt += """
When discussing device classification, be precise about the class (I, II, or III) and explain the regulatory implications.
If there's any ambiguity, mention that proper classification requires review of the specific device details by regulatory professionals.
"""
        elif query_type == QueryType.SUBMISSION.value:
            base_prompt += """
When discussing submission pathways, be clear about the requirements for 510(k), PMA, De Novo, or other submission types.
Emphasize that submission strategies should be determined by regulatory professionals based on the specific device.
"""
        elif query_type == QueryType.COMPLIANCE.value:
            base_prompt += """
When discussing compliance, refer to specific regulations and guidance documents.
Emphasize that compliance strategies should be developed with regulatory professionals.
"""
        elif query_type == QueryType.SOFTWARE.value:
            base_prompt += """
When discussing software and digital health, refer to the latest FDA guidance on Software as a Medical Device (SaMD).
Note that software regulations are evolving, and checking for the most recent guidance is important.
"""
            
        return base_prompt
    
    def _format_context(self, context_docs: List[Dict[str, Any]]) -> str:
        """Format context documents for the prompt."""
        formatted_context = ""
        
        for i, doc in enumerate(context_docs):
            # Make sure to include the actual text content
            text = doc.get("text", "")
            metadata = doc.get("metadata", {})
            title = metadata.get("title", f"Document {i+1}")
            source = metadata.get("source", "Unknown")
            
            # Print debug info to verify text is present
            logger.info(f"DEBUG: Document {i+1} text length: {len(text)}")
            if len(text) < 50:  # If text is suspiciously short
                logger.warning(f"Document has very short text: {text}")
            
            formatted_context += f"Document {i+1}: {title} (Source: {source})\n{text}\n\n"
            
        return formatted_context.strip()
    
    def _format_documents(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format document results for the response."""
        formatted_docs = []
        
        for i, result in enumerate(results):
            doc = {
                "id": result.get("id", f"doc_{i}"),
                "title": result.get("metadata", {}).get("title", "Untitled Document"),
                "score": result.get("score", 0.0),
                "excerpt": result.get("text", "")[:300] + "...",  
                "source": result.get("metadata", {}).get("source", "Unknown Source")
            }
            
            formatted_docs.append(doc)
            
        return formatted_docs
    
    def answer_with_citations(self, query: str, use_cache: bool = True) -> Dict[str, Any]:
        """
        Answer a query with formatted citations.
        
        Args:
            query: User query
            use_cache: Whether to use cached results
            
        Returns:
            Response with answer and citations
        """
        response = self.process_query(query, use_cache)
        
        if not response.get("documents"):
            return response
        
        answer = response.get("answer", "")
        documents = response.get("documents", [])

        if self.citation_style == "numbered":
            cited_answer, citations = self._add_numbered_citations(answer, documents)
        elif self.citation_style == "bracketed":
            cited_answer, citations = self._add_bracketed_citations(answer, documents)
        else:  # footnote
            cited_answer, citations = self._add_footnote_citations(answer, documents)
            
        response["answer"] = cited_answer
        response["citations"] = citations
        
        return response
    
    def _add_numbered_citations(self, answer: str, documents: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
        """Add numbered citations to the answer."""
        citations = []
        
        for i, doc in enumerate(documents):
            citation_marker = f"[{i+1}]"
            citation = {
                "id": i+1,
                "title": doc.get("title", "Untitled"),
                "source": doc.get("source", "Unknown Source")
            }
            citations.append(citation)

            if citation_marker not in answer:
                answer += f"\n\n{citation_marker} Relevant information can be found in: {doc.get('title', 'Untitled')}"
                
        return answer, citations
    
    def _add_bracketed_citations(self, answer: str, documents: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
        """Add bracketed citations to the answer."""
        citations = []
        
        for i, doc in enumerate(documents):
            title = doc.get("title", "Untitled")
            short_title = title[:20] + "..." if len(title) > 20 else title
            citation_marker = f"[{short_title}]"
            
            citation = {
                "id": i+1,
                "title": title,
                "source": doc.get("source", "Unknown Source")
            }
            citations.append(citation)

            if citation_marker not in answer:
                answer += f"\n\n{citation_marker} Relevant information from {doc.get('source', 'Unknown Source')}"
                
        return answer, citations
    
    def _add_footnote_citations(self, answer: str, documents: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
        """Add footnote citations to the answer."""
        citations = []
        footnotes = "\n\nSources:\n"
        
        for i, doc in enumerate(documents):
            marker = f"[{i+1}]"
            citation = {
                "id": i+1,
                "title": doc.get("title", "Untitled"),
                "source": doc.get("source", "Unknown Source")
            }
            citations.append(citation)
            
            footnotes += f"{marker} {doc.get('title', 'Untitled')} ({doc.get('source', 'Unknown Source')})\n"

            if marker not in answer:
                answer += f" {marker}"
                
        answer += footnotes
        
        return answer, citations
    
    def batch_process(self, queries: List[str], save_to_file: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Process multiple queries in batch.
        
        Args:
            queries: List of queries to process
            save_to_file: Optional file path to save results
            
        Returns:
            List of response dictionaries
        """
        results = []
        
        for i, query in enumerate(queries):
            logger.info(f"Processing query {i+1}/{len(queries)}: '{query}'")
            result = self.process_query(query)
            results.append(result)
            
        if save_to_file:
            try:
                with open(save_to_file, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
                logger.info(f"Saved batch results to {save_to_file}")
            except Exception as e:
                logger.error(f"Error saving batch results: {e}")
                
        return results
    
    def update_document_rankings(self, query: str, doc_ids: List[str], relevance_scores: List[float]) -> None:
        """
        Update document relevance rankings based on user feedback.
        
        Args:
            query: Original query
            doc_ids: List of document IDs
            relevance_scores: List of relevance scores (0-1)
        """
        if len(doc_ids) != len(relevance_scores):
            logger.error("Document IDs and relevance scores must have the same length")
            return

        feedback_entry = {
            "query": query,
            "doc_ids": doc_ids,
            "relevance_scores": relevance_scores,
            "timestamp": time.time()
        }
        
        if self.cache_dir:
            feedback_path = os.path.join(self.cache_dir, "relevance_feedback.jsonl")
            try:
                with open(feedback_path, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(feedback_entry) + '\n')
            except Exception as e:
                logger.error(f"Error saving relevance feedback: {e}")
                
        logger.info(f"Saved relevance feedback for query: '{query}'")
        
    def clear_cache(self):
        """Clear the query cache."""
        self.query_cache = {}
        
        if self.cache_dir and os.path.exists(self.cache_dir):
            try:
                cache_files = list(Path(self.cache_dir).glob("*.json"))
                for file_path in cache_files:
                    try:
                        os.remove(file_path)
                    except Exception as e:
                        logger.warning(f"Error removing cache file {file_path}: {e}")
                        
                logger.info(f"Cleared {len(cache_files)} files from cache")
            except Exception as e:
                logger.error(f"Error clearing cache: {e}")