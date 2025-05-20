#!/usr/bin/env python
"""
Example script to demonstrate the integration of the Medical Domain Specialization
modules with the existing RAG system.
"""

import os
import logging
import sys
import json
from pathlib import Path
import argparse

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)

from medical_rag.document_processing.text_cleaner import TextCleaner
from medical_rag.document_processing.chunking import DocumentChunker
from medical_rag.document_processing.metadata_extractor import MetadataExtractor

from medical_rag.vector_database.embeddings import EmbeddingGenerator
from medical_rag.vector_database.cloud_store import CloudVectorStore
from medical_rag.vector_database.retriever import DocumentRetriever

from medical_rag.rag_implementation.rag_pipeline import RAGPipeline
from medical_rag.rag_implementation.remote_llm import LLMInterface, LLMProvider
from medical_rag.rag_implementation.query_processor import QueryProcessor

from medical_rag.medical_domain.terminology import MedicalTerminologyProcessor, DeviceType, RegulatoryPath
from medical_rag.medical_domain.prompts import PromptGenerator, QueryDomain

parser = argparse.ArgumentParser(description='Medical Domain RAG Integration Demo')
parser.add_argument('--repo-id', type=str, default=None, help='Hugging Face dataset repository ID (format: username/repo-name)')
parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
parser.add_argument('--query', type=str, default=None, help='Custom query to process')
parser.add_argument('--cache-dir', type=str, default='.cache', help='Directory for caching')
parser.add_argument('--model', type=str, default='gpt-4.1', help='LLM model to use')

args = parser.parse_args()

log_level = logging.INFO if args.verbose else logging.WARNING
logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("medical_domain_integration.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MedicalDeviceRAGSystem:
    """
    Integrated RAG system with medical domain specialization for
    the Medical Device Regulation Navigator.
    """
    
    def __init__(self, repo_id=None, cache_dir=".cache", model_name="gpt-4.1", verbose=False):
        """
        Initialize the medical device RAG system.
        
        Args:
            repo_id: Hugging Face repository ID (username/repo-name)
            cache_dir: Directory for caching
            model_name: LLM model to use
            verbose: Whether to enable verbose output
        """
        self.cache_dir = cache_dir
        self.verbose = verbose
        os.makedirs(cache_dir, exist_ok=True)
        
        if verbose:
            print(f"Initializing Medical Device RAG System")
            print(f"Cache directory: {cache_dir}")
            print(f"Model: {model_name}")
            if repo_id:
                print(f"Hugging Face repository: {repo_id}")

        self.embedding_generator = EmbeddingGenerator(
            model_name="all-MiniLM-L6-v2",
            cache_dir=os.path.join(cache_dir, "embeddings")
        )

        username = None
        dataset_name = "medical-device-regs"
        if repo_id:
            parts = repo_id.split('/')
            if len(parts) == 2:
                username, dataset_name = parts
        
        self.vector_store = CloudVectorStore(
            dataset_name=dataset_name,
            username=username,
            local_cache_dir=os.path.join(cache_dir, "vector_store"),
            embedding_dim=self.embedding_generator.embedding_dim
        )
        
        self.retriever = DocumentRetriever(
            vector_store=self.vector_store,
            embedding_generator=self.embedding_generator,
            cache_dir=os.path.join(cache_dir, "retriever")
        )

        self.llm_interface = LLMInterface(
            provider="openai",
            model_name=model_name,
            cache_dir=os.path.join(cache_dir, "llm")
        )
        
        self.query_processor = QueryProcessor(
            cache_dir=os.path.join(cache_dir, "queries")
        )
        
        self.rag_pipeline = RAGPipeline(
            retriever=self.retriever,
            llm_interface=self.llm_interface,
            query_processor=self.query_processor,
            cache_dir=os.path.join(cache_dir, "pipeline")
        )

        self.terminology_processor = MedicalTerminologyProcessor(
            cache_dir=os.path.join(cache_dir, "terminology")
        )
        
        self.prompt_generator = PromptGenerator(
            cache_dir=os.path.join(cache_dir, "prompts")
        )
        
        logger.info("Initialized Medical Device RAG System with domain specialization")
        
    def process_query(self, query: str) -> dict:
        """
        Process a query with medical domain specialization.
        
        Args:
            query: User query
            
        Returns:
            Response with answer and supporting documents
        """
        logger.info(f"Processing query: '{query}'")
        if self.verbose:
            print(f"\nProcessing query: '{query}'")

        try:
            query_domain = self.prompt_generator.identify_query_domain(query)
            logger.info(f"Identified query domain: {query_domain.value}")
            if self.verbose:
                print(f"Domain: {query_domain.value}")
        except Exception as e:
            logger.warning(f"Error identifying query domain: {e}")
            query_domain = QueryDomain.GENERAL
            logger.info(f"Defaulted to GENERAL domain")
        
        try:
            device_type = self.terminology_processor.identify_device_type(query)
            logger.info(f"Identified device type: {device_type.name}")
            if self.verbose:
                print(f"Device type: {device_type.name}")
        except Exception as e:
            logger.warning(f"Error identifying device type: {e}")
            device_type = DeviceType.GENERAL
        
        try:
            regulatory_path = self.terminology_processor.identify_regulatory_path(query)
            logger.info(f"Identified regulatory path: {regulatory_path.name}")
            if self.verbose:
                print(f"Regulatory path: {regulatory_path.name}")
        except Exception as e:
            logger.warning(f"Error identifying regulatory path: {e}")
            regulatory_path = RegulatoryPath.GENERAL

        enhanced_query = self.terminology_processor.enhance_query(query)
        logger.info(f"Enhanced query: '{enhanced_query}'")
        if self.verbose:
            print(f"Enhanced query: '{enhanced_query}'")

        search_queries = self.terminology_processor.generate_search_queries(query)
        logger.info(f"Generated search queries: {search_queries}")
        if self.verbose:
            print("Search queries:")
            for i, sq in enumerate(search_queries):
                print(f"  {i+1}. {sq}")

        all_results = []
        for search_query in search_queries:
            results = self.retriever.retrieve(search_query, top_k=5)
            all_results.extend(results)

        seen_ids = set()
        unique_results = []
        for result in all_results:
            doc_id = result.get("id", "")
            if doc_id and doc_id not in seen_ids:
                seen_ids.add(doc_id)
                unique_results.append(result)

        unique_results.sort(key=lambda x: x.get("score", 0), reverse=True)
        top_results = unique_results[:5]
        
        logger.info(f"Retrieved {len(top_results)} unique documents")
        if self.verbose:
            print(f"Retrieved {len(top_results)} unique documents")

        context = self._format_context(top_results)
        
        try:
            prompts = self.prompt_generator.generate_full_prompt(query, context, query_domain)
            logger.info(f"Generated domain-specific prompts for {query_domain.value}")

            if device_type != DeviceType.GENERAL:
                prompts = self.prompt_generator.enhance_prompts_for_device_type(
                    prompts, device_type.name
                )
                logger.info(f"Enhanced prompts with {device_type.name} device information")
        except Exception as e:
            logger.warning(f"Error generating prompts: {e}")
            prompts = {
                "system": "You are a Medical Device Regulatory Assistant. Answer based on the provided context.",
                "user": f"Context:\n{context}\n\nQuestion: {query}",
                "domain": "general"
            }
            logger.info("Using fallback prompts")

        if self.verbose:
            print("\nGenerating answer with specialized prompts...")
            
        answer = self.llm_interface.generate(
            prompt=prompts["user"], 
            system_prompt=prompts["system"]
        )

        try:
            formatted_answer = self.prompt_generator.format_response(
                answer=answer,
                sources=self._format_sources(top_results),
                domain=query_domain 
            )
            logger.info("Successfully formatted response with domain-specific enhancements")
        except Exception as e:
            logger.warning(f"Error formatting response: {e}")
            formatted_answer = answer  
            logger.info("Using unformatted answer as fallback")

        answer_analysis = self.terminology_processor.extract_terms(formatted_answer)

        response = {
            "query": query,
            "enhanced_query": enhanced_query,
            "domain": query_domain.value,
            "device_type": device_type.name if device_type != DeviceType.GENERAL else None,
            "regulatory_path": regulatory_path.name if regulatory_path != RegulatoryPath.GENERAL else None,
            "answer": formatted_answer,
            "documents": self._format_sources(top_results),
            "answer_analysis": answer_analysis,
            "generated": self._get_timestamp()
        }
        
        return response
    
    def _format_context(self, results: list) -> str:
        """Format the context from retrieval results."""
        formatted_context = ""
        
        for i, result in enumerate(results):
            text = result.get("text", "")
            metadata = result.get("metadata", {})
            title = metadata.get("title", f"Document {i+1}")
            source = metadata.get("source", "Unknown")
            
            formatted_context += f"Document {i+1}: {title} (Source: {source})\n{text}\n\n"
            
        return formatted_context.strip()
    
    def _format_sources(self, results: list) -> list:
        """Format the sources from retrieval results."""
        sources = []
        
        for i, result in enumerate(results):
            metadata = result.get("metadata", {})
            source = {
                "id": result.get("id", f"doc_{i}"),
                "title": metadata.get("title", "Untitled Document"),
                "score": result.get("score", 0.0),
                "excerpt": result.get("text", "")[:300] + "...",  
                "source": metadata.get("source", "Unknown Source")
            }
            
            sources.append(source)
            
        return sources
    
    def _get_timestamp(self) -> float:
        """Get the current timestamp."""
        import time
        return time.time()
    
    def get_stats(self) -> dict:
        """Get system statistics."""
        try:
            vector_stats = self.vector_store.stats()
        except:
            vector_stats = {"document_count": "Unknown", "embedding_dimension": "Unknown"}
            
        return {
            "document_count": vector_stats.get("document_count", "Unknown"),
            "embedding_dimension": vector_stats.get("embedding_dimension", "Unknown"),
            "device_types": [dt.name for dt in DeviceType],
            "regulatory_paths": [rp.name for rp in RegulatoryPath],
            "query_domains": [qd.name for qd in QueryDomain]
        }
    
    def batch_process(self, queries: list) -> list:
        """Process multiple queries in batch."""
        results = []
        
        for query in queries:
            try:
                result = self.process_query(query)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing query '{query}': {e}")
                results.append({
                    "query": query,
                    "error": str(e)
                })
                
        return results

def main():
    """Main function to demonstrate the integration."""
    sample_queries = [
        "What is the classification of a blood glucose monitor?",
        "What is required in a 510(k) submission?",
        "How does the FDA regulate medical device software?",
        "What biocompatibility testing is needed for implantable devices?",
        "What are the labeling requirements for Class II devices?"
    ]

    rag_system = MedicalDeviceRAGSystem(
        repo_id=args.repo_id,
        cache_dir=args.cache_dir,
        model_name=args.model,
        verbose=args.verbose
    )

    if args.verbose:
        stats = rag_system.get_stats()
        print("\nSystem Statistics:")
        print(f"Document count: {stats['document_count']}")
        print(f"Embedding dimension: {stats['embedding_dimension']}")

    query = args.query if args.query else sample_queries[0]

    response = rag_system.process_query(query)

    print("\nAnswer:")
    print(response["answer"])

    print("\nSources:")
    for i, source in enumerate(response["documents"]):
        print(f"{i+1}. {source['title']} (Score: {source['score']:.2f})")

    output_file = "medical_domain_response.json"
    with open(output_file, "w") as f:
        json.dump(response, f, indent=2)
    
    print(f"\nResponse saved to {output_file}")

if __name__ == "__main__":
    main()
