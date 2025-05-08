#!/usr/bin/env python
"""
Test script for the Medical Device Regulation Navigator using documents from Hugging Face.
This script downloads regulatory documents from Hugging Face, processes them,
adds them to the vector store, and tests the RAG system.
"""

import os
import logging
import json
import time
import argparse
import sys
import shutil
from pathlib import Path
import tempfile


os.environ["TOKENIZERS_PARALLELISM"] = "false"

script_dir = os.path.dirname(os.path.abspath(__file__))  # rag_implementation dir
medical_rag_dir = os.path.dirname(script_dir)            # medical_rag dir
project_root = os.path.dirname(medical_rag_dir)          # project root
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

# Use this if needed, or get from environment
# os.environ["OPENAI_API_KEY"] = "your-openai-api-key"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("test_with_hf_docs.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

DOCUMENT_METADATA = {
    "classification": [
        {
            "title": "Classification of Products as Drugs and Devices",
            "filename": "classification/classification_products_drugs_devices.pdf",
            "category": "classification"
        },
        {
            "title": "Classification Procedures for Medical Devices",
            "filename": "classification/classification_procedures.pdf",
            "category": "classification"
        },
        {
            "title": "Requests for Feedback and Meetings for Medical Device Submissions",
            "filename": "classification/feedback_meetings_submissions.pdf",
            "category": "classification"
        }
    ],
    "submission": [
        {
            "title": "Format for Traditional and Abbreviated 510(k)s",
            "filename": "submission/traditional_510k_format.pdf",
            "category": "submission"
        },
        {
            "title": "Deciding When to Submit a 510(k) for a Change to an Existing Device",
            "filename": "submission/when_to_submit_510k.pdf",
            "category": "submission"
        },
        {
            "title": "De Novo Classification Process",
            "filename": "submission/de_novo_process.pdf",
            "category": "submission"
        },
        {
            "title": "Premarket Approval Application Content",
            "filename": "submission/pma_content.pdf",
            "category": "submission"
        }
    ],
    "software": [
        {
            "title": "Policy for Device Software Functions and Mobile Medical Applications",
            "filename": "software/software_functions_mobile_apps.pdf",
            "category": "software"
        },
        {
            "title": "Content of Premarket Submissions for Software Contained in Medical Devices",
            "filename": "software/software_submissions.pdf",
            "category": "software"
        },
        {
            "title": "Clinical Decision Support Software",
            "filename": "software/clinical_decision_support.pdf",
            "category": "software"
        },
        {
            "title": "Cybersecurity in Medical Devices",
            "filename": "software/cybersecurity.pdf",
            "category": "software"
        }
    ],
    "compliance": [
        {
            "title": "Medical Device Reporting",
            "filename": "compliance/medical_device_reporting.pdf",
            "category": "compliance"
        },
        {
            "title": "Unique Device Identification",
            "filename": "compliance/udi.pdf",
            "category": "compliance"
        },
        {
            "title": "Postmarket Surveillance",
            "filename": "compliance/postmarket_surveillance.pdf",
            "category": "compliance"
        }
    ],
    "testing": [
        {
            "title": "Use of International Standard ISO 10993-1",
            "filename": "testing/iso_10993.pdf",
            "category": "testing"
        },
        {
            "title": "Design Considerations for Devices Intended for Home Use",
            "filename": "testing/home_use_devices.pdf",
            "category": "testing"
        },
        {
            "title": "Appropriate Use of Voluntary Consensus Standards",
            "filename": "testing/consensus_standards.pdf",
            "category": "testing"
        },
        {
            "title": "General Principles of Software Validation",
            "filename": "testing/software_validation.pdf",
            "category": "testing"
        },
        {
            "title": "Benefit-Risk Factors for Substantial Equivalence",
            "filename": "testing/benefit_risk_factors.pdf",
            "category": "testing"
        }
    ]
}

TEST_QUERIES = {
    "classification": [
        "What is the classification of a blood glucose monitor?",
        "How are medical devices classified by the FDA?",
        "What makes a device Class III?",
    ],
    "submission": [
        "What is required in a 510(k) submission?",
        "When is a PMA required instead of a 510(k)?",
        "What documents are needed for a De Novo submission?",
    ],
    "software": [
        "How does the FDA regulate medical device software?",
        "What is the definition of Software as a Medical Device (SaMD)?",
        "What cybersecurity requirements apply to medical devices?",
    ],
    "compliance": [
        "What are the Quality System Regulation requirements?",
        "How should I report adverse events for my medical device?",
        "What are the labeling requirements for medical devices?",
    ],
    "testing": [
        "What biocompatibility testing is needed for implantable devices?",
        "What testing is required for a Class II device?",
        "How do I validate software in a medical device?",
    ]
}

def download_from_huggingface(repo_id: str, local_dir: str, categories=None):
    """Download medical device regulation documents from Hugging Face."""
    try:
        from huggingface_hub import snapshot_download

        os.makedirs(local_dir, exist_ok=True)
        
        logger.info(f"Downloading files from {repo_id} to {local_dir}")

        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            repo_type="dataset"
        )
        
        logger.info("Download complete")

        downloaded_files = []
        for category in DOCUMENT_METADATA:
            if categories and category not in categories:
                continue
                
            for doc_info in DOCUMENT_METADATA[category]:
                filepath = os.path.join(local_dir, doc_info["filename"])
                if os.path.exists(filepath):
                    downloaded_files.append({
                        "title": doc_info["title"],
                        "filepath": filepath,
                        "category": category
                    })
                else:
                    logger.warning(f"File not found: {filepath}")
        
        logger.info(f"Verified {len(downloaded_files)} files")
        return downloaded_files
    
    except ImportError:
        logger.error("huggingface_hub not installed. Install with: pip install huggingface_hub")
        raise
    except Exception as e:
        logger.error(f"Error downloading from Hugging Face: {e}")
        raise

def stream_file_from_huggingface(repo_id: str, file_path: str, use_stream=True):
    """
    Stream a file directly from Hugging Face without saving it locally.
    
    Args:
        repo_id: Hugging Face repository ID
        file_path: Path to the file within the repository
        use_stream: Whether to stream the file or download it fully
        
    Returns:
        File content as bytes
    """
    try:
        from huggingface_hub import hf_hub_download, hf_hub_url
        import requests
        import io
        
        if use_stream:
            file_url = hf_hub_url(repo_id=repo_id, filename=file_path, repo_type="dataset")

            response = requests.get(file_url, stream=True)
            response.raise_for_status()

            return response.content
        else:
            with tempfile.NamedTemporaryFile() as temp_file:
                hf_hub_download(
                    repo_id=repo_id,
                    filename=file_path,
                    repo_type="dataset",
                    local_dir=os.path.dirname(temp_file.name),
                    local_dir_use_symlinks=False
                )
                
                with open(temp_file.name, 'rb') as f:
                    return f.read()
    
    except ImportError:
        logger.error("huggingface_hub not installed. Install with: pip install huggingface_hub")
        raise
    except Exception as e:
        logger.error(f"Error streaming file from Hugging Face: {e}")
        raise

def process_documents(document_list, output_dir="data/processed"):
    """Process downloaded documents for the RAG system."""
    os.makedirs(output_dir, exist_ok=True)

    cleaner = TextCleaner()
    chunker = DocumentChunker(chunk_size=1500, chunk_overlap=200)
    metadata_extractor = MetadataExtractor()
    
    processed_chunks = []
    
    for doc_info in document_list:
        doc_path = doc_info["filepath"]
        logger.info(f"Processing document: {doc_info['title']}")
        
        try:
            processed_doc = cleaner.process_document(doc_path)

            logger.info(f"Extracted text length: {len(processed_doc.get('processed_text', ''))}")
            logger.info(f"First 100 chars: {processed_doc.get('processed_text', '')[:100]}")
            
            if not processed_doc["success"]:
                logger.warning(f"Failed to process document: {doc_path}")
                continue

            if doc_path.endswith('.pdf'):
                metadata = metadata_extractor.extract_from_pdf(doc_path)
            else:
                metadata = metadata_extractor.process_document(doc_path)
                
            metadata.update({
                "title": doc_info["title"],
                "category": doc_info["category"],
                "source": "FDA"
            })

            chunks = chunker.chunk_document(processed_doc)

            for chunk in chunks:
                chunk["text"] = chunk.get("text", "")

                chunk["id"] = f"{doc_info['category']}_{hash(chunk.get('text', '')[:100])}"

                metadata_copy = metadata.copy()
                metadata_copy["text"] = chunk.get("text", "")
                chunk["metadata"] = metadata_copy

                text_len = len(chunk.get("text", ""))
                logger.info(f"Chunk {chunk['id']} text length: {text_len} characters")

                if text_len < 50:
                    logger.warning(f"Text content suspiciously short: {chunk.get('text', '')}!")
                    if processed_doc.get("processed_text"):
                        chunk["text"] = processed_doc["processed_text"][:1500]
                        logger.info(f"Added fallback text, new length: {len(chunk['text'])}")
                
                processed_chunks.append(chunk)
                
            logger.info(f"Created {len(chunks)} chunks from document: {doc_info['title']}")
            
        except Exception as e:
            logger.error(f"Error processing document {doc_path}: {e}")
    
    logger.info(f"Processed {len(processed_chunks)} total chunks from {len(document_list)} documents")

    chunks_path = os.path.join(output_dir, "processed_chunks.json")
    with open(chunks_path, 'w', encoding='utf-8') as f:
        json.dump(processed_chunks, f, ensure_ascii=False, indent=2)
    
    return processed_chunks

def create_vector_store(chunks, cache_dir=".cache/vector_store"):
    """Create and populate vector store with document chunks."""
    embedding_generator = EmbeddingGenerator(
        model_name="all-MiniLM-L6-v2",
        cache_dir=os.path.join(cache_dir, "embeddings")
    )
    
    vector_store = CloudVectorStore(
        dataset_name="medical-device-regs-test",
        local_cache_dir=cache_dir,
        embedding_dim=embedding_generator.embedding_dim
    )

    logger.info(f"Generating embeddings for {len(chunks)} chunks")
    embedded_chunks = embedding_generator.embed_chunks(chunks, batch_size=4)

    logger.info("Adding chunks to vector store")
    added_count = vector_store.add_many(embedded_chunks)
    
    logger.info(f"Added {added_count} chunks to vector store")
    
    return vector_store, embedding_generator

def setup_rag_system(vector_store, embedding_generator, cache_dir=".cache/rag", use_local_llm=False):
    """Set up the RAG system with the populated vector store."""
    retriever = DocumentRetriever(
        vector_store=vector_store,
        embedding_generator=embedding_generator,
        top_k=5,
        cache_dir=os.path.join(cache_dir, "retriever")
    )

    if use_local_llm:
        llm_interface = LLMInterface(
            provider="local",
            cache_dir=os.path.join(cache_dir, "llm")
        )
    else:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not found in environment. Please set OPENAI_API_KEY.")
            
        llm_interface = LLMInterface(
            provider="openai",
            model_name="gpt-4.1",
            api_key=api_key,
            cache_dir=os.path.join(cache_dir, "llm")
        )

    query_processor = QueryProcessor(
        cache_dir=os.path.join(cache_dir, "queries")
    )

    rag_pipeline = RAGPipeline(
        retriever=retriever,
        llm_interface=llm_interface,
        query_processor=query_processor,
        cache_dir=os.path.join(cache_dir, "pipeline")
    )
    
    return rag_pipeline

def run_tests(rag_pipeline, categories=None, num_queries=1, output_file=None, verbose=False):
    """Run test queries and collect results."""
    results = {}

    if not categories:
        categories = list(TEST_QUERIES.keys())

    categories = [c for c in categories if c in TEST_QUERIES]
    if not categories:
        logger.error("No valid categories specified.")
        return {}
    
    logger.info(f"Running tests for categories: {', '.join(categories)}")
    logger.info(f"Number of queries per category: {num_queries}")
    
    total_queries = 0
    start_time = time.time()
    
    for category in categories:
        results[category] = []
        queries = TEST_QUERIES[category][:num_queries]
        
        for query in queries:
            logger.info(f"Testing query: {query}")
            try:
                result = rag_pipeline.process_query(query)

                result["category"] = category
                result["processing_time"] = time.time() - start_time
                
                results[category].append(result)
                total_queries += 1
                
                if verbose:
                    print(f"\n============ Query ============")
                    print(f"Category: {category}")
                    print(f"Query: {query}")
                    print(f"\n============ Answer ============")
                    print(result["answer"])
                    print(f"\n============ Sources ============")
                    for i, doc in enumerate(result.get("documents", [])):
                        print(f"{i+1}. {doc.get('title', 'Untitled')} (Score: {doc.get('score', 0):.2f})")
                    print("================================\n")
                else:
                    logger.info(f"Got answer with {len(result.get('documents', []))} supporting documents")
                
            except Exception as e:
                logger.error(f"Error processing query '{query}': {e}")
                results[category].append({
                    "query": query,
                    "error": str(e),
                    "category": category
                })
    
    total_time = time.time() - start_time

    summary = {
        "total_queries": total_queries,
        "total_time": total_time,
        "avg_time_per_query": total_time / total_queries if total_queries > 0 else 0,
        "categories_tested": categories,
        "timestamp": time.time()
    }
    
    results["summary"] = summary
    
    logger.info(f"Test complete. Processed {total_queries} queries in {total_time:.2f} seconds.")

    if output_file:
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            logger.info(f"Test results saved to {output_file}")
        except Exception as e:
            logger.error(f"Error saving results: {e}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Test the RAG system with Hugging Face documents')
    parser.add_argument('--repo-id', type=str, default='your-username/medical-device-regulations',
                       help='Hugging Face dataset repository ID')
    parser.add_argument('--skip-download', action='store_true', 
                       help='Skip downloading documents (use existing ones)')
    parser.add_argument('--skip-processing', action='store_true',
                       help='Skip processing documents (use existing processed chunks)')
    parser.add_argument('--download-dir', type=str, default='data/raw',
                       help='Directory for downloaded documents')
    parser.add_argument('--processed-dir', type=str, default='data/processed',
                       help='Directory for processed documents')
    parser.add_argument('--cache-dir', type=str, default='.cache',
                       help='Directory for caching')
    parser.add_argument('--categories', type=str, nargs='+',
                       choices=DOCUMENT_METADATA.keys(),
                       help='Categories to test')
    parser.add_argument('--num-queries', type=int, default=1,
                       help='Number of queries per category')
    parser.add_argument('--output', type=str, default='test_results.json',
                       help='Output file for results')
    parser.add_argument('--use-local-llm', action='store_true',
                       help='Use local LLM instead of cloud API')
    parser.add_argument('--keep-files', action='store_true',
                       help='Keep downloaded files after testing')
    parser.add_argument('--verbose', action='store_true',
                       help='Show detailed output including answers')
    parser.add_argument('--use-streaming', action='store_true',
                       help='Stream files directly from HF without saving permanently (experimental)')
    
    args = parser.parse_args()
    
    try:
        if not args.skip_download:
            logger.info(f"Downloading documents from Hugging Face: {args.repo_id}")
            documents = download_from_huggingface(
                repo_id=args.repo_id,
                local_dir=args.download_dir,
                categories=args.categories
            )
        else:
            documents = []
            for category in DOCUMENT_METADATA:
                if args.categories and category not in args.categories:
                    continue
                    
                for doc_info in DOCUMENT_METADATA[category]:
                    filepath = os.path.join(args.download_dir, doc_info["filename"])
                    if os.path.exists(filepath):
                        documents.append({
                            "title": doc_info["title"],
                            "filepath": filepath,
                            "category": category
                        })
            
            logger.info(f"Using {len(documents)} existing documents")

        if not args.skip_processing:
            logger.info("Processing documents...")
            chunks = process_documents(
                document_list=documents,
                output_dir=args.processed_dir
            )
        else:
            chunks_path = os.path.join(args.processed_dir, "processed_chunks.json")
            if os.path.exists(chunks_path):
                with open(chunks_path, 'r', encoding='utf-8') as f:
                    chunks = json.load(f)
                logger.info(f"Loaded {len(chunks)} existing chunks")
            else:
                logger.error("No existing processed chunks found. Cannot skip processing.")
                return 1

        logger.info("Creating vector store...")
        vector_store, embedding_generator = create_vector_store(
            chunks=chunks,
            cache_dir=os.path.join(args.cache_dir, "vector_store")
        )

        logger.info("Setting up RAG system...")
        rag_pipeline = setup_rag_system(
            vector_store=vector_store,
            embedding_generator=embedding_generator,
            cache_dir=os.path.join(args.cache_dir, "rag"),
            use_local_llm=args.use_local_llm
        )

        logger.info("Running tests...")
        results = run_tests(
            rag_pipeline=rag_pipeline,
            categories=args.categories,
            num_queries=args.num_queries,
            output_file=args.output,
            verbose=args.verbose
        )

        summary = results.get("summary", {})
        print("\nTest Summary:")
        print(f"Total queries: {summary.get('total_queries', 0)}")
        print(f"Total time: {summary.get('total_time', 0):.2f} seconds")
        print(f"Average time per query: {summary.get('avg_time_per_query', 0):.2f} seconds")
        print(f"Detailed results saved to: {args.output}")

        if not args.keep_files:
            logger.info("Cleaning up downloaded files...")
            if os.path.exists(args.download_dir) and not args.skip_download:
                shutil.rmtree(args.download_dir)
            if os.path.exists(args.processed_dir) and not args.skip_processing:
                shutil.rmtree(args.processed_dir)
            logger.info("Cleanup complete")
        
        return 0
    except Exception as e:
        logger.error(f"Test failed: {e}")
        print(f"Test failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
