#!/usr/bin/env python
"""
Command-line interface for the Medical Device Regulation Navigator.
Provides a headless alternative to the Streamlit GUI.
"""

import os
import sys
import argparse
import json
import logging
from typing import List, Dict, Any, Optional
import time
from pathlib import Path
import textwrap
import datetime
import shutil
import readline 

script_dir = os.path.dirname(os.path.abspath(__file__))
interfaces_dir = os.path.dirname(script_dir)
project_root = os.path.dirname(interfaces_dir)
sys.path.append(project_root)

from interfaces.storage_manager import StorageManager

from medical_rag.vector_database.embeddings import EmbeddingGenerator
from medical_rag.vector_database.cloud_store import CloudVectorStore
from medical_rag.vector_database.retriever import DocumentRetriever

from medical_rag.rag_implementation.rag_pipeline import RAGPipeline
from medical_rag.rag_implementation.remote_llm import LLMInterface, LLMProvider
from medical_rag.rag_implementation.query_processor import QueryProcessor

try:
    from medical_rag.medical_domain.terminology import MedicalTerminologyProcessor
    from medical_rag.medical_domain.prompts import PromptGenerator
    MEDICAL_DOMAIN_AVAILABLE = True
except ImportError:
    MEDICAL_DOMAIN_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("cli_app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from file or use defaults.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    default_config = {
        "cache_dir": ".cache",
        "dataset_name": "medical-device-regs",
        "hf_username": None,
        "embedding_model": "all-MiniLM-L6-v2",
        "llm_provider": "openai",
        "model_name": "gpt-4.1",
        "top_k": 5,
        "search_threshold": 0.2,
        "max_history_items": 10,
        "enable_usage_tracking": True,
        "output_format": "text",
        "terminal_width": 80,
        "enable_spell_check": True, 
        "medical_terms_file": "medical_rag/medical_domain/medical_terms.txt"  
    }
    
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                user_config = json.load(f)
            return {**default_config, **user_config}
        except Exception as e:
            logger.warning(f"Error loading config file: {e}")

    default_path = os.path.join(project_root, "config.json")
    if os.path.exists(default_path):
        try:
            with open(default_path, 'r') as f:
                user_config = json.load(f)
            return {**default_config, **user_config}
        except Exception as e:
            logger.warning(f"Error loading default config file: {e}")
    
    return default_config

def initialize_system(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Initialize the RAG system components.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary of system components
    """
    logger.info("Initializing system components")

    cache_dir = config.get("cache_dir", ".cache")
    os.makedirs(cache_dir, exist_ok=True)

    print("Initializing embedding model...")
    embedding_model = config.get("embedding_model", "all-MiniLM-L6-v2")
    embedding_generator = EmbeddingGenerator(
        model_name=embedding_model,
        cache_dir=os.path.join(cache_dir, "embeddings")
    )

    print("Connecting to vector store...")
    dataset_name = config.get("dataset_name", "medical-device-regs")
    username = config.get("hf_username")
    vector_store = CloudVectorStore(
        dataset_name=dataset_name,
        username=username,
        local_cache_dir=os.path.join(cache_dir, "vector_store"),
        embedding_dim=embedding_generator.embedding_dim
    )

    print("Setting up document retriever...")
    retriever = DocumentRetriever(
        vector_store=vector_store,
        embedding_generator=embedding_generator,
        top_k=config.get("top_k", 5),
        cache_dir=os.path.join(cache_dir, "retriever")
    )

    print("Initializing LLM interface...")
    llm_provider = config.get("llm_provider", "openai")
    model_name = config.get("model_name", "gpt-4.1")
    api_key = os.environ.get(f"{llm_provider.upper()}_API_KEY")
    
    if not api_key:
        logger.warning(f"No API key found for {llm_provider}. Set {llm_provider.upper()}_API_KEY environment variable.")
        print(f"WARNING: No API key found for {llm_provider}.")
        
    llm_interface = LLMInterface(
        provider=llm_provider,
        model_name=model_name,
        api_key=api_key,
        cache_dir=os.path.join(cache_dir, "llm")
    )

    query_processor = QueryProcessor(
        cache_dir=os.path.join(cache_dir, "queries"),
        enable_spell_check=config.get("enable_spell_check", True),
        medical_terms_file=config.get("medical_terms_file")
    )

    medical_components = {}
    if MEDICAL_DOMAIN_AVAILABLE:
        try:
            print("Loading medical domain extensions...")
            terminology_processor = MedicalTerminologyProcessor(
                cache_dir=os.path.join(cache_dir, "terminology")
            )
            prompt_generator = PromptGenerator(
                cache_dir=os.path.join(cache_dir, "prompts")
            )
            medical_components = {
                "terminology_processor": terminology_processor,
                "prompt_generator": prompt_generator
            }
            logger.info("Medical domain components initialized")
        except Exception as e:
            logger.warning(f"Could not initialize medical domain components: {e}")

    print("Setting up RAG pipeline...")
    rag_pipeline = RAGPipeline(
        retriever=retriever,
        llm_interface=llm_interface,
        query_processor=query_processor,
        cache_dir=os.path.join(cache_dir, "pipeline")
    )

    storage_manager = StorageManager(cache_dir=cache_dir)
    
    system_components = {
        "embedding_generator": embedding_generator,
        "vector_store": vector_store,
        "retriever": retriever,
        "llm_interface": llm_interface,
        "query_processor": query_processor,
        "rag_pipeline": rag_pipeline,
        "storage_manager": storage_manager,
        **medical_components
    }
    
    logger.info("System components initialized successfully")
    print("System initialized successfully!")
    return system_components

def format_sources(sources: List[Dict[str, Any]], format_type: str = "text") -> str:
    """
    Format source documents for display.
    
    Args:
        sources: List of source documents
        format_type: Output format (text, json, or markdown)
        
    Returns:
        Formatted sources as string
    """
    if not sources:
        return "No sources found."
    
    if format_type == "json":
        return json.dumps(sources, indent=2)
    
    if format_type == "markdown":
        output = "## Sources\n\n"
        for i, source in enumerate(sources):
            output += f"### {i+1}. {source.get('title', 'Untitled')}\n\n"
            output += f"**Source**: {source.get('source', 'Unknown')}\n\n"
            output += f"**Relevance**: {source.get('score', 0):.2f}\n\n"
            output += f"**Excerpt**:\n\n{source.get('excerpt', '')}\n\n"
        return output

    output = "Sources:\n" + "-" * 40 + "\n"
    for i, source in enumerate(sources):
        output += f"{i+1}. {source.get('title', 'Untitled')} "
        output += f"(Source: {source.get('source', 'Unknown')}, "
        output += f"Score: {source.get('score', 0):.2f})\n"
        
        if "excerpt" in source:
            excerpt = source["excerpt"]
            wrapped = textwrap.fill(
                excerpt, 
                width=76,
                initial_indent="   ",
                subsequent_indent="   "
            )
            output += f"{wrapped}\n\n"
    
    return output

def print_banner():
    """Print application banner."""
    width = shutil.get_terminal_size().columns
    banner_width = min(width, 80)
    
    print("\n" + "=" * banner_width)
    print("Medical Device Regulation Navigator".center(banner_width))
    print("Command-line Interface".center(banner_width))
    print("-" * banner_width)
    print("Type 'help' for a list of commands".center(banner_width))
    print("=" * banner_width + "\n")

def print_help():
    """Print help information."""
    help_text = """
Available Commands:
------------------
search [query]     Search medical device regulations
spell [on/off]     Enable or disable spell checking
clear              Clear the screen
stats              Show system statistics
history            Show search history
config             Show current configuration
cache              Show cache information
clean              Clean up cache files
help               Show this help message
exit/quit          Exit the application

Examples:
---------
search What is the classification of a blood glucose monitor?
search What are the requirements for a 510(k) submission?
search How does the FDA regulate Software as a Medical Device?
spell on           Enable spell checking for queries
spell off          Disable spell checking for queries
"""
    print(help_text)

def process_query(system: Dict[str, Any], query: str, config: Dict[str, Any], history: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Process a search query.
    
    Args:
        system: System components
        query: Search query
        config: Configuration
        history: Search history
        
    Returns:
        Response dictionary
    """
    try:
        print(f"Searching for: {query}")
        start_time = time.time()

        spell_check_enabled = system["query_processor"].enable_spell_check

        response = system["rag_pipeline"].process_query(query)

        query_info = response.get("query_info", {})
        if spell_check_enabled and query_info:
            corrected_query = query_info.get("corrected_query")
            if corrected_query and corrected_query != query:
                print(f"Showing results for: {corrected_query}")

        end_time = time.time()
        duration = end_time - start_time
        print(f"Search completed in {duration:.2f} seconds")

        answer = response.get("answer", "")
        sources = response.get("documents", [])

        max_history = config.get("max_history_items", 10)
        history_item = {
            "query": query,
            "answer": answer,
            "sources": sources,
            "timestamp": time.time()
        }

        if spell_check_enabled and query_info:
            corrected_query = query_info.get("corrected_query")
            if corrected_query and corrected_query != query:
                history_item["corrected_query"] = corrected_query
        
        history.insert(0, history_item)

        if len(history) > max_history:
            history = history[:max_history]

        if config.get("enable_usage_tracking", True):
            system["retriever"].save_query_history(query, sources)
        
        return response
    
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        print(f"Error: {str(e)}")
        return {"error": str(e)}

def print_response(response: Dict[str, Any], config: Dict[str, Any]):
    """
    Print formatted response.
    
    Args:
        response: Response dictionary
        config: Configuration
    """
    if "error" in response:
        print(f"Error: {response['error']}")
        return
    
    format_type = config.get("output_format", "text")
    terminal_width = config.get("terminal_width", 80)
    
    answer = response.get("answer", "")
    sources = response.get("documents", [])
    
    if format_type == "json":
        print(json.dumps(response, indent=2))
        return
    
    if format_type == "markdown":
        print("# Answer\n")
        print(answer)
        print("\n" + format_sources(sources, format_type))
        return

    print("\nAnswer:")
    print("-" * terminal_width)

    wrapped_answer = textwrap.fill(
        answer,
        width=terminal_width,
        replace_whitespace=False
    )
    print(wrapped_answer)
    print("\n" + "-" * terminal_width + "\n")

    print(format_sources(sources, format_type))

def print_stats(system: Dict[str, Any], config: Dict[str, Any]):
    """
    Print system statistics.
    
    Args:
        system: System components
        config: Configuration
    """
    print("\nSystem Statistics:")
    print("-" * 40)

    try:
        vector_stats = system["vector_store"].stats()
        print(f"Document Count: {vector_stats.get('document_count', 'Unknown')}")
        print(f"Embedding Dimension: {vector_stats.get('embedding_dimension', 'Unknown')}")
        print(f"Storage Mode: {vector_stats.get('storage_mode', 'Unknown')}")
        print(f"Dataset Name: {vector_stats.get('dataset_name', 'Unknown')}")
    except Exception as e:
        print(f"Error getting vector store stats: {e}")

    try:
        storage_stats = system["storage_manager"].get_storage_stats()
        print(f"\nStorage Usage: {storage_stats.get('total_size_mb', 0):.1f} MB / "
              f"{storage_stats.get('max_storage_mb', 0)} MB "
              f"({storage_stats.get('usage_percent', 0):.1f}%)")
        
        print("\nSubdirectory Sizes:")
        for subdir, info in storage_stats.get("subdirectories", {}).items():
            print(f"  {subdir}: {info.get('size_mb', 0):.1f} MB")
    except Exception as e:
        print(f"Error getting storage stats: {e}")

    print("\nActive Configuration:")
    print(f"  Model: {config.get('model_name', 'Unknown')}")
    print(f"  Provider: {config.get('llm_provider', 'Unknown')}")
    print(f"  Top K: {config.get('top_k', 5)}")
    print(f"  Output Format: {config.get('output_format', 'text')}")
    print(f"  Spell Check: {'Enabled' if config.get('enable_spell_check', True) else 'Disabled'}")

def print_history(history: List[Dict[str, Any]]):
    """
    Print search history.
    
    Args:
        history: Search history
    """
    if not history:
        print("No search history available.")
        return
    
    print("\nSearch History:")
    print("-" * 40)
    
    for i, item in enumerate(history):
        ts = item.get("timestamp", 0)
        date_str = datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")

        query = item.get("query", "")
        if len(query) > 60:
            query = query[:57] + "..."

        correction_info = ""
        if "corrected_query" in item:
            corrected = item["corrected_query"]
            correction_info = f" → '{corrected}'"

        num_sources = len(item.get("sources", []))
        
        print(f"{i+1}. [{date_str}] {query}{correction_info} ({num_sources} sources)")
    
    print("\nTo view a specific result, use 'history [number]'")

def view_history_item(history: List[Dict[str, Any]], index: int, config: Dict[str, Any]):
    """
    View a specific history item.
    
    Args:
        history: Search history
        index: Item index (1-based)
        config: Configuration
    """
    if not history:
        print("No search history available.")
        return
    
    if index < 1 or index > len(history):
        print(f"Invalid history index. Please enter a number between 1 and {len(history)}")
        return
    
    item = history[index-1]
    response = {
        "answer": item.get("answer", ""),
        "documents": item.get("sources", [])
    }

    query = item.get("query", "")
    print(f"\nQuery: {query}")

    if "corrected_query" in item:
        corrected = item["corrected_query"]
        print(f"Corrected to: {corrected}")
    
    print_response(response, config)

def show_config(config: Dict[str, Any]):
    """
    Show current configuration.
    
    Args:
        config: Configuration dictionary
    """
    print("\nCurrent Configuration:")
    print("-" * 40)
    
    for key, value in sorted(config.items()):
        print(f"{key}: {value}")

def show_cache_info(system: Dict[str, Any]):
    """
    Show cache information.
    
    Args:
        system: System components
    """
    print("\nCache Information:")
    print("-" * 40)
    
    storage_stats = system["storage_manager"].get_storage_stats()
    
    print(f"Total Cache Size: {storage_stats.get('total_size_mb', 0):.1f} MB")
    print(f"Maximum Storage: {storage_stats.get('max_storage_mb', 0)} MB")
    print(f"Available Space: {storage_stats.get('available_space_mb', 0):.1f} MB")
    print(f"Usage: {storage_stats.get('usage_percent', 0):.1f}%")
    
    print("\nSubdirectory Sizes:")
    for subdir, info in storage_stats.get("subdirectories", {}).items():
        print(f"  {subdir}: {info.get('size_mb', 0):.1f} MB")

    print("\nOldest Files:")
    oldest_files = system["storage_manager"].get_oldest_files(limit=5)
    for file in oldest_files:
        date_str = file.get("modified_date", "").split("T")[0]
        size_mb = file.get("size_mb", 0)
        path = file.get("path", "")
        print(f"  {date_str} - {size_mb:.1f} MB - {os.path.basename(path)}")

    print("\nLargest Files:")
    largest_files = system["storage_manager"].get_largest_files(limit=5)
    for file in largest_files:
        size_mb = file.get("size_mb", 0)
        path = file.get("path", "")
        print(f"  {size_mb:.1f} MB - {os.path.basename(path)}")

def clean_cache(system: Dict[str, Any]):
    """
    Clean cache files.
    
    Args:
        system: System components
    """
    print("Cleaning cache...")

    before_stats = system["storage_manager"].get_storage_stats()
    before_size = before_stats.get("total_size_mb", 0)

    results = system["storage_manager"].optimize_storage()

    after_stats = system["storage_manager"].get_storage_stats()
    after_size = after_stats.get("total_size_mb", 0)

    saved = before_size - after_size
    print(f"Cleaned up {saved:.1f} MB of cache files.")
    print(f"Storage usage: {after_stats.get('usage_percent', 0):.1f}% "
          f"({after_size:.1f} MB / {after_stats.get('max_storage_mb', 0)} MB)")

def interactive_mode(system: Dict[str, Any], config: Dict[str, Any]):
    """
    Start interactive command-line mode.
    
    Args:
        system: System components
        config: Configuration
    """
    history = []
    print_banner()
    
    while True:
        try:
            cmd = input("\nmedical-device-nav> ").strip()
            
            if not cmd:
                continue
                
            if cmd.lower() in ["exit", "quit", "q"]:
                print("Exiting Medical Device Regulation Navigator...")
                break
                
            if cmd.lower() == "help":
                print_help()
                continue
                
            if cmd.lower() == "clear":
                os.system('cls' if os.name == 'nt' else 'clear')
                print_banner()
                continue
                
            if cmd.lower() == "stats":
                print_stats(system, config)
                continue
                
            if cmd.lower().startswith("history"):
                parts = cmd.split()
                if len(parts) > 1 and parts[1].isdigit():
                    view_history_item(history, int(parts[1]), config)
                else:
                    print_history(history)
                continue
                
            if cmd.lower() == "config":
                show_config(config)
                continue
                
            if cmd.lower() == "cache":
                show_cache_info(system)
                continue
                
            if cmd.lower() == "clean":
                clean_cache(system)
                continue
                
            if cmd.lower().startswith("spell "):
                parts = cmd.split(maxsplit=1)
                if len(parts) > 1:
                    setting = parts[1].lower()
                    if setting in ["on", "enable", "true", "yes", "1"]:
                        system["query_processor"].enable_spell_check = True
                        config["enable_spell_check"] = True
                        print("Spell checking enabled")
                        try:
                            config_path = os.path.join(project_root, "config.json")
                            with open(config_path, "w") as f:
                                json.dump(config, f, indent=2)
                        except Exception as e:
                            print(f"Error saving config: {e}")
                    elif setting in ["off", "disable", "false", "no", "0"]:
                        system["query_processor"].enable_spell_check = False
                        config["enable_spell_check"] = False
                        print("Spell checking disabled")
                        try:
                            config_path = os.path.join(project_root, "config.json")
                            with open(config_path, "w") as f:
                                json.dump(config, f, indent=2)
                        except Exception as e:
                            print(f"Error saving config: {e}")
                    else:
                        print(f"Unknown setting: {setting}. Use 'on' or 'off'.")
                else:
                    print(f"Spell checking is currently {'enabled' if system['query_processor'].enable_spell_check else 'disabled'}")
                continue
                
            if cmd.lower().startswith("search "):
                query = cmd[7:].strip()
                if query:
                    response = process_query(system, query, config, history)
                    print_response(response, config)
                else:
                    print("Please provide a search query.")
                continue

            response = process_query(system, cmd, config, history)
            print_response(response, config)
            
        except KeyboardInterrupt:
            print("\nCTRL+C detected. To exit, type 'exit' or 'quit'.")
        except EOFError:
            print("\nExiting Medical Device Regulation Navigator...")
            break
        except Exception as e:
            print(f"Error: {str(e)}")
            logger.error(f"Unexpected error: {e}")

def process_file(system: Dict[str, Any], input_file: str, output_file: Optional[str] = None, config: Dict[str, Any] = None):
    """
    Process queries from a file.
    
    Args:
        system: System components
        input_file: Input file path (one query per line)
        output_file: Output file path
        config: Configuration dictionary
    """
    if not os.path.exists(input_file):
        print(f"Input file not found: {input_file}")
        return
    
    if config is None:
        config = load_config()
    
    print(f"Processing queries from {input_file}...")
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            queries = [line.strip() for line in f if line.strip()]
        
        results = []
        
        for i, query in enumerate(queries):
            print(f"Processing query {i+1}/{len(queries)}: {query}")
            response = system["rag_pipeline"].process_query(query)

            query_info = response.get("query_info", {})
            corrected_query = None
            if query_info and system["query_processor"].enable_spell_check:
                corrected_query = query_info.get("corrected_query")
                if corrected_query and corrected_query != query:
                    print(f"Corrected to: {corrected_query}")
            
            result = {
                "query": query,
                "answer": response.get("answer", ""),
                "documents": response.get("documents", []),
                "timestamp": time.time()
            }

            if corrected_query and corrected_query != query:
                result["corrected_query"] = corrected_query
                
            results.append(result)

        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to {output_file}")
        else:
            for i, result in enumerate(results):
                print(f"\n\nQuery {i+1}: {result['query']}")
                if "corrected_query" in result:
                    print(f"Corrected to: {result['corrected_query']}")
                print_response(result, config)
                
    except Exception as e:
        logger.error(f"Error processing file: {e}")
        print(f"Error: {str(e)}")

def main():
    """Main entry point for the CLI application."""
    parser = argparse.ArgumentParser(description="Medical Device Regulation Navigator CLI")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--query", type=str, help="Run a single query and exit")
    parser.add_argument("--input-file", type=str, help="Process queries from a file (one per line)")
    parser.add_argument("--output-file", type=str, help="Save results to a file")
    parser.add_argument("--format", type=str, choices=["text", "json", "markdown"], 
                        help="Output format (text, json, or markdown)")
    parser.add_argument("--clean-cache", action="store_true", help="Clean cache files and exit")
    parser.add_argument("--spell-check", type=str, choices=["on", "off"], 
                        help="Enable or disable spell checking")
    
    args = parser.parse_args()

    config = load_config(args.config)

    if args.format:
        config["output_format"] = args.format
    
    if args.spell_check:
        config["enable_spell_check"] = (args.spell_check.lower() == "on")

    system = initialize_system(config)

    system["query_processor"].enable_spell_check = config["enable_spell_check"]

    if args.clean_cache:
        clean_cache(system)
        return
    
    if args.query:
        response = process_query(system, args.query, config, [])
        print_response(response, config)
        return
    
    if args.input_file:
        process_file(system, args.input_file, args.output_file, config)
        return

    interactive_mode(system, config)

if __name__ == "__main__":
    main()