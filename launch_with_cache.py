#!/usr/bin/env python
"""
Enhanced launcher with cache preloading and cleanup on exit.
"""

import os
import sys
import signal
import atexit
import subprocess
import time

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def preload_cache():
    """Preload the document cache before starting the app."""
    print("🔄 Preloading document cache...")
    print("This ensures fast startup and better performance.")
    
    try:
        # Add project path
        sys.path.append('.')
        
        from medical_rag.vector_database.cloud_store import CloudVectorStore
        
        config = {'embedding_model': 'all-MiniLM-L6-v2'}
        store = CloudVectorStore(config)
        
        if store.doc_count > 0:
            print(f"✅ Cache ready: {store.doc_count} documents loaded")

            results = store.search("medical device", top_k=1)
            if results:
                print(f"✅ Search verified: Found '{results[0].get('title', 'Unknown')}'")
            else:
                print(" Search returned no results")
            
            return True
        else:
            print("❌ No documents loaded in cache")
            return False
            
    except Exception as e:
        print(f"❌ Cache preload failed: {e}")
        return False

def cleanup_cache():
    """Clean up cache on exit."""
    print("\n🧹 Cleaning up cache...")
    
    try:
        cache_dir = ".cache"
        if os.path.exists(cache_dir):
            import shutil
            shutil.rmtree(cache_dir)
            print("✅ Cache cleaned up")
        else:
            print("✅ No cache to clean")
    except Exception as e:
        print(f" Cache cleanup failed: {e}")

def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully."""
    print(f"\n Received signal {signum}")
    cleanup_cache()
    print("Goodbye!")
    sys.exit(0)

def main():
    """Main launcher with cache management."""
    print("🚀 Medical Device RAG Navigator")
    print("=" * 40)
    
    # Register cleanup handlers
    atexit.register(cleanup_cache)
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Check for API key
    if not os.environ.get("OPENAI_API_KEY"):
        print("Please set your OpenAI API key first:")
        print("export OPENAI_API_KEY=your_key_here")
        return
    
    print("🔧 Starting up...")
    
    # Preload cache
    cache_ready = preload_cache()
    
    if not cache_ready:
        print("\n Cache preload failed. The app may be slower or fail.")
        print("Continue anyway? (y/n): ", end="")
        response = input().strip().lower()
        if response != 'y':
            print("Exiting...")
            return
    
    print("\n Starting Streamlit app...")
    
    try:
        # Start the Streamlit app
        result = subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "interfaces/gui/app.py",
            "--server.port", "enter_your_server_port",
            "--browser.serverAddress", "localhost"
        ])
        
    except KeyboardInterrupt:
        print("\n App stopped by user")
    except Exception as e:
        print(f"\n App failed: {e}")
    finally:
        cleanup_cache()

if __name__ == "__main__":
    main()
