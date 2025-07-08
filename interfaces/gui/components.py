import streamlit as st
import pandas as pd
import re
import time
from typing import List, Dict, Any, Optional
import datetime


def welcome_message():
    """Display the welcome message for first-time users."""
    st.markdown("""
    ## 👋 Welcome to the Medical Device Regulation Navigator
    
    I'm your AI assistant for understanding FDA medical device regulations. Ask me questions like:
    
    - "What is the classification of a blood glucose monitor?"
    - "What are the requirements for a 510(k) submission?"
    - "How does the FDA regulate Software as a Medical Device?"
    - "What biocompatibility testing is needed for implantable devices?"
    
    Simply type your question above and I'll find relevant information from FDA regulations to help you.
    """)
def loading_placeholder(message: str = "Processing..."):
    """Create a loading placeholder for async operations."""
    placeholder = st.empty()
    with placeholder.container():
        spinner_col, message_col = st.columns([1, 5])
        with spinner_col:
            st.markdown("⏳")
        with message_col:
            st.markdown(f"**{message}**")
    
    return placeholder

def sidebar_config(system, config: Dict[str, Any]):
    """Display sidebar configuration options."""
    st.sidebar.title("⚙️ Configuration")

    st.sidebar.subheader("System Info")
    try:
        doc_count = system["vector_store"].stats().get("document_count", "Unknown")
    except:
        doc_count = "Unknown"
        
    st.sidebar.markdown(f"📚 **Documents**: {doc_count}")
    st.sidebar.markdown(f"🤖 **Model**: {config.get('model_name', 'Unknown')}")

    st.sidebar.subheader("Search Settings")
    
    top_k = st.sidebar.slider(
        "Max results per query:", 
        min_value=1, 
        max_value=10, 
        value=config.get("top_k", 5),
        help="Maximum number of documents to retrieve"
    )
    
    threshold = st.sidebar.slider(
        "Relevance threshold:", 
        min_value=0.0, 
        max_value=1.0, 
        value=config.get("search_threshold", 0.2),
        step=0.05,
        help="Minimum relevance score (0-1) for documents"
    )

    st.sidebar.subheader("Query Processing")
    enable_spell_check = st.sidebar.checkbox(
        "Enable spell checking", 
        value=config.get("enable_spell_check", True),
        help="Automatically correct typos in queries"
    )

    if enable_spell_check != config.get("enable_spell_check", True):
        config["enable_spell_check"] = enable_spell_check
        system["query_processor"].enable_spell_check = enable_spell_check

        try:
            import json
            import os
            script_dir = os.path.dirname(os.path.abspath(__file__))
            gui_dir = os.path.dirname(script_dir)
            interfaces_dir = os.path.dirname(gui_dir)
            project_root = os.path.dirname(interfaces_dir)
            config_path = os.path.join(project_root, "config.json")
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            st.success("Configuration updated")
        except Exception as e:
            st.warning(f"Could not save configuration: {e}")

    st.sidebar.subheader("System Management")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("Clear Cache", use_container_width=True):
            with st.spinner("Clearing cache..."):
                system["storage_manager"].clear_cache()
                st.success("Cache cleared successfully")
    
    with col2:
        if st.button("Sync Data", use_container_width=True):
            with st.spinner("Synchronizing data..."):
                success = system["vector_store"].synchronize()
                if success:
                    st.success("Data synchronized successfully")
                else:
                    st.error("Error synchronizing data")


def display_answer(answer: str):
    """Display the answer in a chatbot-like format."""
    if not answer:
        display_no_results()
        return

    st.markdown("### 🤖 Assistant")
    st.write(answer)
def display_no_results():
    """Display a message when no results are found."""
    st.warning("""
    No relevant information found in the regulatory database. 
    
    Try rephrasing your question or using more specific medical device terminology.
    """)


def display_search_results(results: List[Dict[str, Any]]):
    """Display search results in a chat-friendly format."""
    if not results:
        display_no_results()
        return
    
    with st.expander("📚 Sources Used", expanded=False):
        for i, result in enumerate(results):
            st.markdown(f"**Source {i+1}**: {result.get('title', 'Untitled')} (Relevance: {result.get('score', 0):.2f})")
            st.markdown(f"*Excerpt*: {result.get('excerpt', '')}")
            st.divider()

def display_result_sources(sources: List[Dict[str, Any]]):
    """Display the sources in a chat-friendly format."""
    if not sources:
        return

    with st.expander("📚 Sources", expanded=False):
        for i, source in enumerate(sources):
            st.markdown(f"**Source {i+1}**: {source.get('title', 'Untitled')}")
            st.markdown(f"*From*: {source.get('source', 'Unknown')}")
            st.markdown(f"*Relevance*: {source.get('score', 0):.2f}")
            if 'excerpt' in source:
                st.markdown(f"*Excerpt*: {source['excerpt']}")
            st.divider()
def format_datetime(timestamp):
    """Format a timestamp into a readable date and time."""
    dt = datetime.datetime.fromtimestamp(timestamp)
    return dt.strftime("%Y-%m-%d %H:%M:%S")

def display_query_history(history: List[Dict[str, Any]]):
    """Display the query history."""
    if not history:
        st.info("No search history yet.")
        return
    
    history_data = []
    for item in history:
        query_text = item.get("query", "")
        if "corrected_query" in item:
            query_text += f" (corrected to: {item['corrected_query']})"
            
        history_data.append({
            "Query": query_text,
            "Time": format_datetime(item.get("timestamp", 0)),
            "Results": len(item.get("sources", []))
        })
    
    if history_data:
        history_df = pd.DataFrame(history_data)
        st.dataframe(history_df, use_container_width=True, hide_index=True)

def display_error_message(error: str):
    """Display an error message."""
    st.error(f"""
    An error occurred: {error}
    
    Please try again or contact the system administrator if the problem persists.
    """)