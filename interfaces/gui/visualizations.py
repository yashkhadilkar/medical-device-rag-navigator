import streamlit as st
import pandas as pd
import numpy as np
from typing import List, Dict, Any
import matplotlib.pyplot as plt
import re 
import altair as alt
import math

def display_document_visualization(sources: List[Dict[str, Any]]):
    """Create a visual representation of the documents and their relationships.
    
    Args:
        sources: List of source documents with relevance scores
    """
    if not sources or len(sources) == 0:
        return
    
    st.markdown("### Document Relevance")

    data = []
    for i, source in enumerate(sources):
        title = source.get("title", f"Document {i+1}")
        score = source.get("score", 0)
        if len(title) > 40:
            title = title[:37] + "..."
            
        data.append({
            "title": title,
            "score": score,
            "index": i
        })
    
    df = pd.DataFrame(data)

    chart = alt.Chart(df).mark_bar().encode(
        x=alt.X('score:Q', title='Relevance Score'),
        y=alt.Y('title:N', title=None, sort='-x'),
        color=alt.Color('score:Q', scale=alt.Scale(scheme='blueorange'), legend=None),
        tooltip=['title', 'score']
    ).properties(
        height=min(len(sources) * 40, 300)
    ).interactive()
    
    st.altair_chart(chart, use_container_width=True)

    st.markdown("### Document Categories")

    categories = {}
    for source in sources:
        source_type = source.get("source", "Unknown")
        if source_type in categories:
            categories[source_type] += 1
        else:
            categories[source_type] = 1

    category_df = pd.DataFrame({
        'Category': list(categories.keys()),
        'Count': list(categories.values())
    })
    
    pie_chart = alt.Chart(category_df).mark_arc().encode(
        theta=alt.Theta(field="Count", type="quantitative"),
        color=alt.Color(field="Category", type="nominal", legend=alt.Legend(title="Document Type")),
        tooltip=['Category', 'Count']
    ).properties(
        width=350,
        height=250
    )
    
    st.altair_chart(pie_chart, use_container_width=True)
    
def display_term_frequency(sources: List[Dict[str, Any]], query: str):
    """Display the frequency of key terms from the query in the documents.
    
    Args:
        sources: List of source documents
        query: User query
    """
    stop_words = {"a", "an", "the", "and", "or", "but", "is", "are", "was", "were", 
                  "in", "on", "at", "to", "for", "with", "by", "about", "as", "what",
                  "how", "when", "where", "why", "which", "who"}
    
    terms = [word.lower() for word in re.findall(r'\b\w+\b', query) 
             if word.lower() not in stop_words and len(word) > 3]

    term_counts = {term: 0 for term in terms}
    
    for source in sources:
        excerpt = source.get("excerpt", "").lower()
        for term in terms:
            term_counts[term] += len(re.findall(r'\b' + re.escape(term) + r'\b', excerpt))

    term_counts = {k: v for k, v in term_counts.items() if v > 0}
    
    if not term_counts:
        return
    
    st.markdown("### Key Term Frequency")

    term_df = pd.DataFrame({
        'Term': list(term_counts.keys()),
        'Frequency': list(term_counts.values())
    }).sort_values('Frequency', ascending=False)
    
    chart = alt.Chart(term_df).mark_bar().encode(
        x='Frequency:Q',
        y=alt.Y('Term:N', sort='-x'),
        color=alt.Color('Frequency:Q', scale=alt.Scale(scheme='viridis'), legend=None),
        tooltip=['Term', 'Frequency']
    ).properties(
        height=min(len(term_counts) * 30, 300)
    )
    
    st.altair_chart(chart, use_container_width=True)
    
def display_system_stats(stats: Dict[str, Any]):
    """Display system statistics including storage usage.
    
    Args:
        stats: Dictionary of storage statistics
    """
    st.sidebar.subheader("Storage Usage")

    if "total_size_mb" in stats and "max_storage_mb" in stats:
        total_size = stats["total_size_mb"]
        max_storage = stats["max_storage_mb"]
        usage_percent = min(100, (total_size / max_storage * 100))

        storage_label = f"{total_size:.1f} MB / {max_storage} MB"

        if usage_percent < 60:
            bar_color = "green"
        elif usage_percent < 85:
            bar_color = "orange"
        else:
            bar_color = "red"
            
        st.sidebar.progress(usage_percent / 100, text=storage_label)
    else:
        st.sidebar.info("Storage statistics not available")

    if "cache_size_mb" in stats:
        st.sidebar.markdown(f"📊 **Cache Size**: {stats['cache_size_mb']:.1f} MB")
    
    if "document_count" in stats:
        st.sidebar.markdown(f"📑 **Documents**: {stats['document_count']}")
        
def display_regulatory_pathway_visualization():
    """Display a visualization of FDA regulatory pathways for medical devices.
    """
    st.markdown("### FDA Regulatory Pathways")

    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **Class I**
        * Low risk
        * General controls
        * Most exempt from 510(k)
        """)
        
    with col2:
        st.markdown("""
        **Class II**
        * Moderate risk
        * Special controls
        * Most require 510(k)
        """)
        
    with col3:
        st.markdown("""
        **Class III**
        * High risk
        * General & special controls
        * PMA required
        """)

    st.markdown("### Typical Regulatory Timeline")
    
    timeline_data = pd.DataFrame([
        {"Phase": "Pre-submission", "Weeks": 8, "Description": "Preparation and consultation"},
        {"Phase": "Submission", "Weeks": 4, "Description": "Document preparation and submission"},
        {"Phase": "FDA Review", "Weeks": 12, "Description": "510(k) review (90 days typical)"},
        {"Phase": "Response", "Weeks": 6, "Description": "Responding to FDA questions"},
        {"Phase": "Final Decision", "Weeks": 2, "Description": "FDA final determination"}
    ])

    timeline_data["Position"] = timeline_data["Weeks"].cumsum() - timeline_data["Weeks"]/2

    timeline_chart = alt.Chart(timeline_data).mark_bar().encode(
        x=alt.X('Position', axis=None),
        x2=alt.X2('Position:Q'),
        y=alt.Y('Phase:N', axis=alt.Axis(labelLimit=200)),
        width=alt.value(20),  # fixed bar width
        color=alt.Color('Phase:N', legend=None),
        tooltip=['Phase', 'Weeks', 'Description']
    ).properties(
        width=600,
        height=200
    )
    
    st.altair_chart(timeline_chart, use_container_width=True)
    
def display_device_classification_tree():
    """
    Display an interactive device classification decision tree.
    """
    st.markdown("### Device Classification Decision Tree")

    q1 = st.radio(
        "Is the device used to support or sustain human life?",
        ["Yes", "No"],
        key="q1"
    )
    
    if q1 == "Yes":
        st.markdown("🔴 **Likely Class III device**")
        st.markdown("Typically requires Premarket Approval (PMA)")
    else:
        q2 = st.radio(
            "Does the device present potential unreasonable risk of illness or injury?",
            ["Yes", "No"],
            key="q2"
        )
        
        if q2 == "Yes":
            st.markdown("🔴 **Likely Class III device**")
            st.markdown("Typically requires Premarket Approval (PMA)")
        else:
            q3 = st.radio(
                "Is the device intended for diagnosing, treating, preventing, or mitigating disease?",
                ["Yes", "No"],
                key="q3"
            )
            
            if q3 == "Yes":
                st.markdown("🟠 **Likely Class II device**")
                st.markdown("Typically requires 510(k) submission")
            else:
                st.markdown("🟢 **Likely Class I device**")
                st.markdown("May be 510(k) exempt, check regulation")

    st.markdown("""
    **Note**: This is a simplified decision tree for educational purposes only. 
    Actual device classification should be determined through consultation with 
    regulatory professionals and reference to FDA regulations.
    """)
    
def display_similarity_matrix(sources: List[Dict[str, Any]]):
    """Display a similarity matrix between different retrieved documents.
    
    Args:
        sources: List of source documents
    """
    if not sources or len(sources) < 2:
        return

    n = len(sources)
    similarity = np.zeros((n, n))
    
    for i in range(n):
        similarity[i, i] = 1.0  # Self-similarity is 1.0
        for j in range(i+1, n):
            score_i = sources[i].get("score", 0)
            score_j = sources[j].get("score", 0)
            sim = 0.5 + (min(score_i, score_j) / max(score_i, score_j)) * 0.5
            similarity[i, j] = sim
            similarity[j, i] = sim

    labels = [f"Doc {i+1}" for i in range(n)]

    similarity_df = pd.DataFrame(similarity, index=labels, columns=labels)
    
    similarity_melted = similarity_df.reset_index().melt(
        id_vars="index", 
        var_name="column", 
        value_name="similarity"
    )

    heatmap = alt.Chart(similarity_melted).mark_rect().encode(
        x='column:O',
        y='index:O',
        color=alt.Color('similarity:Q', scale=alt.Scale(scheme='viridis', domain=[0, 1])),
        tooltip=['index', 'column', 'similarity']
    ).properties(
        width=300,
        height=300,
        title="Document Similarity Matrix"
    )
    
    st.altair_chart(heatmap, use_container_width=True)
