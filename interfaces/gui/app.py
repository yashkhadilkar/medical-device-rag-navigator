import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
import logging
import json
import time
import re
from pathlib import Path
import streamlit as st

script_dir = os.path.dirname(os.path.abspath(__file__))  
interfaces_dir = os.path.dirname(script_dir)      
project_root = os.path.dirname(interfaces_dir)     

# Add project root to path
sys.path.insert(0, project_root)

try:
    from medical_rag.vector_database.embeddings import EmbeddingGenerator
    from medical_rag.vector_database.cloud_store import CloudVectorStore
    from medical_rag.vector_database.retriever import DocumentRetriever
    from medical_rag.rag_implementation.rag_pipeline import RAGPipeline
    from medical_rag.rag_implementation.remote_llm import LLMInterface, LLMProvider
    from medical_rag.rag_implementation.query_processor import QueryProcessor
except ImportError as e:
    st.error(f"Error importing medical_rag modules: {e}")
    st.error("Please make sure all medical_rag modules are properly installed.")
    st.stop()

class SimpleStorageManager:
    def __init__(self, cache_dir):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def get_storage_stats(self):
        return {"total_size_mb": 0, "usage_percent": 0}

try:
    from interfaces.storage_manager import StorageManager
except ImportError:
    StorageManager = SimpleStorageManager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def load_custom_css():
    st.markdown("""
    <style>
    /* Hide Streamlit header and footer */
    header[data-testid="stHeader"] {
        display: none;
    }
    
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 900px;
    }
    
    /* Chat message styling */
    .chat-message {
        padding: 1.5rem;
        margin: 1rem 0;
        border-radius: 12px;
        max-width: 85%;
        word-wrap: break-word;
        line-height: 1.7;
        font-size: 16px;
    }
    
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin-left: auto;
        margin-right: 0;
        border-bottom-right-radius: 4px;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
    }
    
    .assistant-message {
        background: linear-gradient(135deg, #f093fb 0%, #f5f7fa 100%);
        color: #2c3e50;
        margin-left: 0;
        margin-right: auto;
        border-bottom-left-radius: 4px;
        border: 1px solid #e8ecf0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    /* Better bullet points */
    .assistant-message ul {
        margin: 1rem 0;
        padding-left: 0;
        list-style: none;
    }
    
    .assistant-message li {
        position: relative;
        padding-left: 24px;
        margin: 8px 0;
        line-height: 1.6;
    }
    
    .assistant-message li:before {
        content: "●";
        color: #3498db;
        position: absolute;
        left: 0;
        font-weight: bold;
        font-size: 18px;
    }
    
    /* Headers in responses */
    .assistant-message h4 {
        color: #2c3e50;
        margin: 1.2rem 0 0.8rem 0;
        font-size: 18px;
        font-weight: 600;
    }
    
    /* Regulatory references */
    .reg-reference {
        background: #ecf0f1;
        padding: 8px 12px;
        border-radius: 6px;
        margin: 12px 0;
        font-style: italic;
        color: #7f8c8d;
        border-left: 3px solid #3498db;
    }
    
    /* Input styling */
    .stTextInput > div > div > input {
        border-radius: 25px;
        border: 2px solid #e0e0e0;
        padding: 0.75rem 1.25rem;
        font-size: 16px;
        transition: all 0.3s ease;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* Button styling */
    .stButton > button {
        border-radius: 25px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.6rem 2rem;
        font-weight: 600;
        font-size: 16px;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Welcome message styling */
    .welcome-container {
        text-align: center;
        padding: 60px 20px;
        color: #555;
        max-width: 700px;
        margin: 0 auto;
    }
    
    .example-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
        gap: 20px;
        margin: 32px 0;
    }
    
    .example-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5f7fa 100%);
        padding: 20px;
        border-radius: 16px;
        border-left: 4px solid #667eea;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    
    .example-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 24px rgba(0,0,0,0.2);
    }
    
    /* Loading states */
    .loading-message {
        text-align: center;
        color: #667eea;
        font-style: italic;
        padding: 2rem;
    }
    
    /* Status indicators */
    .status-indicator {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        margin-right: 8px;
    }
    
    .status-ready { background-color: #2ecc71; }
    .status-loading { background-color: #f39c12; }
    .status-error { background-color: #e74c3c; }
    </style>
    """, unsafe_allow_html=True)


@st.cache_data(ttl=7200, max_entries=200)  # Increased TTL to 2 hours, more entries
def get_cached_response(query_normalized):
    """Enhanced cache with similarity matching."""
    return None

def check_similar_cache(query):
    """Check for similar cached responses."""
    normalized_query = normalize_query(query)

    cache_file = ".cache/response_cache.json"
    cached_responses = {}
    
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r') as f:
                cached_responses = json.load(f)
        except:
            pass
    
    # Check for similarity matches
    for cached_query, cached_data in cached_responses.items():
        if calculate_query_similarity(normalized_query, cached_query) > 0.8:
            logger.info(f"Found similar cached response")
            return cached_data['answer']
    
    return None

def calculate_query_similarity(query1, query2):
    """Calculate similarity between normalized queries."""
    words1 = set(query1.split())
    words2 = set(query2.split())
    
    if not words1 or not words2:
        return 0.0
    
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    return len(intersection) / len(union)

def save_to_cache(query, answer):
    """Save response to persistent cache."""
    cache_file = ".cache/response_cache.json"
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    
    cached_responses = {}
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r') as f:
                cached_responses = json.load(f)
        except:
            pass
    
    normalized_query = normalize_query(query)
    cached_responses[normalized_query] = {
        'answer': answer,
        'timestamp': time.time()
    }

    if len(cached_responses) > 100:
        sorted_items = sorted(cached_responses.items(), key=lambda x: x[1]['timestamp'])
        cached_responses = dict(sorted_items[-100:])
    
    try:
        with open(cache_file, 'w') as f:
            json.dump(cached_responses, f)
    except:
        pass

def normalize_query(query):
    """More aggressive normalization for better cache hits."""
    normalized = re.sub(r'[^\w\s]', '', query.lower().strip())

    stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'what', 'how', 'when', 'where', 'why', 'do', 'does', 'can', 'will', 'would', 'should'}
    words = [w for w in normalized.split() if w not in stopwords]

    standardizations = {
        'five ten k': '510k',
        '510 k': '510k', 
        'five hundred ten k': '510k',
        'class 1': 'class i',
        'class 2': 'class ii',
        'class 3': 'class iii',
        'pma': 'premarket approval',
        'biocompat': 'biocompatibility',
        'med device': 'medical device',
        'samd': 'software medical device'
    }

    text = ' '.join(sorted(words))
    for old_term, new_term in standardizations.items():
        text = text.replace(old_term, new_term)
    
    return text

def preprocess_query_for_efficiency(query):
    """Optimize query to get better, more focused results."""
    regulatory_terms = {
        'classification': ['class', 'classify', 'classification'],
        '510k': ['510k', '510(k)', 'premarket notification'],
        'pma': ['pma', 'premarket approval'],
        'software': ['software', 'samd', 'app', 'digital'],
        'testing': ['test', 'biocompatibility', 'clinical', 'validation']
    }
    
    query_lower = query.lower()
    detected_topics = []
    
    for topic, terms in regulatory_terms.items():
        if any(term in query_lower for term in terms):
            detected_topics.append(topic)

    if detected_topics:
        enhanced_query = f"{query} {' '.join(detected_topics)}"
        return enhanced_query
    
    return query

def optimize_context_smartly(context_docs, max_chars=10000):
    """Query-adaptive context optimization that preserves keyword-relevant content."""
    if not context_docs:
        return context_docs
    
    optimized_docs = []
    total_chars = 0

    sorted_docs = sorted(context_docs, key=lambda x: x.get('score', 0), reverse=True)[:4]
    
    for doc in sorted_docs:
        text = doc.get('text', '').strip()

        if len(text) > 2500:
            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip() for s in sentences if len(s.strip()) > 15]

            preserved_sentences = sentences[:3]
            preserved_text = '. '.join(preserved_sentences) + '.'

            remaining_sentences = sentences[3:]
            important_keywords = [
                'biocompatibility', 'biocompatible', 'iso 10993', 'cytotoxicity', 
                'implant', 'testing', 'sterilization', 'sensitization', 'validation',
                '510k', 'pma', 'classification', 'class ii', 'class iii', 'software',
                'samd', 'clinical decision support', 'submission', 'clearance'
            ]
            
            for sentence in remaining_sentences:
                sentence_lower = sentence.lower()
                if any(keyword in sentence_lower for keyword in important_keywords):
                    if len(preserved_text) + len(sentence) + 2 <= 3500:  
                        preserved_text += ' ' + sentence + '.'
                    else:
                        break
            
            text = preserved_text

        if total_chars + len(text) <= max_chars:
            doc_copy = doc.copy()
            doc_copy['text'] = text
            optimized_docs.append(doc_copy)
            total_chars += len(text)
        else:
            remaining_chars = max_chars - total_chars
            if remaining_chars > 800:
                doc_copy = doc.copy()
                doc_copy['text'] = text[:remaining_chars-100] + '...'
                optimized_docs.append(doc_copy)
            break
    
    logger.info(f"Query-adaptive context: {len(context_docs)} -> {len(optimized_docs)} docs, {total_chars} chars")
    return optimized_docs
    formatted_lines = []
    in_list = False
    
    for line in lines:
        line = line.strip()
        if not line:
            if in_list:
                formatted_lines.append('</ul>')
                in_list = False
            formatted_lines.append('')
            continue
            
        if re.match(r'^[\*\-\•]\s+', line):
            if not in_list:
                formatted_lines.append('<ul>')
                in_list = True
            content = re.sub(r'^[\*\-\•]\s+', '', line)
            formatted_lines.append(f'<li>{content}</li>')
        else:
            if in_list:
                formatted_lines.append('</ul>')
                in_list = False

            if re.match(r'^[A-Z][^:]*:$', line):
                formatted_lines.append(f'<h4>{line}</h4>')
            else:
                formatted_lines.append(line)
    
    if in_list:
        formatted_lines.append('</ul>')
    
    result = '\n'.join(formatted_lines)

    result = re.sub(
        r'(Regulatory reference:|Reference:|CFR [0-9]+.*?[0-9]+.*)',
        r'<div class="reg-reference">\1</div>',
        result,
        flags=re.IGNORECASE
    )
    
    return result

@st.cache_resource
def initialize_system_fast(config):
    """Initialize the RAG system with immediate startup and S3 integration."""
    logger.info("Starting fast system initialization with S3 CloudVectorStore")
    
    cache_dir = config.get("cache_dir", ".cache")
    os.makedirs(cache_dir, exist_ok=True)
    
    try:
        embedding_model = config.get("embedding_model", "all-MiniLM-L6-v2")
        embedding_generator = EmbeddingGenerator(
            model_name=embedding_model,
            cache_dir=os.path.join(cache_dir, "embeddings")
        )
        
        # Initialize S3-based vector store (NEW - simplified config)
        vector_store = CloudVectorStore(config)

        retriever = DocumentRetriever(
            vector_store=vector_store,
            embedding_generator=embedding_generator,
            top_k=4,
            min_score=config.get("search_threshold", 0.1),
            cache_dir=os.path.join(cache_dir, "retriever")
        )

        llm_provider = config.get("llm_provider", "openai")
        model_name = config.get("model_name", "gpt-4")
        api_key = os.environ.get(f"{llm_provider.upper()}_API_KEY")
        
        if not api_key:
            st.error(f" No API key found for {llm_provider}. Please set {llm_provider.upper()}_API_KEY environment variable.")
            st.stop()
        
        llm_interface = LLMInterface(
            provider=llm_provider,
            model_name=model_name,
            api_key=api_key,
            cache_dir=os.path.join(cache_dir, "llm"),
            max_tokens=500,
            temperature=0.1
        )

        query_processor = QueryProcessor(
            cache_dir=os.path.join(cache_dir, "queries"),
            enable_spell_check=False
        )

        rag_pipeline = RAGPipeline(
            retriever=retriever,
            llm_interface=llm_interface,
            query_processor=query_processor,
            cache_dir=os.path.join(cache_dir, "pipeline"),
            top_k=4
        )

        def generate_balanced_efficient_prompt(self, query_info):
            return """You are an FDA medical device regulatory expert. Answer based on the provided context.

If the context contains relevant information, provide:
- Direct answer (2-3 sentences)
- Key requirements or points (bullets)
- Regulatory reference when applicable

If you can extract any useful information from the context, provide it. Only say you need more information if the context is completely irrelevant to the question.

Keep responses focused and under 200 words."""

        rag_pipeline._generate_system_prompt = generate_balanced_efficient_prompt.__get__(rag_pipeline, RAGPipeline)

        storage_manager = StorageManager(cache_dir=cache_dir)
        
        system_components = {
            "embedding_generator": embedding_generator,
            "vector_store": vector_store,
            "retriever": retriever,
            "llm_interface": llm_interface,
            "query_processor": query_processor,
            "rag_pipeline": rag_pipeline,
            "storage_manager": storage_manager
        }
        
        logger.info(f"Fast system initialization complete with S3 - loaded {vector_store.doc_count} documents")
        return system_components
        
    except Exception as e:
        st.error(f"Failed to initialize system: {e}")
        logger.error(f"System initialization failed: {e}")
        st.stop()

def load_config():
    """Load configuration from file or use defaults with cost optimizations."""
    config_path = os.path.join(project_root, "config.json")
    default_config = {
        "app_title": "Medical Device Regulation Navigator",
        "app_icon": "🧬", 
        "cache_dir": ".cache",
        "dataset_name": "medical-device-regulations",
        "hf_username": "ykhad",
        "embedding_model": "all-MiniLM-L6-v2",
        "llm_provider": "openai",
        "model_name": "gpt-4",  # Keeping GPT-4 as requested
        "top_k": 3,  # Reduced from 5 for cost optimization
        "enable_spell_check": False,
        "search_threshold": 0.2,  # More selective retrieval
        "max_tokens": 600,  # Optimized response length
        "temperature": 0.1
    }
    
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            return {**default_config, **config}
        except Exception as e:
            logger.warning(f"Error loading config file: {e}")
    
    return default_config

def format_response_text(text):
    """Format the response text with proper bullet points and structure."""
    text = re.sub(r'<[^>]+>', '', text) 
    text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', text)  

    lines = text.split('\n')
    formatted_lines = []
    in_list = False
    
    for line in lines:
        line = line.strip()
        if not line:
            if in_list:
                formatted_lines.append('</ul>')
                in_list = False
            formatted_lines.append('')
            continue

        if re.match(r'^[\*\-\•]\s+', line):
            if not in_list:
                formatted_lines.append('<ul>')
                in_list = True
            content = re.sub(r'^[\*\-\•]\s+', '', line)
            formatted_lines.append(f'<li>{content}</li>')
        else:
            if in_list:
                formatted_lines.append('</ul>')
                in_list = False

            if re.match(r'^[A-Z][^:]*:$', line):
                formatted_lines.append(f'<h4>{line}</h4>')
            else:
                formatted_lines.append(line)
    
    if in_list:
        formatted_lines.append('</ul>')
    
    result = '\n'.join(formatted_lines)

    result = re.sub(
        r'(Regulatory reference:|Reference:|CFR [0-9]+.*?[0-9]+.*)',
        r'<div class="reg-reference">\1</div>',
        result,
        flags=re.IGNORECASE
    )
    
    return result

def display_message(message: str, is_user: bool = False):
    """Display a chat message with proper formatting."""
    
    if not is_user:
        message = format_response_text(message)
    
    message_class = "user-message" if is_user else "assistant-message"
    
    st.markdown(f'''
    <div class="chat-message {message_class}">
        {message}
    </div>
    ''', unsafe_allow_html=True)

def display_welcome_message():
    """Display welcome message when chat is empty."""
    welcome_html = """
    <div class="welcome-container">
        <div style='font-size: 64px; margin-bottom: 24px;'>🧬</div>
        <h1 style='color: #2c3e50; margin-bottom: 16px; font-weight: 300;'>Medical Device Navigator</h1>
        <p style='font-size: 20px; line-height: 1.6; margin-bottom: 32px; color: #7f8c8d;'>
            Get expert answers about FDA medical device regulations from real regulatory documents.
        </p>
        <div class="example-grid">
            <div class="example-card">
                <strong style='color: #2c3e50;'>📋 Device Classification</strong><br/>
                <span style='color: #7f8c8d;'>"What class is a blood glucose monitor?"</span>
            </div>
            <div class="example-card">
                <strong style='color: #2c3e50;'>📝 510(k) Process</strong><br/>
                <span style='color: #7f8c8d;'>"What documents are needed for 510(k)?"</span>
            </div>
            <div class="example-card">
                <strong style='color: #2c3e50;'>💻 Software Devices</strong><br/>
                <span style='color: #7f8c8d;'>"How is medical device software regulated?"</span>
            </div>
            <div class="example-card">
                <strong style='color: #2c3e50;'>🧪 Testing Requirements</strong><br/>
                <span style='color: #7f8c8d;'>"What testing is needed for implants?"</span>
            </div>
        </div>
    </div>
    """
    
    st.markdown(welcome_html, unsafe_allow_html=True)

def display_cost_tracking():
    """Show cost tracking in sidebar - DISABLED per user request."""
    pass  

def display_sidebar_info(config, system_status="ready"):
    """Display information in the sidebar."""
    
    st.sidebar.title("🧬 Medical Device Navigator")
    
    st.sidebar.markdown("""
    ## About This System
    
    This navigator uses AI to search through real FDA regulatory documents and provide accurate, cited answers about medical device regulations.
    
    **What you can ask:**
    - Device classification questions
    - 510(k) submission requirements  
    - Software as Medical Device (SaMD) regulations
    - Biocompatibility and testing requirements
    - Quality system regulations (QSR)
    - Premarket approval (PMA) processes
    
    ---
    
    **How it works:**
    1. Your question is analyzed
    2. Relevant regulatory documents are retrieved
    3. AI generates an answer based on official sources
    4. Response includes regulatory references
    
    ---
    
    *⚠️ Always consult qualified regulatory professionals for official guidance.*
    """)

def main():
    """Main Streamlit application with fast startup and cost optimizations."""

    st.set_page_config(
        page_title="Medical Device Navigator",
        page_icon="🧬",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    load_custom_css()

    config = load_config()

    with st.spinner("🚀 Starting up..."):
        system = initialize_system_fast(config)

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "total_queries" not in st.session_state:
        st.session_state.total_queries = 0
    if "estimated_cost" not in st.session_state:
        st.session_state.estimated_cost = 0.0
    if "cache_hits" not in st.session_state:
        st.session_state.cache_hits = 0

    display_sidebar_info(config, "ready")

    if not st.session_state.messages:
        display_welcome_message()

    for message in st.session_state.messages:
        display_message(
            message["content"], 
            is_user=message["role"] == "user"
        )

    st.markdown("---")

    with st.form(key="chat_form", clear_on_submit=True):
        col1, col2 = st.columns([5, 1])
        
        with col1:
            user_input = st.text_input(
                "Ask about FDA medical device regulations:",
                placeholder="e.g., What are the requirements for Class II devices?",
                label_visibility="collapsed"
            )
        
        with col2:
            submit_button = st.form_submit_button("Send", use_container_width=True)

    if submit_button and user_input.strip():
        st.session_state.messages.append({
            "role": "user",
            "content": user_input
        })

        display_message(user_input, is_user=True)
        
        # Check similarity cache first
        # cached_answer = None  # DISABLED for fresh retrieval
        
        if False:  # DISABLED: cached_answer:
            st.success("💰 Found similar answer in cache - $0.00 cost!")
            
            st.session_state.messages.append({
                "role": "assistant",
                "content": cached_answer
            })
            st.session_state.cache_hits += 1
        else:
            with st.spinner("🔍 Searching (cost-optimized)..."):
                try:
                    start_time = time.time()
                    
                    # Enhanced query preprocessing
                    optimized_query = preprocess_query_for_efficiency(user_input)
                    
                    logger.info(f"Processing with FRESH retrieval: {optimized_query}")
                    
                    # Process with the RAG system
                    response = system["rag_pipeline"].process_query(optimized_query, use_cache=False)  # FRESH retrieval
                    
                    # Balanced context optimization (max 3200 chars)
                    if 'documents' in response:
                        original_docs = len(response.get('documents', []))
                        response['documents'] = optimize_context_smartly(
                            response['documents'], max_chars=8000  # Balanced optimization
                        )
                        optimized_docs = len(response['documents'])
                        
                        if optimized_docs < original_docs:
                            st.info(f"📄 Context optimized for cost: {optimized_docs}/{original_docs} documents")
 
                    if 'documents' in response and response['documents']:
                        st.info("🔍 **Documents Retrieved:**")
                        for i, doc in enumerate(response['documents'], 1):
                            title = doc.get('title', 'No Title')
                            score = doc.get('score', 0)
                            doc_id = doc.get('id', 'No ID')
                            st.write(f"**{i}.** {title} (Score: {score:.3f}, ID: {doc_id})")

                    answer = response.get("answer", "I couldn't generate a response. Please try again.")
                    
                    # Cache the response for future use
                    save_to_cache(user_input, answer)

                    input_tokens = len(user_input.split()) * 1.3
                    output_tokens = len(answer.split()) * 1.3
                    total_tokens = input_tokens + output_tokens
                    
                    cost_per_query = (input_tokens * 0.00003) + (output_tokens * 0.00006)
                    
                    st.session_state.total_queries += 1
                    st.session_state.estimated_cost += cost_per_query

                    st.info(f"💰 Cost: ${cost_per_query:.4f} (~{total_tokens:.0f} tokens) | Total: ${st.session_state.estimated_cost:.3f}")

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer
                    })

                    response_time = time.time() - start_time
                    logger.info(f"Query: {response_time:.2f}s, {total_tokens:.0f} tokens, ${cost_per_query:.4f}")
                    
                except Exception as e:
                    logger.error(f"Error processing query: {e}")
                    error_message = "I encountered an error. Please try rephrasing your question."
                    
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_message
                    })

        st.rerun()

if __name__ == "__main__":
    main()