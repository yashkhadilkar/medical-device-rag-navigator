import re 
import logging
import math
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    logging.warning("tiktoken not available. Install with: pip install tiktoken")

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import sent_tokenize, word_tokenize
import nltk

try:
    sent_tokenize("test sentence")
except LookupError:
    nltk.download('punkt_tab', quiet=True)

try:
    from nltk.corpus import stopwords
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords', quiet=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DocumentChunker:
    """Split documents into meaningful chunks for embedding and retrieval.
    Memory and storage efficient version for use with cloud-based storage.
    """
    
    def __init__(self, chunk_size: int = 1500, chunk_overlap: int = 200, 
                 max_tokens: int = 1800, chunk_method: str = 'semantic', tokenizer_name: str = "cl100k_base"):
        
        """Initialize the document chunker with larger default chunk sizes to reduce total vector count.

        Args:
            chunk_size: Target chunk size in characters (increased for efficiency)
            chunk_overlap: Overlap between chunks in characters
            max_tokens: Maximum number of tokens per chunk
            chunk_method: Method for chunking ("simple", "semantic", or "hierarchical")
            tokenizer_name: Name of the tokenizer for token counting
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_tokens = max_tokens
        self.chunk_method = chunk_method

        self.tokenizer = None
        if TIKTOKEN_AVAILABLE:
            try:
                self.tokenizer = tiktoken.get_encoding(tokenizer_name)
                logger.info(f"Using {tokenizer_name} tokenizer for token counting")
            except Exception as e:
                logger.warning(f"Could not load tokenizer {tokenizer_name}: {e}")

        self.vectorizer = TfidfVectorizer(
            min_df=2, max_df=0.95, 
            token_pattern=r'\b[\w\-\.]+\b',
            stop_words='english'
        )
        
    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in the text.

        Args:
            text: Input Text

        Returns:
            Number of tokens
        """
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        else:
            return len(text.split())
        
    def simple_chunk(self, text: str) -> List[str]:
        """Split text into chunks of roughly even size with overlap.
        Optimized for fewer, larger chunks.

        Args:
            text: Input Text

        Returns:
            List of text chunks
        """
        chunks = []
        start = 0

        if len(text) <= self.chunk_size:
            return [text]
        
        while start < len(text):
            end = start + self.chunk_size
            
            if end < len(text):
                next_para = text.find('\n\n', end - self.chunk_size // 2, end + self.chunk_size // 2)
                if next_para != -1:
                    end = next_para
                else:
                    next_sent = text.find('. ', end - 100, end + 100)
                    if next_sent != -1:
                        end = next_sent + 1
                        
            chunk = text[start:end].strip()

            token_count = self.count_tokens(chunk)
            if token_count > self.max_tokens:
                approx_chars = int(len(chunk) * (self.max_tokens / token_count))
                breakpoint = chunk.rfind('. ', 0, approx_chars)
                if breakpoint != -1:
                    chunk = chunk[:breakpoint+1]
                    
            if chunk:
                chunks.append(chunk)
                
            start = end - self.chunk_overlap
            if start <= 0 or start >= len(text):
                break
        
        return chunks
    
    def semantic_chunk(self, text: str) -> List[str]:
        """Split text into chunks based on semantic similarity.
        Optimized for fewer, more meaningful chunks.

        Args:
            text: Input text

        Returns:
            List of text chunks
        """
        paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]

        if len(paragraphs) < 3:
            new_paragraphs = []
            for para in paragraphs:
                if len(para) > self.chunk_size * 1.5:
                    sentences = sent_tokenize(para)
                    current_chunk = ""
                    for sentence in sentences:
                        if len(current_chunk) + len(sentence) <= self.chunk_size:
                            current_chunk += " " + sentence if current_chunk else sentence
                        else:
                            new_paragraphs.append(current_chunk)
                            current_chunk = sentence
                    if current_chunk:
                        new_paragraphs.append(current_chunk)
                else:
                    new_paragraphs.append(para)
            paragraphs = new_paragraphs

        if len(paragraphs) >= 5:
            try:
                tfidf_matrix = self.vectorizer.fit_transform(paragraphs)

                from sklearn.metrics.pairwise import cosine_similarity
                similarities = cosine_similarity(tfidf_matrix)

                chunks = []
                used = set()
                current_chunk = ""
                current_tokens = 0
                
                for i in range(len(paragraphs)):
                    if i in used:
                        continue

                    para_tokens = self.count_tokens(paragraphs[i])

                    if current_tokens + para_tokens > self.max_tokens:
                        if current_chunk:
                            chunks.append(current_chunk)
                        current_chunk = paragraphs[i]
                        current_tokens = para_tokens
                    else:
                        current_chunk += "\n\n" + paragraphs[i] if current_chunk else paragraphs[i]
                        current_tokens += para_tokens
                    
                    used.add(i)

                    similar_indices = np.argsort(similarities[i])[::-1]
                    for j in similar_indices:
                        if j not in used and similarities[i][j] > 0.3: 
                            para_tokens = self.count_tokens(paragraphs[j])
                            if current_tokens + para_tokens <= self.max_tokens:
                                current_chunk += "\n\n" + paragraphs[j]
                                current_tokens += para_tokens
                                used.add(j)

                if current_chunk and current_chunk not in chunks:
                    chunks.append(current_chunk)
                
                return chunks
            
            except Exception as e:
                logger.warning(f"Error in semantic chunking: {e}. Falling back to simple chunking.")

        return self.simple_chunk(text)
    
    def hierarchical_chunk(self, text: str, sections: List[Tuple[str, str]]) -> List[Dict[str, Any]]:
        """
        Create hierarchical chunks based on document sections.
        
        Args:
            text: Full document text
            sections: List of (section_title, section_content) tuples
            
        Returns:
            List of dictionaries containing chunks with hierarchical metadata
        """
        chunks = []
        
        for section_idx, (section_title, section_content) in enumerate(sections):
            if not section_content.strip():
                continue

            section_chunks = self.simple_chunk(section_content)
            
            for chunk_idx, chunk_text in enumerate(section_chunks):
                chunks.append({
                    "text": chunk_text,
                    "metadata": {
                        "section_title": section_title,
                        "section_idx": section_idx,
                        "chunk_idx": chunk_idx,
                        "total_section_chunks": len(section_chunks),
                        "is_first_chunk": chunk_idx == 0,
                        "is_last_chunk": chunk_idx == len(section_chunks) - 1
                    }
                })
                
        return chunks
    
    def chunk_document(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Split a document into chunks for embedding.
        Optimized for fewer chunks to reduce storage and API usage.
        
        Args:
            document: Document dictionary with text and metadata
            
        Returns:
            List of chunks with metadata
        """
        if not document.get("success", False):
            logger.warning(f"Skipping failed document: {document.get('file_path', 'unknown')}")
            return []

        text = document.get("processed_text", "")
        if not text:
            logger.warning(f"Empty text in document: {document.get('file_path', 'unknown')}")
            return []
        
        chunks = []
        file_name = document.get("file_name", "unknown")

        import hashlib
        doc_hash = hashlib.md5(text[:1000].encode()).hexdigest()[:8]

        if self.chunk_method == "semantic":
            text_chunks = self.semantic_chunk(text)
            for i, chunk_text in enumerate(text_chunks):
                chunks.append({
                    "text": chunk_text,
                    "chunk_id": f"{doc_hash}_{i}",
                    "chunk_index": i,
                    "total_chunks": len(text_chunks),
                    "file_path": document.get("file_path", ""),
                    "file_name": file_name,
                    "token_count": self.count_tokens(chunk_text)
                })
                
        elif self.chunk_method == "hierarchical" and document.get("sections"):
            hierarchical_chunks = self.hierarchical_chunk(text, document["sections"])

            for i, chunk in enumerate(hierarchical_chunks):
                chunk["chunk_id"] = f"{doc_hash}_{i}"
                chunk["file_path"] = document.get("file_path", "")
                chunk["file_name"] = file_name
                chunk["token_count"] = self.count_tokens(chunk["text"])
                chunks.append(chunk)
                
        else:
            text_chunks = self.simple_chunk(text)
            for i, chunk_text in enumerate(text_chunks):
                chunks.append({
                    "text": chunk_text,
                    "chunk_id": f"{doc_hash}_{i}",
                    "chunk_index": i,
                    "total_chunks": len(text_chunks),
                    "file_path": document.get("file_path", ""),
                    "file_name": file_name,
                    "token_count": self.count_tokens(chunk_text)
                })
        
        logger.info(f"Created {len(chunks)} chunks from document: {file_name}")
        return chunks
    
    def chunk_collection(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process and chunk a collection of documents

        Args:
            documents: List of document dictionaries

        Returns:
            List of all chunks from all documents.
        """
        all_chunks = []
        
        for doc in documents:
            doc_chunks = self.chunk_document(doc)
            all_chunks.extend(doc_chunks)
            
        logger.info(f"Created {len(all_chunks)} total chunks from {len(documents)} documents")
        return all_chunks
