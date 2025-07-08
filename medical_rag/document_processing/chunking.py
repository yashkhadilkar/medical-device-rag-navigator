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
    Optimized for smaller, more targeted chunks.
    """
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100, 
                 max_tokens: int = 700, chunk_method: str = 'paragraph', tokenizer_name: str = "cl100k_base"):
        
        """Initialize the document chunker with parameters optimized for retrieval precision.

        Args:
            chunk_size: Target chunk size in characters (reduced for better targeting)
            chunk_overlap: Overlap between chunks in characters
            max_tokens: Maximum number of tokens per chunk
            chunk_method: Method for chunking ("simple", "paragraph", "semantic" or "hierarchical")
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
            min_df=1, max_df=0.95, 
            token_pattern=r'\b[\w\-\.]+\b',
            stop_words='english'
        )
        
        logger.info(f"Initialized DocumentChunker with {chunk_method} method, size={chunk_size}, overlap={chunk_overlap}")
        
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
                # Try to find a paragraph break for a clean cut
                next_para = text.find('\n\n', end - self.chunk_size // 2, min(end + self.chunk_size // 2, len(text)))
                if next_para != -1:
                    end = next_para
                else:
                    # Try to find a sentence break
                    next_sent = text.find('. ', end - 100, min(end + 100, len(text)))
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
    
    def paragraph_chunk(self, text: str) -> List[str]:
        """Split text into paragraph-based chunks.
        This creates more natural chunks that align with document organization.

        Args:
            text: Input text

        Returns:
            List of text chunks
        """
        paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
        
        if not paragraphs:
            return [text]
            
        chunks = []
        current_chunk = ""
        current_tokens = 0
        
        for para in paragraphs:
            para_tokens = self.count_tokens(para)

            if para_tokens > self.max_tokens:
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = ""
                    current_tokens = 0

                para_chunks = self.simple_chunk(para)
                chunks.extend(para_chunks)
                continue

            if current_tokens + para_tokens > self.max_tokens:
                if current_chunk:
                    chunks.append(current_chunk)

                current_chunk = para
                current_tokens = para_tokens
            else:
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para
                    
                current_tokens += para_tokens

        if current_chunk:
            chunks.append(current_chunk)
            
        return chunks
    
    def semantic_chunk(self, text: str) -> List[str]:
        """Split text into chunks based on semantic similarity.
        Groups similar paragraphs together for more meaningful chunks.

        Args:
            text: Input text

        Returns:
            List of text chunks
        """
        paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]

        if len(paragraphs) < 3:
            return self.paragraph_chunk(text)

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
            logger.warning(f"Error in semantic chunking: {e}. Falling back to paragraph chunking.")

        return self.paragraph_chunk(text)
    
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

            section_paragraphs = self.paragraph_chunk(section_content)
            
            for chunk_idx, chunk_text in enumerate(section_paragraphs):
                chunks.append({
                    "text": chunk_text,
                    "metadata": {
                        "section_title": section_title,
                        "section_idx": section_idx,
                        "chunk_idx": chunk_idx,
                        "total_section_chunks": len(section_paragraphs),
                        "is_first_chunk": chunk_idx == 0,
                        "is_last_chunk": chunk_idx == len(section_paragraphs) - 1
                    }
                })
                
        return chunks
    
    def _detect_section_title(self, chunk_text: str) -> Optional[str]:
        """
        Detect a section title from the chunk text.
        
        Args:
            chunk_text: Chunk text
            
        Returns:
            Section title if detected, None otherwise
        """
        lines = chunk_text.split("\n")
        if not lines:
            return None
            
        first_line = lines[0].strip()

        if len(first_line) < 100 and (
            first_line.isupper() or 
            re.match(r'^[0-9]+\.\s+', first_line) or 
            re.match(r'^[IVXLCDM]+\.\s+', first_line) or  # Roman numerals
            re.match(r'^[A-Z][\.\)]\s+', first_line)      # Letter headings
        ):
            return first_line
            
        # Check for common section headers in regulatory documents
        title_patterns = [
            r'(PURPOSE|SCOPE|INTRODUCTION|BACKGROUND|DISCUSSION)',
            r'(DEFINITIONS|REQUIREMENTS|POLICY|PROCEDURE|REFERENCES)',
            r'(COMPLIANCE|ENFORCEMENT|RECOMMENDATIONS|GUIDANCE)',
            r'(APPENDIX|SUBMISSION|PREMARKET\s+NOTIFICATION)',
            r'(SUBSTANTIAL\s+EQUIVALENCE|SPECIAL\s+CONTROLS|GENERAL\s+CONTROLS)',
            r'(DEVICE\s+DESCRIPTION|PERFORMANCE\s+TESTING|CLINICAL\s+STUDIES)'
        ]
        
        for pattern in title_patterns:
            match = re.search(pattern, first_line)
            if match:
                return first_line
                
        return None
    
    def chunk_document(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Split a document into chunks for embedding.
        Optimized for smaller, more targeted chunks to improve retrieval precision.
        
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

        # Choose chunking method based on configuration
        if self.chunk_method == "semantic":
            text_chunks = self.semantic_chunk(text)
            
        elif self.chunk_method == "paragraph":
            text_chunks = self.paragraph_chunk(text)
            
        elif self.chunk_method == "hierarchical" and document.get("sections"):
            hierarchical_chunks = self.hierarchical_chunk(text, document["sections"])

            for i, chunk in enumerate(hierarchical_chunks):
                chunk["chunk_id"] = f"{doc_hash}_{i}"
                chunk["doc_id"] = doc_hash
                chunk["file_path"] = document.get("file_path", "")
                chunk["file_name"] = file_name
                chunk["token_count"] = self.count_tokens(chunk["text"])
                chunks.append(chunk)
                
            logger.info(f"Created {len(chunks)} hierarchical chunks from document: {file_name}")
            return chunks
                
        else:
            text_chunks = self.simple_chunk(text)

        # Generate metadata and add chunks
        for i, chunk_text in enumerate(text_chunks):
            chunk = {
                "text": chunk_text,
                "chunk_id": f"{doc_hash}_{i}",
                "doc_id": doc_hash,
                "chunk_index": i,
                "total_chunks": len(text_chunks),
                "file_path": document.get("file_path", ""),
                "file_name": file_name,
                "token_count": self.count_tokens(chunk_text)
            }
            
            # Add section detection based on text analysis
            section_title = self._detect_section_title(chunk_text)
            if section_title:
                chunk["section_title"] = section_title
                
            chunks.append(chunk)
        
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
    
    def optimize_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Optimize chunks for better retrieval by adding metadata and keywords.
        
        Args:
            chunks: List of chunks
            
        Returns:
            Optimized chunks
        """
        for chunk in chunks:
            text = chunk.get("text", "")
            
            # Extract keywords
            keywords = self._extract_keywords(text)
            if keywords:
                chunk["keywords"] = keywords

            doc_type = self._identify_document_type(text)
            if doc_type:
                chunk["doc_type"] = doc_type
                
            # Check for regulatory references
            references = self._extract_regulatory_references(text)
            if references:
                chunk["references"] = references
                
        return chunks
    
    def _extract_keywords(self, text: str) -> List[str]:
        """
        Extract keywords from text.
        
        Args:
            text: Input text
            
        Returns:
            List of keywords
        """
        stopwords_set = set(stopwords.words('english'))
        
        # Add domain-specific stopwords
        domain_stopwords = {"may", "shall", "must", "should", "can", "device", "devices",
                          "medical", "fda", "requirements", "guidance", "please", "section"}
        stopwords_set.update(domain_stopwords)
        
        # Tokenize text and filter out stopwords
        words = word_tokenize(text.lower())
        words = [word for word in words if word.isalpha() and 
                word not in stopwords_set and len(word) > 3]
        
        # Count word frequencies
        from collections import Counter
        word_counts = Counter(words)
        
        # Get top keywords
        top_keywords = [word for word, count in word_counts.most_common(10) if count > 1]
        
        return top_keywords
    
    def _identify_document_type(self, text: str) -> Optional[str]:
        """
        Identify the type of regulatory document.
        
        Args:
            text: Document text
            
        Returns:
            Document type if identified, None otherwise
        """
        text_lower = text.lower()
        
        if "guidance" in text_lower and "draft" in text_lower:
            return "draft_guidance"
        elif "guidance" in text_lower:
            return "guidance"
        elif "510(k)" in text_lower or "510k" in text_lower:
            return "510k"
        elif "pma" in text_lower or "premarket approval" in text_lower:
            return "pma"
        elif "de novo" in text_lower:
            return "de_novo"
        elif "classification" in text_lower:
            return "classification"
        elif "quality system" in text_lower or "qsr" in text_lower:
            return "qsr"
        elif "software" in text_lower and "medical device" in text_lower:
            return "software"
            
        return None
    
    def _extract_regulatory_references(self, text: str) -> List[str]:
        """
        Extract regulatory references from text.
        
        Args:
            text: Input text
            
        Returns:
            List of regulatory references
        """
        references = []
        
        # Look for CFR references
        cfr_matches = re.findall(r'(?:\d+)\s*CFR\s*(?:Part)?\s*(?:\d+(?:\.\d+)?)', text)
        references.extend(cfr_matches)
        
        # Look for FDA guidance references
        guidance_matches = re.findall(r'Guidance\s+(?:Document|for)?\s+#\s*([A-Z0-9\-]+)', text)
        references.extend(guidance_matches)
        
        # Look for ISO standards
        iso_matches = re.findall(r'ISO\s+\d+(?:-\d+)?(?::\d+)?', text)
        references.extend(iso_matches)
        
        return references
    
    def merge_small_chunks(self, chunks: List[Dict[str, Any]], min_token_count: int = 50) -> List[Dict[str, Any]]:
        """
        Merge small chunks to avoid inefficient embedding of tiny text fragments.
        
        Args:
            chunks: List of chunks
            min_token_count: Minimum token count for a standalone chunk
            
        Returns:
            List of merged chunks
        """
        if not chunks:
            return []
            
        # Sort chunks by document and chunk index
        chunks.sort(key=lambda x: (x.get("doc_id", ""), x.get("chunk_index", 0)))
        
        merged_chunks = []
        current_chunk = None
        
        for chunk in chunks:
            token_count = chunk.get("token_count", 0)
            
            if token_count < min_token_count:
                if current_chunk:
                    if current_chunk.get("doc_id") == chunk.get("doc_id"):
                        current_chunk["text"] += "\n\n" + chunk.get("text", "")

                        current_chunk["token_count"] = self.count_tokens(current_chunk["text"])

                        if "metadata" in current_chunk and "metadata" in chunk:
                            current_chunk["metadata"]["merged"] = True
                            current_chunk["metadata"]["merged_chunks"] = \
                                current_chunk["metadata"].get("merged_chunks", 1) + 1

                        if "keywords" in chunk and "keywords" in current_chunk:
                            current_chunk["keywords"] = list(set(
                                current_chunk["keywords"] + chunk.get("keywords", [])
                            ))
                        
                        continue
                    else:
                        merged_chunks.append(current_chunk)

                current_chunk = chunk.copy()
            else:
                if current_chunk:
                    merged_chunks.append(current_chunk)
                    
                current_chunk = chunk.copy()

        if current_chunk:
            merged_chunks.append(current_chunk)
            
        logger.info(f"Merged chunks: {len(chunks)} -> {len(merged_chunks)}")
        return merged_chunks