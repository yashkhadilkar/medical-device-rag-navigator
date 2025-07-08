import re
import logging
import unicodedata
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize

try:
    import fitz 
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    logging.warning("PyMuPDF not available. Install with: pip install pymupdf")

try:
    sent_tokenize("test sentence")
    stopwords.words('english')
except LookupError:
    nltk.download('punkt_tab', quiet=True)
    nltk.download('stopwords', quiet=True)
    
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TextCleaner:
    """Clean and preprocess text extracted from documents, particularly FDA regulatory texts.
    Storage-efficient version for use with cloud-based document storage.
    """

    def __init__(self, remove_stopwords: bool = False, remove_headers_footers: bool = True,
                 max_content_size: int = 100000):
        """Initialize the text cleaner.

        Args:
            remove_stopwords: Whether to remove common stopwords.
            remove_headers_footers: Whether to attempt to remove headers and footers
            max_content_size: Maximum content size to process (for very large documents)
        """
        
        self.remove_stopwords = remove_stopwords
        self.remove_headers_footers = remove_headers_footers
        self.max_content_size = max_content_size
        self.stop_words = set(stopwords.words('english')) if remove_stopwords else set()
        
        # Common headers and footers in FDA documents
        self.header_footer_patterns = [
            r'Contains Nonbinding Recommendations',
            r'DRAFT GUIDANCE',
            r'GUIDANCE DOCUMENT',
            r'Page \d+ of \d+',
            r'www\.fda\.gov',
            r'\d+/\d+/\d+', 
            r'U\.S\. Food and Drug Administration',
            r'Center for Devices and Radiological Health',
            r'Department of Health and Human Services'
        ]
        
        # Regex patterns for common medical device terminology to preserve
        self.preserve_patterns = [
            r'510\(k\)',
            r'PMA',
            r'De Novo',
            r'SAMD',
            r'Class [I|II|III]',
            r'CFR \d+ Part \d+',
            r'ISO \d+'
        ]
        
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from a PDF file.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            Extracted text in string format.
        """
        if not PYMUPDF_AVAILABLE:
            logger.error("PyMuPDF not available. Cannot extract text from PDF.")
            return ""
            
        try:
            doc = fitz.open(pdf_path)
            text = ""
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text += page.get_text()

                if len(text) > self.max_content_size:
                    logger.info(f"Reached maximum content size of {self.max_content_size} characters. Truncating.")
                    break
                    
            logger.info(f"Extracted {len(text)} characters from {pdf_path}")
            return text
        except Exception as e:
            logger.error(f"Error extracting text from {pdf_path}: {e}")
            return ""
        
    def clean_text(self, text: str) -> str:
        """Clean and normalize text.

        Args:
            text: Input text string

        Returns:
            Cleaned Text
        """
        if len(text) > self.max_content_size:
            logger.info(f"Truncating text from {len(text)} to {self.max_content_size} characters")
            text = text[:self.max_content_size]

        text = ''.join(ch for ch in text if not unicodedata.category(ch).startswith('C'))

        text = re.sub(r'\s+', ' ', text).strip()

        if self.remove_headers_footers:
            for pattern in self.header_footer_patterns:
                text = re.sub(pattern, '', text)

        preserved_terms = {}
        for i, pattern in enumerate(self.preserve_patterns):
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                placeholder = f"__PRESERVED_TERM_{i}_{match.start()}__"
                preserved_terms[placeholder] = match.group()
                text = text.replace(match.group(), placeholder)

        if self.remove_stopwords:
            words = word_tokenize(text)
            filtered_words = [word for word in words if word.lower() not in self.stop_words]
            text = ' '.join(filtered_words)

        for placeholder, term in preserved_terms.items():
            text = text.replace(placeholder, term)

        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def preprocess_regulatory_document(self, text: str) -> str:
        """Apply specialized preprocessing for regulatory documents.

        Args:
            text: Input text from a regulatory document.

        Returns:
            Preprocessed text.
        """

        text = self.clean_text(text)
        

        text = re.sub(r'This document is meant for guidance only.*?(?=\n\n)', '', text, flags=re.DOTALL)
        text = re.sub(r'Draft - Not for Implementation.*?(?=\n\n)', '', text, flags=re.DOTALL)

        text = re.sub(r'Table of Contents.*?(?=\n\n\d+\.)', '', text, flags=re.DOTALL)

        text = re.sub(r'\n\s*Contains Nonbinding Recommendations\s*\n', '\n', text)
        
        return text
    
    def identify_document_sections(self, text: str) -> List[Tuple[str, str]]:
        """Identify main sections in a regulatory document. 

        Args:
            text: Document text

        Returns:
            List of (section_title, section_content) tuples
        """
        section_patterns = [
            r'^\s*(\d+\.\s*[\w\s]+)\s*\n',  # 1. Section Title
            r'^\s*(\d+\.\d+\.\s*[\w\s]+)\s*\n',  # 1.1. Section Title
            r'^\s*([IVX]+\.\s*[\w\s]+)\s*\n',  # IV. Section Title
            r'^\s*([A-Z][A-Z\s]+[A-Z])\s*\n'  # ALL CAPS TITLE
        ]
        
        sections = []
        current_section_title = "Introduction"
        current_section_content = ""
        
        lines = text.split('\n')
        i = 0
        
        while i < len(lines):
            line = lines[i]
            
            is_section_heading = False
            for pattern in section_patterns:
                match = re.match(pattern, line)
                if match:
                    if current_section_content.strip():
                        sections.append((current_section_title, current_section_content.strip()))

                    current_section_title = match.group(1).strip()
                    current_section_content = ""
                    is_section_heading = True
                    break
            
            if not is_section_heading:
                current_section_content += line + "\n"
            
            i += 1    

        if current_section_content.strip():
            sections.append((current_section_title, current_section_content.strip()))
        
        logger.info(f"Identified {len(sections)} sections in the document")
        return sections
    
    def process_document(self, file_path: str) -> Dict[str, Any]:
        """Process a document file and return the cleaned content with metadata.
        Storage-efficient version that extracts only the necessary information.

        Args:
            file_path: Path to the document file.

        Returns:
            Dictionary with document text and metadata
        """
        file_path = Path(file_path)
        result = {
            "file_path": str(file_path),
            "file_name": file_path.name,
            "file_extension": file_path.suffix.lower(),
            "processed_text": "",
            "sections": [],
            "success": False
        }
        
        try:
            if file_path.suffix.lower() == '.pdf' and PYMUPDF_AVAILABLE:
                text = self.extract_text_from_pdf(str(file_path))
            elif file_path.suffix.lower() in ['.txt', '.json']:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    text = f.read()

                    if file_path.suffix.lower() == '.json':
                        try:
                            import json
                            data = json.loads(text)
                            if isinstance(data, dict) and 'text' in data:
                                text = data['text']
                        except:
                            pass
            else:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    text = f.read()

            cleaned_text = self.preprocess_regulatory_document(text)
            result["processed_text"] = cleaned_text

            result["sections"] = self.identify_document_sections(cleaned_text)

            result["success"] = True

            import hashlib
            result["document_hash"] = hashlib.md5(cleaned_text[:5000].encode()).hexdigest()
            
        except Exception as e:
            logger.error(f"Error processing document {file_path}: {e}")
            result["error"] = str(e)
        
        return result
    
    def process_text_content(self, text: str, source_name: str = "text_input") -> Dict[str, Any]:
        """Process text content directly without reading from file.
        Useful for processing extracted text that's already been stored efficiently.

        Args:
            text: Text content to process
            source_name: Source identifier for the text

        Returns:
            Dictionary with processed text and metadata
        """
        result = {
            "file_path": None,
            "file_name": source_name,
            "processed_text": "",
            "sections": [],
            "success": False
        }
        
        try:
            cleaned_text = self.preprocess_regulatory_document(text)
            result["processed_text"] = cleaned_text

            result["sections"] = self.identify_document_sections(cleaned_text)

            result["success"] = True

            import hashlib
            result["document_hash"] = hashlib.md5(cleaned_text[:5000].encode()).hexdigest()
            
        except Exception as e:
            logger.error(f"Error processing text content: {e}")
            result["error"] = str(e)
        
        return result

    def extract_metadata_from_text(self, text: str) -> Dict[str, Any]:
        """Extract basic metadata from document text.
        Provides a lightweight alternative to the full MetadataExtractor.

        Args:
            text: Document text

        Returns:
            Dictionary of extracted metadata
        """
        metadata = {}

        doc_number_patterns = [
            r'Document Number:\s*([A-Za-z0-9\-\.]+)',
            r'Document No\.?\s*([A-Za-z0-9\-\.]+)',
            r'FDA-\d{4}-[A-Za-z]+-\d+'
        ]
        
        for pattern in doc_number_patterns:
            match = re.search(pattern, text)
            if match:
                if '(' in pattern: 
                    metadata["document_number"] = match.group(1).strip()
                else:
                    metadata["document_number"] = match.group(0).strip()
                break

        date_patterns = [
            r'(?:Issued|Published|Effective)(?:\s+on)?(?:\s+date)?[:\s]+([A-Za-z]+\s+\d{1,2},?\s+\d{4})',
            r'(?:Date of Issuance|Publication Date)[:\s]+([A-Za-z]+\s+\d{1,2},?\s+\d{4})'
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, text)
            if match:
                metadata["publication_date"] = match.group(1).strip()
                break

        if "guidance" in text.lower()[:1000]:
            if "draft guidance" in text.lower()[:1000]:
                metadata["document_type"] = "Draft Guidance"
            elif "final guidance" in text.lower()[:1000]:
                metadata["document_type"] = "Final Guidance"
            else:
                metadata["document_type"] = "Guidance Document"

        lines = text.split('\n')
        for line in lines[:20]:
            line = line.strip()
            if len(line) > 20 and len(line) < 200 and not line.startswith(('http', 'www')):
                metadata["title"] = line
                break
        
        return metadata