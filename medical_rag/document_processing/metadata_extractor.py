import re 
import logging
import datetime
from typing import List, Dict, Any, Optional, Set, Tuple
from pathlib import Path
import json
import hashlib
import os

try:
    import dateutil.parser
    DATEUTIL_AVAILABLE = True
except ImportError:
    DATEUTIL_AVAILABLE = False
    logging.warning("dateutil not available. Install with: pip install python-dateutil")

try:
    import fitz 
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    logging.warning("PyMuPDF not available. Install with: pip install pymupdf")

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    logging.warning("spaCy not available. Install with: pip install spacy")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MetadataExtractor:
    """Extract metadata from regulatory documents to enhance search and retrieval.
    Storage-efficient version that focuses only on essential metadata.
    """
    
    def __init__(self, use_nlp: bool = False, regulatory_terms_file: Optional[str] = None):
        """Initialize the metadata extractor.

        Args:
            use_nlp: Whether to use NLP for entity extraction (default False to save resources)
            regulatory_terms_file: Path to JSON file with regulatory terms
        """
        self.use_nlp = use_nlp and SPACY_AVAILABLE
        
        if self.use_nlp:
            try:
                self.nlp = spacy.load("en_core_web_sm")
                logger.info("Loaded spaCy model for entity extraction")
            except (ImportError, OSError) as e:
                logger.warning(f"Could not load spaCy model: {e}")
                logger.warning("Run 'python -m spacy download en_core_web_sm' to install")
                self.use_nlp = False
                self.nlp = None
        else:
            self.nlp = None

        self.regulatory_terms = self._load_minimal_terms(regulatory_terms_file)

        self.patterns = {
            "document_date": [
                r'(?:Date of Issuance|Publication Date|Issued on|Published on):\s*(\w+ \d{1,2},? \d{4})',
                r'(\w+ \d{1,2},? \d{4})',
                r'(\d{1,2}/\d{1,2}/\d{4})'
            ],
            "document_number": [
                r'Document Number[:\s]+([A-Z0-9\-\.]+)',
                r'Guidance #[:\s]+([A-Z0-9\-\.]+)',
                r'(?:No\.|Number):\s*([A-Z0-9\-\.]+)'
            ],
            "document_type": [
                r'(Guidance for Industry)',
                r'(Draft Guidance)',
                r'(Final Guidance)'
            ],
            "cfr_references": [
                r'(\d+\s*CFR\s*(?:Part)?\s*\d+(?:\.\d+)?)'
            ],
            "iso_references": [
                r'(ISO\s*\d+(?:-\d+)?:\d+)'
            ]
        }
        
    def _load_minimal_terms(self, filename: Optional[str]) -> Dict[str, List[str]]:
        """Load minimal set of regulatory terms.

        Args:
            filename: Path to JSON file with terms (optional)

        Returns:
            Dictionary of term categories and terms
        """
        minimal_terms = {
            "device_classifications": ["Class I", "Class II", "Class III"],
            "submission_types": ["510(k)", "PMA", "De Novo"],
            "regulatory_bodies": ["FDA", "CDRH"]
        }
        
        if filename and os.path.exists(filename):
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    terms = json.load(f)
                logger.info(f"Loaded regulatory terms from {filename}")
                return terms
            except Exception as e:
                logger.error(f"Error loading regulatory terms: {e}")
        
        return minimal_terms
        
    def extract_from_text(self, text: str, max_text_length: int = 10000) -> Dict[str, Any]:
        """
        Extract metadata from document text with limits on text processing.
        
        Args:
            text: Document text
            max_text_length: Maximum text length to process
            
        Returns:
            Dictionary of extracted metadata
        """
        text_to_process = text[:max_text_length]
        metadata = {}
        
        for field, patterns in self.patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, text_to_process, re.IGNORECASE)
                if matches:
                    if field in ["cfr_references", "iso_references"]:
                        metadata[field] = list(set(matches))
                        break
                    else:
                        metadata[field] = matches[0]
                        break

        if "document_date" in metadata and DATEUTIL_AVAILABLE:
            try:
                parsed_date = dateutil.parser.parse(metadata["document_date"])
                metadata["document_date"] = parsed_date.strftime("%Y-%m-%d")
            except Exception:
                pass

        if "title" not in metadata:
            first_lines = text_to_process.split('\n')[:10]
            for line in first_lines:
                line = line.strip()
                if len(line) > 20 and len(line) < 150 and not line.startswith('http'):
                    metadata["title"] = line
                    break

        if self.use_nlp and self.nlp:
            first_chunk = text_to_process[:5000]
            doc = self.nlp(first_chunk)

            entities = {
                "organizations": [],
                "regulatory_references": []
            }
            
            for ent in doc.ents:
                if ent.label_ == "ORG":
                    entities["organizations"].append(ent.text)
                elif ent.label_ == "LAW":
                    entities["regulatory_references"].append(ent.text)

            for key, values in entities.items():
                if values:
                    metadata[key] = list(set(values))

            self._extract_regulatory_entities(text_to_process, metadata)

        if text:
            metadata["document_hash"] = hashlib.md5(text[:5000].encode('utf-8')).hexdigest()
        
        return metadata
    
    def _extract_regulatory_entities(self, text: str, metadata: Dict[str, Any]) -> None:
        """
        Extract regulatory-specific entities from text.
        
        Args:
            text: Document text
            metadata: Metadata dictionary to update
        """
        # Only look for the most important terms
        for category, terms in self.regulatory_terms.items():
            found_terms = []
            for term in terms:
                pattern = r'\b' + re.escape(term) + r'\b'
                if re.search(pattern, text, re.IGNORECASE):
                    found_terms.append(term)
            
            if found_terms:
                metadata[category] = found_terms
                
    def extract_from_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """
        Extract metadata from PDF file, with minimal PDF parsing.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Dictionary of extracted metadata
        """
        metadata = {
            "filename": os.path.basename(pdf_path),
            "file_extension": ".pdf",
            "file_size_bytes": os.path.getsize(pdf_path),
            "extraction_date": datetime.datetime.now().strftime("%Y-%m-%d")
        }

        if PYMUPDF_AVAILABLE:
            try:
                doc = fitz.open(pdf_path)
                pdf_metadata = doc.metadata
                if pdf_metadata:
                    metadata.update({
                        "title": pdf_metadata.get("title", ""),
                        "author": pdf_metadata.get("author", ""),
                        "subject": pdf_metadata.get("subject", ""),
                        "page_count": len(doc)
                    })

                text = ""
                for page_idx in range(min(5, len(doc))):
                    page = doc[page_idx]
                    text += page.get_text()
                doc.close()

                text_metadata = self.extract_from_text(text)

                for key, value in text_metadata.items():
                    if key not in metadata or not metadata[key]:
                        metadata[key] = value
                
            except Exception as e:
                logger.error(f"Error extracting from PDF {pdf_path}: {e}")
        
        return metadata
    
    def extract_from_json(self, json_path: str) -> Dict[str, Any]:
        """
        Extract metadata from JSON file containing pre-extracted text.
        
        Args:
            json_path: Path to JSON file
            
        Returns:
            Dictionary of extracted metadata
        """
        metadata = {
            "filename": os.path.basename(json_path),
            "file_extension": ".json",
            "file_size_bytes": os.path.getsize(json_path),
            "extraction_date": datetime.datetime.now().strftime("%Y-%m-%d")
        }
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if isinstance(data, dict):
                text = data.get("text", "")

                if "metadata" in data and isinstance(data["metadata"], dict):
                    for key, value in data["metadata"].items():
                        metadata[key] = value
            else:
                text = str(data)

            if text:
                text_metadata = self.extract_from_text(text)

                for key, value in text_metadata.items():
                    if key not in metadata or not metadata[key]:
                        metadata[key] = value
            
        except Exception as e:
            logger.error(f"Error extracting from JSON {json_path}: {e}")
        
        return metadata
    
    def process_document(self, doc_path: str, output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a document and extract its metadata.
        
        Args:
            doc_path: Path to document file
            output_dir: Directory to save metadata JSON file (optional)
            
        Returns:
            Dictionary of extracted metadata
        """
        file_path = Path(doc_path)
        
        if not file_path.exists():
            logger.error(f"File not found: {doc_path}")
            return {}

        if file_path.suffix.lower() == '.pdf':
            metadata = self.extract_from_pdf(doc_path)
        elif file_path.suffix.lower() == '.json':
            metadata = self.extract_from_json(doc_path)
        else:
            try:
                with open(doc_path, 'r', encoding='utf-8', errors='ignore') as f:
                    text = f.read()
                metadata = self.extract_from_text(text)

                metadata["filename"] = file_path.name
                metadata["file_extension"] = file_path.suffix.lower()
                metadata["file_size_bytes"] = os.path.getsize(doc_path)
                
            except Exception as e:
                logger.error(f"Error processing file {doc_path}: {e}")
                return {}

        if output_dir:
            output_path = Path(output_dir) / f"{file_path.stem}_metadata.json"
            try:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, indent=2, ensure_ascii=False)
                logger.info(f"Saved metadata to {output_path}")
            except Exception as e:
                logger.error(f"Error saving metadata: {e}")
        
        return metadata
    
    def batch_process(self, input_dir: str, output_dir: str, file_types: List[str] = ['.pdf', '.txt', '.json'], max_files: int = 100) -> List[Dict[str, Any]]:
        """
        Process a limited number of documents to conserve resources.
        
        Args:
            input_dir: Directory containing documents
            output_dir: Directory to save metadata files
            file_types: List of file extensions to process
            max_files: Maximum number of files to process
            
        Returns:
            List of metadata dictionaries
        """
        os.makedirs(output_dir, exist_ok=True)
        
        results = []
        input_path = Path(input_dir)
        files_processed = 0

        for file_type in file_types:
            for file_path in input_path.glob(f"**/*{file_type}"):
                if files_processed >= max_files:
                    logger.info(f"Reached maximum file limit of {max_files}")
                    break
                    
                logger.info(f"Processing {file_path}")
                metadata = self.process_document(str(file_path), output_dir)
                if metadata:
                    results.append(metadata)
                    files_processed += 1
            
            if files_processed >= max_files:
                break

        summary = {
            "processed_files": len(results),
            "processing_date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "file_types": file_types
        }
        
        summary_path = Path(output_dir) / "batch_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        
        return results