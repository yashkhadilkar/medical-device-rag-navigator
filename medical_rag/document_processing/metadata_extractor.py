import re 
import logging
import datetime
from typing import List, Dict, Any, Optional, Set, Tuple
from pathlib import Path
import json
import spacy
import dateutil.parser
from PyMuPDF import fitz
import os
import hashlib


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MetadataExtractor:
    """Extract metadata from regulatory documents to enhance search and retrieval.
    """
    
    def __init__(self, use_nlp: bool = True, regulatory_terms_file: Optional[str] = None):
        """Initialize the metadata extractor.

        Args:
            use_nlp: Whether to use NLP for entity extraction
            regulatory_terms_file: Path to JSON file with regulatory terms
        """
        self.use_nlp = use_nlp
        
        if use_nlp:
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
        
        self.regulatory_terms = self._load_regulatory_terms(regulatory_terms_file)
        
        self.patterns = {
            "document_date": [
                r'(?:Date of Issuance|Publication Date|Issued on|Published on):\s*(\w+ \d{1,2},? \d{4})',
                r'(\w+ \d{1,2},? \d{4})',
                r'(\d{1,2}/\d{1,2}/\d{4})',
                r'(\d{4}-\d{2}-\d{2})'
            ],
            "document_number": [
                r'Document Number[:\s]+([A-Z0-9\-\.]+)',
                r'Guidance #[:\s]+([A-Z0-9\-\.]+)',
                r'(?:No\.|Number):\s*([A-Z0-9\-\.]+)'
            ],
            "document_type": [
                r'(Guidance for Industry)',
                r'(Guidance for Industry and Food and Drug Administration Staff)',
                r'(Draft Guidance)',
                r'(Final Guidance)',
                r'(Technical Considerations)'
            ],
            "cfr_references": [
                r'(\d+\s*CFR\s*(?:Part)?\s*\d+(?:\.\d+)?)',
                r'(Title \d+,? Part \d+(?:\.\d+)?)'
            ],
            "iso_references": [
                r'(ISO\s*\d+(?:-\d+)?:\d+)',
                r'(International Standard\s*\d+(?:-\d+)?:\d+)'
            ]
        }
        
    def _load_regulatory_terms(self, filename: Optional[str]) -> Dict[str, List[str]]:
        """Load regulatory terms from a JSON file.

        Args:
            filename: Path to JSON file with terms.

        Returns:
            Dictionary of term categories and terms
        """
        if not filename or not os.path.exists(filename):
            return {
                "device_classifications": ["Class I", "Class II", "Class III"],
                "submission_types": ["510(k)", "PMA", "De Novo", "HDE", "IDE"],
                "standards": ["ISO", "IEC", "ASTM", "ANSI"],
                "regulatory_bodies": ["FDA", "CDRH", "CBER", "CDER", "EMA", "PMDA", "Health Canada"]
            }
        
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                terms = json.load(f)
            logger.info(f"Loaded {sum(len(v) for v in terms.values())} regulatory terms from {filename}")
            return terms
        except Exception as e:
            logger.error(f"Error loading regulatory terms from {filename}: {e}")
            return {}   
        
    def extract_pdf_metadata(self, pdf_path: str) -> Dict[str, Any]:
        """Extract metadata from PDF document properties.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Dictionary of PDF metadata
        """
        metadata = {}
        
        try:
            doc = fitz.open(pdf_path)

            pdf_metadata = doc.metadata
            if pdf_metadata:
                if "title" in pdf_metadata and pdf_metadata["title"]:
                    metadata["title"] = pdf_metadata["title"]
                if "author" in pdf_metadata and pdf_metadata["author"]:
                    metadata["author"] = pdf_metadata["author"]
                if "subject" in pdf_metadata and pdf_metadata["subject"]:
                    metadata["subject"] = pdf_metadata["subject"]
                if "keywords" in pdf_metadata and pdf_metadata["keywords"]:
                    metadata["keywords"] = [k.strip() for k in pdf_metadata["keywords"].split(",")]
                if "creationDate" in pdf_metadata and pdf_metadata["creationDate"]:
                    try:
                        date_str = pdf_metadata["creationDate"]
                        if date_str.startswith("D:"):
                            date_str = date_str[2:]
                            if len(date_str) >= 8:
                                date = datetime.datetime.strptime(date_str[:8], "%Y%m%d")
                                metadata["creation_date"] = date.strftime("%Y-%m-%d")
                    except Exception as e:
                        logger.warning(f"Error parsing PDF creation date: {e}")
                        
            metadata["page_count"] = len(doc)
            
            metadata["file_size_bytes"] = os.path.getsize(pdf_path)
            metadata["file_size_mb"] = round(metadata["file_size_bytes"] / (1024 * 1024), 2)
            
            doc.close()
        except Exception as e:
            logger.error(f"Error extracting PDF metadata from {pdf_path}: {e}")
        
        return metadata
    
    def extract_from_text(self, text: str) -> Dict[str, Any]:
        """
        Extract metadata from document text.
        
        Args:
            text: Document text
            
        Returns:
            Dictionary of extracted metadata
        """
        metadata = {}
        
        for field, patterns in self.patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    if field in ["cfr_references", "iso_references"]:
                        metadata[field] = list(set(matches))
                        break
                    else:
                        metadata[field] = matches[0]
                        break
        
        if "document_date" in metadata:
            try:
                parsed_date = dateutil.parser.parse(metadata["document_date"])
                metadata["document_date"] = parsed_date.strftime("%Y-%m-%d")
            except Exception:
                pass
        
        if "title" not in metadata:
            first_lines = text.split('\n')[:10]
            for line in first_lines:
                line = line.strip()
                if len(line) > 20 and len(line) < 150 and not line.startswith('http'):
                    metadata["title"] = line
                    break
        
        if self.use_nlp and self.nlp:
            first_chunk = text[:min(10000, len(text))]
            doc = self.nlp(first_chunk)
            
            orgs = [ent.text for ent in doc.ents if ent.label_ == "ORG"]
            if orgs:
                metadata["organizations"] = list(set(orgs))
            
            self._extract_regulatory_entities(text, metadata)

            persons = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
            if persons:
                metadata["persons"] = list(set(persons))

            locations = [ent.text for ent in doc.ents if ent.label_ == "GPE" or ent.label_ == "LOC"]
            if locations:
                metadata["locations"] = list(set(locations))
            
            dates = [ent.text for ent in doc.ents if ent.label_ == "DATE"]
            if dates:
                metadata["dates_mentioned"] = list(set(dates))

        if text:
            metadata["document_hash"] = hashlib.md5(text.encode('utf-8')).hexdigest()
        
        return metadata
    
    def _extract_regulatory_entities(self, text: str, metadata: Dict[str, Any]) -> None:
        """
        Extract regulatory-specific entities from text.
        
        Args:
            text: Document text
            metadata: Metadata dictionary to update
        """
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
        Extract metadata from PDF file, combining PDF properties and text-based extraction.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Dictionary of extracted metadata
        """
        metadata = self.extract_pdf_metadata(pdf_path)
        
        try:
            doc = fitz.open(pdf_path)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()

            text_metadata = self.extract_from_text(text)

            for key, value in text_metadata.items():
                if key not in metadata:
                    metadata[key] = value

            file_path = Path(pdf_path)
            metadata["filename"] = file_path.name
            metadata["file_extension"] = file_path.suffix.lower()
            metadata["extraction_date"] = datetime.datetime.now().strftime("%Y-%m-%d")
            
        except Exception as e:
            logger.error(f"Error extracting text from PDF {pdf_path}: {e}")
        
        return metadata
    
    def save_metadata(self, metadata: Dict[str, Any], output_path: str) -> bool:
        """
        Save extracted metadata to JSON file.
        
        Args:
            metadata: Dictionary of metadata
            output_path: Path to save JSON file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved metadata to {output_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving metadata to {output_path}: {e}")
            return False

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
        else:
            try:
                with open(doc_path, 'r', encoding='utf-8', errors='ignore') as f:
                    text = f.read()
                metadata = self.extract_from_text(text)

                metadata["filename"] = file_path.name
                metadata["file_extension"] = file_path.suffix.lower()
                metadata["file_size_bytes"] = os.path.getsize(doc_path)
                metadata["file_size_mb"] = round(metadata["file_size_bytes"] / (1024 * 1024), 2)
                metadata["extraction_date"] = datetime.datetime.now().strftime("%Y-%m-%d")
                
            except Exception as e:
                logger.error(f"Error processing file {doc_path}: {e}")
                return {}

        if output_dir:
            output_path = Path(output_dir) / f"{file_path.stem}_metadata.json"
            self.save_metadata(metadata, str(output_path))
        
        return metadata
    
    def batch_process(self, input_dir: str, output_dir: str, file_types: List[str] = ['.pdf', '.txt']) -> List[Dict[str, Any]]:
        """
        Process all documents of specified types in a directory.
        
        Args:
            input_dir: Directory containing documents
            output_dir: Directory to save metadata files
            file_types: List of file extensions to process
            
        Returns:
            List of metadata dictionaries
        """
        os.makedirs(output_dir, exist_ok=True)
        
        results = []
        input_path = Path(input_dir)
        
        for file_type in file_types:
            for file_path in input_path.glob(f"*{file_type}"):
                logger.info(f"Processing {file_path}")
                metadata = self.process_document(str(file_path), output_dir)
                if metadata:
                    results.append(metadata)
        
        summary = {
            "processed_files": len(results),
            "processing_date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "file_types": file_types
        }
        
        summary_path = Path(output_dir) / "batch_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        
        return results
    
