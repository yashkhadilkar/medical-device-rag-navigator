import os 
import logging
import json 
from typing import List, Dict, Any, Optional, Set, Tuple
from pathlib import Path
import fitz
import re
import time
import requests
from urllib.parse import urlparse
import hashlib
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TextExtractor:
    """Extract text content from PDFs and web pages without storing the original files. 
    Designed for efficient storage and minimal desk usage.
    """
    
    def __init__(self, output_dir: str = "data/text_content", delay: float = 1.0, max_retries: int = 3):
        """
        Initialize the text extractor.
        
        Args:
            output_dir: Directory to save extracted text
            delay: Delay between requests in seconds
            max_retries: Maximum number of retry attempts
        """
        self.output_dir = output_dir
        self.delay = delay
        self.max_retries = max_retries

        os.makedirs(output_dir, exist_ok=True)

        for subdir in ["fda_guidance", "iso_standards", "cfr"]:
            os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)

        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        self.extracted_content = {}
        self.failed_extractions = []
        

    def extract_text_from_pdf_file(self, pdf_path: str) -> Dict[str, Any]:
        """
        Extract text from a local PDF file without storing the PDF.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary with extracted text and metadata
        """
        result = {
            "source_path": pdf_path,
            "source_filename": os.path.basename(pdf_path),
            "text": "",
            "metadata": {},
            "extraction_time": datetime.now().isoformat(),
            "success": False
        }
        
    def extract_text_from_pdf_file(self, pdf_path: str) -> Dict[str, Any]:
        """
        Extract text from a local PDF file without storing the PDF.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary with extracted text and metadata
        """
        result = {
            "source_path": pdf_path,
            "source_filename": os.path.basename(pdf_path),
            "text": "",
            "metadata": {},
            "extraction_time": datetime.now().isoformat(),
            "success": False
        }
        
        try:
            logger.info(f"Extracting text from: {pdf_path}")

            doc = fitz.open(pdf_path)

            pdf_metadata = doc.metadata
            result["metadata"] = {
                "title": pdf_metadata.get("title", ""),
                "author": pdf_metadata.get("author", ""),
                "subject": pdf_metadata.get("subject", ""),
                "keywords": pdf_metadata.get("keywords", ""),
                "page_count": len(doc)
            }

            full_text = ""
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                page_text = page.get_text()
                full_text += page_text + "\n\n"

            result["text"] = full_text
            result["success"] = True

            result["metadata"]["file_size"] = os.path.getsize(pdf_path)
            result["metadata"]["extraction_date"] = datetime.now().strftime("%Y-%m-%d")

            doc_number = self._extract_document_number(full_text)
            if doc_number:
                result["metadata"]["document_number"] = doc_number

            pub_date = self._extract_document_date(full_text)
            if pub_date:
                result["metadata"]["publication_date"] = pub_date

            doc_type = self._extract_document_type(full_text)
            if doc_type:
                result["metadata"]["document_type"] = doc_type

            doc.close()
            
            return result
        
        except Exception as e:
            logger.error(f"Error extracting text from {pdf_path}: {e}")
            result["error"] = str(e)
            return result

    def extract_text_from_pdf_url(self, url: str, category: str = "fda_guidance", 
                                retry_count: int = 0) -> Dict[str, Any]:
        """
        Download a PDF from a URL, extract its text, and discard the PDF.
        
        Args:
            url: URL of the PDF file
            category: Document category
            retry_count: Current retry attempt
            
        Returns:
            Dictionary with extracted text and metadata
        """
        result = {
            "source_url": url,
            "category": category,
            "text": "",
            "metadata": {},
            "extraction_time": datetime.now().isoformat(),
            "success": False
        }
        
        try:
            logger.info(f"Downloading and extracting text from: {url}")
            
            response = self.session.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            url_hash = hashlib.md5(url.encode()).hexdigest()
            temp_path = os.path.join(self.output_dir, f"temp_{url_hash}.pdf")

            with open(temp_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

            result = self.extract_text_from_pdf_file(temp_path)

            result["source_url"] = url
            result["category"] = category

            try:
                os.remove(temp_path)
            except OSError:
                pass

            result["extraction_time"] = datetime.now().isoformat()

            time.sleep(self.delay)
            
            return result
        
        except requests.exceptions.RequestException as e:
            logger.error(f"Error downloading PDF from {url}: {e}")

            if retry_count < self.max_retries:
                retry_count += 1
                wait_time = retry_count * 2
                logger.info(f"Retrying in {wait_time} seconds... (Attempt {retry_count}/{self.max_retries})")
                time.sleep(wait_time)
                return self.extract_text_from_pdf_url(url, category, retry_count)
            else:
                logger.error(f"Failed to download after {self.max_retries} attempts: {url}")
                result["error"] = str(e)
                self.failed_extractions.append((url, str(e)))
                return result
                
        except Exception as e:
            logger.error(f"Error processing {url}: {e}")
            result["error"] = str(e)
            self.failed_extractions.append((url, str(e)))
            return result
        
    def save_extracted_text(self, result: Dict[str, Any], output_filename: Optional[str] = None) -> str:
        """
        Save extracted text and metadata to a JSON file.
        
        Args:
            result: Extraction result dictionary
            output_filename: Optional custom filename for output
            
        Returns:
            Path to the saved file
        """
        if not result.get("success", False):
            logger.warning("Not saving unsuccessful extraction result")
            return ""

        if not output_filename:
            if "document_number" in result.get("metadata", {}):
                base_name = result["metadata"]["document_number"]
            elif "source_filename" in result:
                base_name = os.path.splitext(result["source_filename"])[0]
            else:
                url = result.get("source_url", "")
                if url:
                    base_name = hashlib.md5(url.encode()).hexdigest()
                else:
                    base_name = f"document_{int(time.time())}"
            
            output_filename = f"{base_name}.json"

        category = result.get("category", "fda_guidance")
        subdir = category if category in ["fda_guidance", "iso_standards", "cfr"] else "other"
        
        output_path = os.path.join(self.output_dir, subdir, output_filename)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved extracted text to: {output_path}")

        self.extracted_content[output_path] = {
            "path": output_path,
            "size": os.path.getsize(output_path),
            "extraction_time": result["extraction_time"]
        }
        
        return output_path
    
    def extract_multiple_pdfs(self, urls: List[str], category: str = "fda_guidance", 
                             parallel: bool = False) -> List[str]:
        """
        Extract text from multiple PDF URLs.
        
        Args:
            urls: List of PDF URLs
            category: Document category
            parallel: Whether to use parallel processing (not implemented)
            
        Returns:
            List of paths to saved text files
        """
        output_paths = []
        
        for url in urls:
            result = self.extract_text_from_pdf_url(url, category)
            
            if result["success"]:
                path = self.save_extracted_text(result)
                if path:
                    output_paths.append(path)
        
        logger.info(f"Extracted text from {len(output_paths)} of {len(urls)} PDFs")
        return output_paths
        
    def extract_fda_guidance_documents(self, urls: List[str]) -> List[str]:
        """
        Extract text from FDA guidance document PDFs.
        
        Args:
            urls: List of FDA guidance document URLs
            
        Returns:
            List of paths to saved text files
        """
        return self.extract_multiple_pdfs(urls, category="fda_guidance")
    
    def extract_cfr_documents(self, urls: List[str]) -> List[str]:
        """
        Extract text from CFR document PDFs.
        
        Args:
            urls: List of CFR document URLs
            
        Returns:
            List of paths to saved text files
        """
        return self.extract_multiple_pdfs(urls, category="cfr")
    
    def extract_iso_standards(self, urls: List[str]) -> List[str]:
        """
        Extract text from ISO standard PDFs.
        
        Args:
            urls: List of ISO standard URLs
            
        Returns:
            List of paths to saved text files
        """
        return self.extract_multiple_pdfs(urls, category="iso_standards")
    
    def _extract_document_number(self, text: str) -> str:
        """
        Extract document number from text.
        
        Args:
            text: Document text
            
        Returns:
            Document number if found, otherwise empty string
        """
        patterns = [
            r'Document Number:\s*([A-Za-z0-9\-\.]+)',
            r'Document No\.?\s*([A-Za-z0-9\-\.]+)',
            r'FDA-\d{4}-[A-Za-z]+-\d+',
            r'(\d{8}dft)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                if '(' in pattern:  # If we used a capture group
                    return match.group(1).strip()
                else:
                    return match.group(0).strip()
        
        return ""
    
    def _extract_document_date(self, text: str) -> str:
        """
        Extract publication date from document text.
        
        Args:
            text: Document text
            
        Returns:
            Publication date if found, otherwise empty string
        """
        patterns = [
            r'(?:Issued|Published|Effective)(?:\s+on)?(?:\s+date)?[:\s]+([A-Za-z]+\s+\d{1,2},?\s+\d{4})',
            r'(?:Date of Issuance|Publication Date)[:\s]+([A-Za-z]+\s+\d{1,2},?\s+\d{4})',
            r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}',
            r'\b\d{1,2}/\d{1,2}/\d{4}\b',
            r'\b\d{1,2}-\d{1,2}-\d{4}\b'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    return match.group(1).strip()
                except IndexError:
                    return match.group(0).strip()
        
        return ""
    
    def _extract_document_type(self, text: str) -> str:
        """
        Extract document type from text.
        
        Args:
            text: Document text
            
        Returns:
            Document type if found, otherwise empty string
        """
        start_text = text[:1000].lower()
        
        if "guidance for industry" in start_text:
            return "Guidance for Industry"
        elif "draft guidance" in start_text:
            return "Draft Guidance"
        elif "final guidance" in start_text:
            return "Final Guidance"
        elif "technical considerations" in start_text:
            return "Technical Considerations"
        elif "guidance" in start_text:
            return "Guidance Document"
        
        return ""
    
    def generate_extraction_report(self) -> str:
        """
        Generate a report of extracted content.
        
        Returns:
            Path to the report file
        """
        report = {
            "extraction_summary": {
                "total_files": len(self.extracted_content),
                "total_size_mb": sum(item["size"] for item in self.extracted_content.values()) / (1024 * 1024),
                "failed_extractions": len(self.failed_extractions),
                "failed_urls": [{"url": url, "error": error} for url, error in self.failed_extractions]
            },
            "extracted_files": list(self.extracted_content.values())
        }
        
        report_path = os.path.join(self.output_dir, f"extraction_report_{time.strftime('%Y%m%d_%H%M%S')}.json")
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Generated extraction report: {report_path}")
        return report_path
    

if __name__ == "__main__":
    extractor = TextExtractor()
    
    # Example URL of FDA guidance document
    sample_url = "https://www.fda.gov/media/82395/download" 
    
    # Extract text from sample URL
    result = extractor.extract_text_from_pdf_url(sample_url)
    
    if result["success"]:
        # Save extracted text
        output_path = extractor.save_extracted_text(result)
        print(f"Saved extracted text to: {output_path}")
        
        # Show document info
        print(f"Document number: {result['metadata'].get('document_number', 'N/A')}")
        print(f"Publication date: {result['metadata'].get('publication_date', 'N/A')}")
        print(f"Document type: {result['metadata'].get('document_type', 'N/A')}")
        print(f"Text length: {len(result['text'])} characters")
    else:
        print("Extraction failed:", result.get("error", "Unknown error"))
