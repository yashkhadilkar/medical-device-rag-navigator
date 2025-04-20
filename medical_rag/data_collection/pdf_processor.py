import os 
import PyPDF2 
import re 
from typing import Dict, Any, List, Optional 
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PDFProcessor:
    """Process PDF files containing FDA guidance and regulatory info.
    """
    
    def __init__(self, output_dir: str = "data/processed"):
        """Initialize the PDF processor. 

        Args:
            output_dir: Directory to save processed content
        """
        
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def extract_text_from_pdf(self, pdf_path: str) -> Optional[str]:
        """Extract text content from a PDF file

        Args:
            pdf_path: Path to the PDF file

        Returns:
            Extracted text or None if extraction failed
        """
        
        if not os.path.exists(pdf_path):
            logger.error(f"PDF file not found: {pdf_path}")
            return None
        
        try:
            logger.info(f"Extracting text from: {pdf_path}")
            text = ""
            
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                num_pages = len(pdf_reader.pages)
                
                logger.info(f"PDF has {num_pages} pages")
                
                for page_num in range(num_pages):
                    page = pdf_reader.pages[page_num]
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n\n"
                        
            return text
        except Exception as e:
            logger.error(f"Error extracting text from {pdf_path}: {e}")
            return None
        
    def extract_metadata_from_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """Extract metadata from a PDF file. 

        Args:
            pdf_path: PDF Path

        Returns:
            Dictionary containing PDF metadata
        """
        
        metadata = {
            'filename': os.path.basename(pdf_path), 
            'path': pdf_path,
            'size_bytes': os.path.getsize(pdf_path),
            'title': '',
            'author': '',
            'subject': '',
            'creation_date': '',
            'modification_date': '',
            'page_count': 0
        }
        
        try: 
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                info = pdf_reader.metadata
                metadata['page_count'] = len(pdf_reader.pages)
                
                if info:
                    metadata['title'] = info.title if hasattr(info, 'title') and info.title else ''
                    metadata['author'] = info.author if hasattr(info, 'author') and info.author else ''
                    metadata['subject'] = info.subject if hasattr(info, 'subject') and info.subject else ''
                    metadata['creation_date'] = info.creation_date if hasattr(info, 'creation_date') and info.creation_date else ''
                    metadata['modification_date'] = info.modification_date if hasattr(info, 'modification_date') and info.modification_date else ''
                    
            return metadata
        except Exception as e:
            logger.error(f"Error extracting metadata from {pdf_path}: {e}")
            return metadata       
        
        
    def extract_document_number(self, text: str) -> str:
        """Extract FDA document number from text.

        Args:
            text: Text content to search

        Returns:
            Document number if found, otherwise empty string
        """
        
        common_patterns = [
            r'Document Number:\s*([A-Za-z0-9\-]+)', 
            r'Document No\.?\s*([A-Za-z0-9\-]+)', 
            r'FDA-\d{4}-[A-Za-z]+-\d+',
            r'(\d{8}dft)'  # Pattern for the specific format in your PDF (45993101dft)
        ]
        
        # First check first 1000 characters for draft number
        first_part = text[:1000]
        for pattern in common_patterns:
            matches = re.search(pattern, first_part)
            if matches:
                if '(' in pattern:  # If we used a capture group
                    return matches.group(1).strip()
                else:
                    return matches.group(0).strip()
        
        # Then check the whole document
        for pattern in common_patterns:
            matches = re.search(pattern, text)
            if matches:
                if '(' in pattern:  # If we used a capture group
                    return matches.group(1).strip()
                else:
                    return matches.group(0).strip()
                
        return ""
    
    def extract_guidance_date(self, text: str) -> str:
        """Extract publication date from guidance document text.

        Args:
            text: Text content to search

        Returns:
            Publication date if found, otherwise empty string
        """
        # Add pattern for month and year format (January 2025)
        date_patterns = [
            r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}',
            r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}',
            r'\b\d{1,2}/\d{1,2}/\d{4}\b',
            r'\b\d{1,2}-\d{1,2}-\d{4}\b',
        ]
        
        # Look for dates near certain keywords
        context_patterns = [
            r'(?:Issued|Published|Effective)(?:\s+on)?(?:\s+date)?[:\s]+([A-Za-z]+\s+\d{1,2},?\s+\d{4})',
            r'(?:Date of Issuance|Publication Date)[:\s]+([A-Za-z]+\s+\d{1,2},?\s+\d{4})',
            r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}'
        ]
        
        # Look specifically for dates in the header or footer
        header_footer_patterns = [
            r'U\.S\. Department of Health and Human Services.*\n.*\n.*([A-Za-z]+\s+\d{4})',
        ]
        
        # Check header/footer patterns
        for pattern in header_footer_patterns:
            matches = re.search(pattern, text, re.IGNORECASE)
            if matches:
                return matches.group(1).strip()
        
        # Check context patterns
        for pattern in context_patterns:
            matches = re.search(pattern, text, re.IGNORECASE)
            if matches:
                # Check if group exists in the match
                try:
                    return matches.group(1).strip()
                except IndexError:
                    return matches.group(0).strip()
        
        # Using first 3000 characters in first few pages to get date
        first_part = text[:3000]
        for pattern in date_patterns:
            matches = re.search(pattern, first_part)
            if matches:
                return matches.group(0).strip()
            
        return ""
    
    def process_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """Processing a PDF file to extract content and meta data.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            Dictionary with extracted content and metadata
        """
        result = {
            'path': pdf_path, 
            'filename': os.path.basename(pdf_path),
            'text': '',
            'metadata': {},
            'document_number': '',
            'publication_date': '',
            'processing_success': False
        }
        
        text = self.extract_text_from_pdf(pdf_path)
        if not text:
            return result
        
        result['text'] = text
        result['metadata'] = self.extract_metadata_from_pdf(pdf_path)
        
        result['document_number'] = self.extract_document_number(text)
        result['publication_date'] = self.extract_guidance_date(text)
        
        if not result['metadata']['title']:
            filename_without_ext = os.path.splitext(result['filename'])[0]
            
            clean_title = ' '.join(word.capitalize() for word in filename_without_ext.replace('_', ' ').split())
            result['metadata']['title'] = clean_title
            
        result['processing_success'] = True
        
        return result
    
    def save_processed_content(self, content: Dict[str, Any], output_filename: Optional[str] = None) -> str:
        """Save processed content to text file. 

        Args:
            content: Processed content dictionary 
            output_filename: Optional custom filename for output

        Returns:
            Path saved to file.
        """
        
        if not output_filename:
            base_name = content['document_number'] if content['document_number'] else os.path.splitext(content['filename'])[0]
            output_filename = f"{base_name}.txt"
            
        output_path = os.path.join(self.output_dir, output_filename)
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"Title: {content['metadata'].get('title', '')}\n")
            f.write(f"Document Number: {content['document_number']}\n")
            f.write(f"Publication Date: {content['publication_date']}\n")
            f.write(f"Source: {content['path']}\n")
            f.write(f"Page Count: {content['metadata'].get('page_count', 0)}\n")
            f.write("\n" + "="*50 + "\n\n")
            
            f.write(content['text'])
            
        logger.info(f"Saved processed content to: {output_path}")
        return output_path
    
    def batch_process_directory(self, input_dir: str) -> List[Dict[str, Any]]:
        """Process all PDF files in a directory. 

        Args:
            input_dir: Directory containing PDF files.

        Returns:
            List of processing results
        """
        
        results = []
        
        if not os.path.exists(input_dir):
            logger.error(f"Input directory not found: {input_dir}")
            return results
        
        for filename in os.listdir(input_dir):
            if filename.lower().endswith('.pdf'):
                pdf_path = os.path.join(input_dir, filename)
                result = self.process_pdf(pdf_path)
                
                if result['processing_success']:
                    self.save_processed_content(result)
                    results.append(result)
                    
        logger.info(f"Processed {len(results)} PDF files from {input_dir}")
        return results
    
    
if __name__ == "__main__":
    processor = PDFProcessor(output_dir="data/processed/fda_guidance")
    
    sample_pdf = "data/raw/fda_guidance/sample_guidance.pdf"
    if os.path.exists(sample_pdf):
        result = processor.process_pdf(sample_pdf)
        if result['processing_success']:
            processor.save_processed_content(result)
            print(f"Successfully processed: {sample_pdf}")
            print(f"Document number: {result['document_number']}")
            print(f"Publication date: {result['publication_date']}")
            
    else:
        print(f"Sample PDF not found. Please download PDF guidance documents to {os.path.dirname(sample_pdf)}")
        
    input_dir = "data/raw/fda_guidance"
    if os.path.exists(input_dir) and os.listdir(input_dir):
        results = processor.batch_process_directory(input_dir)
        print(f"Processed {len(results)} PDF files")
