import os 
import requests
import time 
import logging
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import json 
from urllib.parse import urlparse
from tqdm import tqdm
import re 
import concurrent.futures

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataDownloader:
    """Download FDA regulatory documents and data.
    """
    
    def __init__(self, output_dir: str = "data/raw", delay: float = 1.0, max_retries: int = 3):
        """Initialize the data downloader.

        Args:
            output_dir: Base directory to save downloaded files
            delay: Delay between requests in seconds
            max_retries: Maximum number of retry attempts for failed downloads
        """
        
        self.output_dir = output_dir
        self.dellay = delay
        self.max_retries = max_retries
        self.session = requests.Session()
        
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "fda_guidance"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "iso_standards"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "cfr"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "metadata"), exist_ok=True)
        
        self.downloaded_files = set()
        
        self.failed_downloads = []
        
    def download_file(self, url: str, output_path: Optional[str] = None, retry_count: int = 0) -> Optional[str]:
        """Download a file from a url.

        Args:
            url:  URL of the file to download
            output_path: Path to save the file (if None, determined from URL)
            retry_count: Current retry attempt

        Returns:
            Optional[str]: _description_
        """
        if not output_path:
            filename = os.path.basename(urlparse(url).path)
            if not filename:
                filename = f"downloaded_file_{int(time.time())}"
            
            output_path = os.path.join(self.output_dir, filename)
        
        if os.path.exists(output_path):
            logger.info(f"File already exists: {output_path}")
            self.downloaded_files.add(output_path)
            return output_path
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        try:
            logger.info(f"Downloading: {url}")
            response = self.session.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            file_size = int(response.headers.get('content-length', 0))
            
            with open(output_path, 'wb') as f:
                if file_size > 1024*1024:  
                    with tqdm(total=file_size, unit='B', unit_scale=True, desc=os.path.basename(output_path)) as pbar:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                                pbar.update(len(chunk))
                else:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
            
            logger.info(f"Successfully downloaded to: {output_path}")
            self.downloaded_files.add(output_path)
            
            time.sleep(self.delay)
            
            return output_path
        except requests.exceptions.RequestException as e:
            logger.error(f"Error downloading {url}: {e}")
            
            if retry_count < self.max_retries:
                retry_count += 1
                wait_time = retry_count * 2 
                logger.info(f"Retrying in {wait_time} seconds... (Attempt {retry_count}/{self.max_retries})")
                time.sleep(wait_time)
                return self.download_file(url, output_path, retry_count)
            else:
                logger.error(f"Failed to download after {self.max_retries} attempts: {url}")
                self.failed_downloads.append((url, str(e)))
                return None
            
    def download_pdf(self, url: str, subfolder: str = "fda_guidance") -> Optional[str]:
        """Download a PDF file and save it to a specific folder. 

        Args:
            url: URL of the PDF file
            subfolder: Subfolder within output_dir to save the file

        Returns:
            Path to the downloaded PDF or None if failed. 
        """
        
        filename = os.path.basename(urlparse(url).path)
        if not filename or filename.lower() == "download":
            # Try extracting the ID (e.g., 82395) from the URL
            match = re.search(r'/media/(\d+)/', url)
            if match:
                filename = f"{match.group(1)}.pdf"
            else:
                filename = f"document_{int(time.time())}.pdf"
    
    def download_multiple_pdfs(self, urls: List[str], subfolder: str = "fda_guidance", parallel: bool = False) -> List[str]:
        """
        Download multiple PDF files.
        
        Args:
            urls: List of PDF URLs
            subfolder: Subfolder to save the files
            parallel: Whether to download files in parallel
            
        Returns:
            List of paths to downloaded files
        """
        downloaded_files = []
        
        if parallel and len(urls) > 5: 
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                future_to_url = {executor.submit(self.download_pdf, url, subfolder): url for url in urls}
                
                for future in tqdm(concurrent.futures.as_completed(future_to_url), total=len(urls), desc=f"Downloading {subfolder} files"):
                    url = future_to_url[future]
                    try:
                        file_path = future.result()
                        if file_path:
                            downloaded_files.append(file_path)
                    except Exception as e:
                        logger.error(f"Error downloading {url}: {e}")
                        self.failed_downloads.append((url, str(e)))
        else:
            for url in tqdm(urls, desc=f"Downloading {subfolder} files"):
                file_path = self.download_pdf(url, subfolder)
                if file_path:
                    downloaded_files.append(file_path)
        
        logger.info(f"Downloaded {len(downloaded_files)} of {len(urls)} PDFs to {subfolder}")
        return downloaded_files
    
    def download_iso_standard_summaries(self, urls: List[str], parallel: bool = False) -> List[str]:
        """Download ISO standard summaries.

        Args:
            urls: List of ISO standard summary URLs
            parallel: Whether to download files in parallel

        Returns:
            List of paths to downloaded files.
        """
        return self.download_multiple_pdfs(urls, subfolder="iso_standards", parallel=parallel)
    
    def download_cfr_documents(self, urls: List[str], parallel: bool = False) -> List[str]:
        """
        Download Code of Federal Regulations documents.
        
        Args:
            urls: List of CFR document URLs
            parallel: Whether to download files in parallel
            
        Returns:
            List of paths to downloaded files
        """
        return self.download_multiple_pdfs(urls, subfolder="cfr", parallel=parallel) 
    
    def save_metadata(self, metadata: List[Dict[str, Any]], filename: str) -> str:
        """
        Save metadata to a JSON file.
        
        Args:
            metadata: List of metadata dictionaries
            filename: Filename for the JSON file
            
        Returns:
            Path to the saved file
        """
        if not filename.endswith('.json'):
            filename += '.json'
            
        output_path = os.path.join(self.output_dir, "metadata", filename)
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved metadata to: {output_path}")
        return output_path     
    
    def get_fda_guidance_urls(self) -> List[str]:
        """
        Get URLs for FDA guidance documents about medical devices.
        
        Returns:
            List of URLs for guidance document PDFs
        """
        guidance_pages = [
            "https://www.fda.gov/medical-devices/device-advice-comprehensive-regulatory-assistance/guidance-documents-medical-devices-and-radiation-emitting-products",
            "https://www.fda.gov/medical-devices/pre-submissions-and-meetings/pre-submission-program-feedback-medical-devices",
            "https://www.fda.gov/medical-devices/premarket-notification-510k/510k-forms",
            "https://www.fda.gov/medical-devices/device-advice-comprehensive-regulatory-assistance/how-study-and-market-your-device",
            "https://www.fda.gov/medical-devices/software-medical-device-samd/digital-health-policies-and-public-health-solutions",
            "https://www.fda.gov/medical-devices/device-software-functions-including-mobile-medical-applications/policy-device-software-functions-and-mobile-medical-applications"
        ]
        
        pdf_urls = []
        
        for page_url in tqdm(guidance_pages, desc="Searching FDA pages for guidance documents"):
            try:
                logger.info(f"Searching for PDFs on: {page_url}")
                response = self.session.get(page_url, timeout=30)
                response.raise_for_status()
                
                content = response.text
                pdf_links = re.findall(r'href=[\'"]([^\'"]+\.pdf)[\'"]', content)
                
                for link in pdf_links:
                    if not link.startswith(('http://', 'https://')):
                        if link.startswith('/'):
                            link = f"https://www.fda.gov{link}"
                        else:
                            link = f"{os.path.dirname(page_url)}/{link}"
                    
                    pdf_urls.append(link)
                
                time.sleep(self.delay)
                
            except requests.exceptions.RequestException as e:
                logger.error(f"Error accessing {page_url}: {e}")
                self.failed_downloads.append((page_url, str(e)))
        
        pdf_urls = list(set(pdf_urls))
        
        logger.info(f"Found {len(pdf_urls)} unique PDF guidance documents")
        return pdf_urls
    
    def get_primary_fda_guidance_urls(self) -> List[str]:
        """Get URLs for primary FDA guidance documents for medical devices.

        Returns:
            List of URLs for key guidance document PDFs.
        """
        
        primary_urls = [
            "https://www.fda.gov/media/82395/download",  # The 510(k) Program: Evaluating Substantial Equivalence
            "https://www.fda.gov/media/83888/download",  # Refuse to Accept Policy for 510(k)s
            "https://www.fda.gov/media/89179/download",  # Benefit-Risk Factors to Consider When Determining Substantial Equivalence
            "https://www.fda.gov/media/71975/download",  # Guidance on Medical Device Patient Labeling
            "https://www.fda.gov/media/71018/download",  # Design Considerations for Pivotal Clinical Investigations
            "https://www.fda.gov/media/72768/download",  # Factors to Consider When Making Benefit-Risk Determinations in Medical Device Premarket Approval
            "https://www.fda.gov/media/97071/download",  # Deciding When to Submit a 510(k) for a Change to an Existing Device
            "https://www.fda.gov/media/81431/download",  # De Novo Classification Process (Evaluation of Automatic Class III Designation)
            "https://www.fda.gov/media/70702/download",  # Software as a Medical Device (SAMD): Clinical Evaluation
            "https://www.fda.gov/media/80958/download",  # The 510(k) Program: Evaluating Substantial Equivalence in Premarket Notifications
            "https://www.fda.gov/media/144458/download",  # Safer Technologies Program for Medical Devices
            "https://www.fda.gov/media/163660/download",  # Cybersecurity in Medical Devices
            "https://www.fda.gov/media/108834/download",  # Technical Considerations for Additive Manufactured Medical Devices
            "https://www.fda.gov/media/150003/download",  # Computer Software Assurance for Production and Quality System Software
            "https://www.fda.gov/media/145050/download"   # Clinical Decision Support Software
        ]
        
        return primary_urls
    
    def get_cfr_urls(self) -> List[str]:
        """
        Get URLs for relevant Code of Federal Regulations (CFR) documents for medical devices.
        
        Returns:
            List of URLs for CFR PDFs
        """
        cfr_urls = [
            # 21 CFR Part 800-1299 (Medical Devices)
            "https://www.govinfo.gov/content/pkg/CFR-2020-title21-vol8/pdf/CFR-2020-title21-vol8.pdf",  # Parts 800-1299 (Medical Devices)
            "https://www.govinfo.gov/content/pkg/CFR-2020-title21-vol1/pdf/CFR-2020-title21-vol1.pdf",  # Parts 1-99 (FDA General)
            "https://www.govinfo.gov/content/pkg/CFR-2020-title21-vol5/pdf/CFR-2020-title21-vol5.pdf",  # Parts 300-499 (Drugs)
            
            # Specific important parts for medical devices
            "https://www.accessdata.fda.gov/scripts/cdrh/cfdocs/cfcfr/CFRSearch.cfm?CFRPart=807",  # Part 807 (Registration and Listing)
            "https://www.accessdata.fda.gov/scripts/cdrh/cfdocs/cfcfr/CFRSearch.cfm?CFRPart=820",  # Part 820 (Quality System Regulation)
            "https://www.accessdata.fda.gov/scripts/cdrh/cfdocs/cfcfr/CFRSearch.cfm?CFRPart=801",  # Part 801 (Labeling)
            "https://www.accessdata.fda.gov/scripts/cdrh/cfdocs/cfcfr/CFRSearch.cfm?CFRPart=803",  # Part 803 (Medical Device Reporting)
            "https://www.accessdata.fda.gov/scripts/cdrh/cfdocs/cfcfr/CFRSearch.cfm?CFRPart=812",  # Part 812 (Investigational Device Exemptions)
            "https://www.accessdata.fda.gov/scripts/cdrh/cfdocs/cfcfr/CFRSearch.cfm?CFRPart=814",  # Part 814 (Premarket Approval)
            "https://www.accessdata.fda.gov/scripts/cdrh/cfdocs/cfcfr/CFRSearch.cfm?CFRPart=860"   # Part 860 (Medical Device Classification Procedures)
        ]
        
        return cfr_urls
    
    def get_iso_standards_urls(self) -> List[Dict[str, Any]]:
        """
        Get information about relevant ISO standards for medical devices.
        
        Note: Many ISO standards need to be purchased. This method returns
        information about the standards, not direct download links.
        
        Returns:
            List of dictionaries containing standard info
        """
        # Key ISO standards for medical devices
        iso_standards = [
            {
                "standard": "ISO 13485:2016",
                "title": "Medical devices — Quality management systems — Requirements for regulatory purposes",
                "url": "https://www.iso.org/standard/59752.html",
                "description": "Specifies requirements for a quality management system where an organization needs to demonstrate its ability to provide medical devices and related services"
            },
            {
                "standard": "ISO 14971:2019",
                "title": "Medical devices — Application of risk management to medical devices",
                "url": "https://www.iso.org/standard/72704.html",
                "description": "Specifies terminology, principles and a process for risk management of medical devices"
            },
            {
                "standard": "IEC 62304:2006",
                "title": "Medical device software — Software life cycle processes",
                "url": "https://www.iso.org/standard/38421.html",
                "description": "Defines the life cycle requirements for medical device software"
            },
            {
                "standard": "ISO 10993-1:2018", 
                "title": "Biological evaluation of medical devices — Part 1: Evaluation and testing within a risk management process",
                "url": "https://www.iso.org/standard/68936.html",
                "description": "Describes the general principles governing the biological evaluation of medical devices within a risk management process"
            },
            {
                "standard": "IEC 60601-1:2020",
                "title": "Medical electrical equipment — Part 1: General requirements for basic safety and essential performance",
                "url": "https://www.iso.org/standard/76026.html",
                "description": "Specifies general requirements for electrical medical equipment safety"
            }
        ]
        
        # Save this as metadata
        self.save_metadata(iso_standards, "iso_standards_metadata.json")
        
        # Return just the URLs
        return [item["url"] for item in iso_standards]
    
    def extract_document_metadata(self, file_path: str) -> Dict[str, Any]:
        """Extract metadata from a downloaded document.

        Args:
            file_path: Path to the document

        Returns:
            Dictionary containing document metadata
        """
        
        file_name = os.path.basename(file_path)
        file_size = os.path.getsize(file_path)
        
        metadata = {
            "file_name": file_name,
            "file_path": file_path,
            "file_size_bytes": file_size,
            "file_size_mb": round(file_size / (1024 * 1024), 2),
            "download_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "file_type": os.path.splitext(file_name)[1].lower()[1:],
            "category": self._determine_category(file_path)            
        }
        
        return metadata
    
    def determine_category(self, file_path: str) -> str:
        """Determine the category of a document based on its path. 

        Args:
            file_path: Path to the document

        Returns:
            Category name (fda_guidance, iso_standards, cfr, or unknown)
        """
        

# medical_rag/data_collection/data_downloader.py
import os
import requests
import time
import logging
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import json
from urllib.parse import urlparse
from tqdm import tqdm
import re
import concurrent.futures

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataDownloader:
    """Download FDA regulatory documents and data."""
    
    def __init__(self, output_dir: str = "data/raw", delay: float = 1.0, max_retries: int = 3):
        """
        Initialize the data downloader.
        
        Args:
            output_dir: Base directory to save downloaded files
            delay: Delay between requests in seconds
            max_retries: Maximum number of retry attempts for failed downloads
        """
        self.output_dir = output_dir
        self.delay = delay
        self.max_retries = max_retries
        self.session = requests.Session()
        # Set a user agent to avoid being blocked
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "fda_guidance"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "iso_standards"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "cfr"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "metadata"), exist_ok=True)
        
        # Track downloaded files to avoid duplicates
        self.downloaded_files = set()
        
        # Track failed downloads for reporting
        self.failed_downloads = []
    
    def download_file(self, url: str, output_path: Optional[str] = None, retry_count: int = 0) -> Optional[str]:
        """
        Download a file from a URL.
        
        Args:
            url: URL of the file to download
            output_path: Path to save the file (if None, determined from URL)
            retry_count: Current retry attempt
            
        Returns:
            Path to the downloaded file or None if failed
        """
        if not output_path:
            # Extract filename from URL
            filename = os.path.basename(urlparse(url).path)
            if not filename:
                filename = f"downloaded_file_{int(time.time())}"
            
            output_path = os.path.join(self.output_dir, filename)
        
        # Check if file already exists
        if os.path.exists(output_path):
            logger.info(f"File already exists: {output_path}")
            self.downloaded_files.add(output_path)
            return output_path
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        try:
            logger.info(f"Downloading: {url}")
            response = self.session.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            # Get file size if available
            file_size = int(response.headers.get('content-length', 0))
            
            # Save the file with progress bar for larger files
            with open(output_path, 'wb') as f:
                if file_size > 1024*1024:  # Only show progress for files > 1MB
                    with tqdm(total=file_size, unit='B', unit_scale=True, desc=os.path.basename(output_path)) as pbar:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                                pbar.update(len(chunk))
                else:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
            
            logger.info(f"Successfully downloaded to: {output_path}")
            self.downloaded_files.add(output_path)
            
            # Delay to avoid overwhelming the server
            time.sleep(self.delay)
            
            return output_path
        except requests.exceptions.RequestException as e:
            logger.error(f"Error downloading {url}: {e}")
            
            # Retry logic
            if retry_count < self.max_retries:
                retry_count += 1
                wait_time = retry_count * 2  # Exponential backoff
                logger.info(f"Retrying in {wait_time} seconds... (Attempt {retry_count}/{self.max_retries})")
                time.sleep(wait_time)
                return self.download_file(url, output_path, retry_count)
            else:
                logger.error(f"Failed to download after {self.max_retries} attempts: {url}")
                self.failed_downloads.append((url, str(e)))
                return None
    
    def download_pdf(self, url: str, subfolder: str = "fda_guidance") -> Optional[str]:
        """
        Download a PDF file and save it to a specific subfolder.
        
        Args:
            url: URL of the PDF file
            subfolder: Subfolder within output_dir to save the file
            
        Returns:
            Path to the downloaded PDF or None if failed
        """
        # Handle FDA's URLs that end with /download but point to PDFs
        if url.endswith('/download'):
            # Extract ID from URLs like https://www.fda.gov/media/82395/download
            match = re.search(r'/media/(\d+)/', url)
            if match:
                filename = f"{match.group(1)}.pdf"
            else:
                filename = f"document_{int(time.time())}.pdf"
        else:
            # Standard URL handling
            filename = os.path.basename(urlparse(url).path)
            if not filename:
                filename = f"document_{int(time.time())}.pdf"
            elif not filename.lower().endswith('.pdf'):
                filename += '.pdf'
        
        output_path = os.path.join(self.output_dir, subfolder, filename)
        
        return self.download_file(url, output_path)
    
    def download_multiple_pdfs(self, urls: List[str], subfolder: str = "fda_guidance", parallel: bool = False) -> List[str]:
        """
        Download multiple PDF files.
        
        Args:
            urls: List of PDF URLs
            subfolder: Subfolder to save the files
            parallel: Whether to download files in parallel
            
        Returns:
            List of paths to downloaded files
        """
        downloaded_files = []
        
        if parallel and len(urls) > 5:  # Only use parallel for larger batches
            # Use ThreadPoolExecutor for parallel downloads
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                # Create a dictionary mapping futures to URLs for tracking
                future_to_url = {executor.submit(self.download_pdf, url, subfolder): url for url in urls}
                
                # Process completed futures
                for future in tqdm(concurrent.futures.as_completed(future_to_url), total=len(urls), desc=f"Downloading {subfolder} files"):
                    url = future_to_url[future]
                    try:
                        file_path = future.result()
                        if file_path:
                            downloaded_files.append(file_path)
                    except Exception as e:
                        logger.error(f"Error downloading {url}: {e}")
                        self.failed_downloads.append((url, str(e)))
        else:
            # Sequential downloads
            for url in tqdm(urls, desc=f"Downloading {subfolder} files"):
                file_path = self.download_pdf(url, subfolder)
                if file_path:
                    downloaded_files.append(file_path)
        
        logger.info(f"Downloaded {len(downloaded_files)} of {len(urls)} PDFs to {subfolder}")
        return downloaded_files
    
    def download_fda_guidance_documents(self, urls: List[str], parallel: bool = False) -> List[str]:
        """
        Download FDA guidance documents.
        
        Args:
            urls: List of guidance document URLs
            parallel: Whether to download files in parallel
            
        Returns:
            List of paths to downloaded files
        """
        return self.download_multiple_pdfs(urls, subfolder="fda_guidance", parallel=parallel)
    
    def download_iso_standard_summaries(self, urls: List[str], parallel: bool = False) -> List[str]:
        """
        Download ISO standard summaries.
        
        Args:
            urls: List of ISO standard summary URLs
            parallel: Whether to download files in parallel
            
        Returns:
            List of paths to downloaded files
        """
        return self.download_multiple_pdfs(urls, subfolder="iso_standards", parallel=parallel)
    
    def download_cfr_documents(self, urls: List[str], parallel: bool = False) -> List[str]:
        """
        Download Code of Federal Regulations documents.
        
        Args:
            urls: List of CFR document URLs
            parallel: Whether to download files in parallel
            
        Returns:
            List of paths to downloaded files
        """
        return self.download_multiple_pdfs(urls, subfolder="cfr", parallel=parallel)
    
    def save_metadata(self, metadata: List[Dict[str, Any]], filename: str) -> str:
        """
        Save metadata to a JSON file.
        
        Args:
            metadata: List of metadata dictionaries
            filename: Filename for the JSON file
            
        Returns:
            Path to the saved file
        """
        if not filename.endswith('.json'):
            filename += '.json'
            
        output_path = os.path.join(self.output_dir, "metadata", filename)
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved metadata to: {output_path}")
        return output_path
    
    def get_fda_guidance_urls(self) -> List[str]:
        """
        Get URLs for FDA guidance documents about medical devices.
        
        Returns:
            List of URLs for guidance document PDFs
        """
        # Key FDA guidance pages
        guidance_pages = [
            "https://www.fda.gov/medical-devices/device-advice-comprehensive-regulatory-assistance/guidance-documents-medical-devices-and-radiation-emitting-products",
            "https://www.fda.gov/medical-devices/pre-submissions-and-meetings/pre-submission-program-feedback-medical-devices",
            "https://www.fda.gov/medical-devices/premarket-notification-510k/510k-forms",
            "https://www.fda.gov/medical-devices/device-advice-comprehensive-regulatory-assistance/how-study-and-market-your-device",
            "https://www.fda.gov/medical-devices/software-medical-device-samd/digital-health-policies-and-public-health-solutions",
            "https://www.fda.gov/medical-devices/device-software-functions-including-mobile-medical-applications/policy-device-software-functions-and-mobile-medical-applications"
        ]
        
        pdf_urls = []
        
        for page_url in tqdm(guidance_pages, desc="Searching FDA pages for guidance documents"):
            try:
                logger.info(f"Searching for PDFs on: {page_url}")
                response = self.session.get(page_url, timeout=30)
                response.raise_for_status()
                
                # Find PDF links
                content = response.text
                # Regex pattern to find PDF links
                pdf_links = re.findall(r'href=[\'"]([^\'"]+\.pdf)[\'"]', content)
                
                # Process each PDF link
                for link in pdf_links:
                    # Convert relative URLs to absolute
                    if not link.startswith(('http://', 'https://')):
                        if link.startswith('/'):
                            link = f"https://www.fda.gov{link}"
                        else:
                            link = f"{os.path.dirname(page_url)}/{link}"
                    
                    pdf_urls.append(link)
                
                # Respect FDA's servers
                time.sleep(self.delay)
                
            except requests.exceptions.RequestException as e:
                logger.error(f"Error accessing {page_url}: {e}")
                self.failed_downloads.append((page_url, str(e)))
        
        # Remove duplicates
        pdf_urls = list(set(pdf_urls))
        
        logger.info(f"Found {len(pdf_urls)} unique PDF guidance documents")
        return pdf_urls
    
    def get_primary_fda_guidance_urls(self) -> List[str]:
        """
        Get URLs for primary FDA guidance documents for medical devices.
        
        Returns:
            List of URLs for key guidance document PDFs
        """
        # List of important FDA guidance documents
        # These are some of the most relevant ones for medical device regulations
        primary_urls = [
            "https://www.fda.gov/media/82395/download",  # The 510(k) Program: Evaluating Substantial Equivalence
            "https://www.fda.gov/media/83888/download",  # Refuse to Accept Policy for 510(k)s
            "https://www.fda.gov/media/89179/download",  # Benefit-Risk Factors to Consider When Determining Substantial Equivalence
            "https://www.fda.gov/media/71975/download",  # Guidance on Medical Device Patient Labeling
            "https://www.fda.gov/media/71018/download",  # Design Considerations for Pivotal Clinical Investigations
            "https://www.fda.gov/media/72768/download",  # Factors to Consider When Making Benefit-Risk Determinations in Medical Device Premarket Approval
            "https://www.fda.gov/media/97071/download",  # Deciding When to Submit a 510(k) for a Change to an Existing Device
            "https://www.fda.gov/media/81431/download",  # De Novo Classification Process (Evaluation of Automatic Class III Designation)
            "https://www.fda.gov/media/70702/download",  # Software as a Medical Device (SAMD): Clinical Evaluation
            "https://www.fda.gov/media/80958/download",  # The 510(k) Program: Evaluating Substantial Equivalence in Premarket Notifications
            "https://www.fda.gov/media/144458/download",  # Safer Technologies Program for Medical Devices
            "https://www.fda.gov/media/163660/download",  # Cybersecurity in Medical Devices
            "https://www.fda.gov/media/108834/download",  # Technical Considerations for Additive Manufactured Medical Devices
            "https://www.fda.gov/media/150003/download",  # Computer Software Assurance for Production and Quality System Software
            "https://www.fda.gov/media/145050/download"   # Clinical Decision Support Software
        ]
        
        return primary_urls
    
    def get_cfr_urls(self) -> List[str]:
        """
        Get URLs for relevant Code of Federal Regulations (CFR) documents for medical devices.
        
        Returns:
            List of URLs for CFR PDFs
        """
        # Medical device related CFR titles
        cfr_urls = [
            # 21 CFR Part 800-1299 (Medical Devices)
            "https://www.govinfo.gov/content/pkg/CFR-2020-title21-vol8/pdf/CFR-2020-title21-vol8.pdf",  # Parts 800-1299 (Medical Devices)
            "https://www.govinfo.gov/content/pkg/CFR-2020-title21-vol1/pdf/CFR-2020-title21-vol1.pdf",  # Parts 1-99 (FDA General)
            "https://www.govinfo.gov/content/pkg/CFR-2020-title21-vol5/pdf/CFR-2020-title21-vol5.pdf",  # Parts 300-499 (Drugs)
            
            # Specific important parts for medical devices
            "https://www.accessdata.fda.gov/scripts/cdrh/cfdocs/cfcfr/CFRSearch.cfm?CFRPart=807",  # Part 807 (Registration and Listing)
            "https://www.accessdata.fda.gov/scripts/cdrh/cfdocs/cfcfr/CFRSearch.cfm?CFRPart=820",  # Part 820 (Quality System Regulation)
            "https://www.accessdata.fda.gov/scripts/cdrh/cfdocs/cfcfr/CFRSearch.cfm?CFRPart=801",  # Part 801 (Labeling)
            "https://www.accessdata.fda.gov/scripts/cdrh/cfdocs/cfcfr/CFRSearch.cfm?CFRPart=803",  # Part 803 (Medical Device Reporting)
            "https://www.accessdata.fda.gov/scripts/cdrh/cfdocs/cfcfr/CFRSearch.cfm?CFRPart=812",  # Part 812 (Investigational Device Exemptions)
            "https://www.accessdata.fda.gov/scripts/cdrh/cfdocs/cfcfr/CFRSearch.cfm?CFRPart=814",  # Part 814 (Premarket Approval)
            "https://www.accessdata.fda.gov/scripts/cdrh/cfdocs/cfcfr/CFRSearch.cfm?CFRPart=860"   # Part 860 (Medical Device Classification Procedures)
        ]
        
        return cfr_urls
    
    def get_iso_standards_urls(self) -> List[Dict[str, Any]]:
        """
        Get information about relevant ISO standards for medical devices.
        
        Note: Many ISO standards need to be purchased. This method returns
        information about the standards, not direct download links.
        
        Returns:
            List of dictionaries containing standard info
        """
        # Key ISO standards for medical devices
        iso_standards = [
            {
                "standard": "ISO 13485:2016",
                "title": "Medical devices — Quality management systems — Requirements for regulatory purposes",
                "url": "https://www.iso.org/standard/59752.html",
                "description": "Specifies requirements for a quality management system where an organization needs to demonstrate its ability to provide medical devices and related services"
            },
            {
                "standard": "ISO 14971:2019",
                "title": "Medical devices — Application of risk management to medical devices",
                "url": "https://www.iso.org/standard/72704.html",
                "description": "Specifies terminology, principles and a process for risk management of medical devices"
            },
            {
                "standard": "IEC 62304:2006",
                "title": "Medical device software — Software life cycle processes",
                "url": "https://www.iso.org/standard/38421.html",
                "description": "Defines the life cycle requirements for medical device software"
            },
            {
                "standard": "ISO 10993-1:2018", 
                "title": "Biological evaluation of medical devices — Part 1: Evaluation and testing within a risk management process",
                "url": "https://www.iso.org/standard/68936.html",
                "description": "Describes the general principles governing the biological evaluation of medical devices within a risk management process"
            },
            {
                "standard": "IEC 60601-1:2020",
                "title": "Medical electrical equipment — Part 1: General requirements for basic safety and essential performance",
                "url": "https://www.iso.org/standard/76026.html",
                "description": "Specifies general requirements for electrical medical equipment safety"
            }
        ]
        
        # Save this as metadata
        self.save_metadata(iso_standards, "iso_standards_metadata.json")
        
        # Return just the URLs
        return [item["url"] for item in iso_standards]
    
    def extract_document_metadata(self, file_path: str) -> Dict[str, Any]:
        """
        Extract metadata from a downloaded document.
        
        Args:
            file_path: Path to the document
            
        Returns:
            Dictionary containing document metadata
        """
        file_name = os.path.basename(file_path)
        file_size = os.path.getsize(file_path)
        
        metadata = {
            "file_name": file_name,
            "file_path": file_path,
            "file_size_bytes": file_size,
            "file_size_mb": round(file_size / (1024 * 1024), 2),
            "download_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "file_type": os.path.splitext(file_name)[1].lower()[1:],
            "category": self._determine_category(file_path)
        }
        
        return metadata
    
    def _determine_category(self, file_path: str) -> str:
        """
        Determine the category of a document based on its path.
        
        Args:
            file_path: Path to the document
            
        Returns:
            Category name (fda_guidance, iso_standards, cfr, or unknown)
        """
        if "fda_guidance" in file_path:
            return "fda_guidance"
        elif "iso_standards" in file_path:
            return "iso_standards"
        elif "cfr" in file_path:
            return "cfr"
        else:
            return "unknown"
        
    def generate_download_report(self) -> str:
        """
        Generate a report of downloaded files.
        
        Returns:
            Path to the report file
        """
        downloaded_metadata = []
        
        for file_path in self.downloaded_files:
            if os.path.exists(file_path):
                metadata = self.extract_document_metadata(file_path)
                downloaded_metadata.append(metadata)
        
        categories = {}
        for metadata in downloaded_metadata:
            category = metadata["category"]
            if category not in categories:
                categories[category] = []
            categories[category].append(metadata)
        
        summary = {
            "download_summary": {
                "total_files": len(downloaded_metadata),
                "total_size_mb": sum(item["file_size_mb"] for item in downloaded_metadata),
                "by_category": {cat: len(items) for cat, items in categories.items()},
                "failed_downloads": len(self.failed_downloads),
                "failed_urls": [{"url": url, "error": error} for url, error in self.failed_downloads]
            },
            "downloaded_files": downloaded_metadata
        }
        
        report_path = self.save_metadata(summary, f"download_report_{time.strftime('%Y%m%d_%H%M%S')}.json")
        
        return report_path
        
    def download_all_resources(self, parallel: bool = True) -> Dict[str, Any]:
        """Download all available FDA, CFR, and ISO resources.

        Args:
            parallel: Whether to download the files in parallel

        Returns:
            Summary dictionary of downloaded resources
        """
    
        logger.info("Gathering URLs for all resources...")
        
        fda_primary_urls = self.get_primary_fda_guidance_urls()
        fda_additional_urls = self.get_fda_guidance_urls()
        cfr_urls = self.get_cfr_urls()
        iso_info = self.get_iso_standards_urls()
        
        fda_additional_urls = list(set(fda_additional_urls) - set(fda_primary_urls))
        
        logger.info("Downloading FDA primary guidance documents...")
        fda_primary_files = self.download_fda_guidance_documents(fda_primary_urls, parallel=parallel)
        
        logger.info("Downloading additional FDA guidance documents...")
        fda_additional_files = self.download_fda_guidance_documents(fda_additional_urls, parallel=parallel)
        
        logger.info("Downloading CFR documents...")
        cfr_files = self.download_cfr_documents(cfr_urls, parallel=parallel)
        
        logger.info("Downloading ISO standard summaries...")
        iso_files = self.download_iso_standard_summaries(iso_info, parallel=parallel)

        report_path = self.generate_download_report()
        
        return {
            "fda_primary_count": len(fda_primary_files),
            "fda_additional_count": len(fda_additional_files),
            "cfr_count": len(cfr_files),
            "iso_count": len(iso_files),
            "total_downloaded": len(fda_primary_files) + len(fda_additional_files) + len(cfr_files) + len(iso_files),
            "failed_downloads": len(self.failed_downloads),
            "report_path": report_path
        }
        
if __name__ == "__main__":
    downloader = DataDownloader(delay=1.5)  # Slightly longer delay to be respectful of servers
    
    # Full download of all resources
    # Uncomment to run a full download (takes time)
    # summary = downloader.download_all_resources(parallel=True)
    # print(f"Download summary: {summary}")
    
    # Example: Get URLs for primary FDA guidance documents
    primary_urls = downloader.get_primary_fda_guidance_urls()
    print(f"Found {len(primary_urls)} primary guidance document URLs")
    
    # Example: Download a few primary guidance documents
    if primary_urls:
        downloaded_files = downloader.download_fda_guidance_documents(primary_urls[:3])
        print(f"Downloaded {len(downloaded_files)} files")
        
        # Generate report of downloaded files
        report_path = downloader.generate_download_report()
        print(f"Download report saved to: {report_path}")
