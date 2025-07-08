import requests
from bs4 import BeautifulSoup
import os 
import time
from typing import List, Dict, Any, Optional 
from urllib.parse import urljoin, urlparse
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FDAWebScraper:
    """Scape the FDA Website content related to medical devices."""
    def __init__(self, base_url: str = "https://www.fda.gov", delay: float = 1.0):
        """Initialize the FDA web scraper. 

        Args:
            base_url (_type_, optional): Base URL for FDA website.
            delay (float, optional): Delay between requests
        """
        
        self.base_url = base_url
        self.delay = delay
        self.session = requests.Session()
        
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
    def _get_page(self, url: str) -> Optional[BeautifulSoup]:
        """Fetch and parse a web page.

        Args:
            url: URL to fetch
            
        Returns:
            BeautifulSoup object or None if request failed 
        """
        try: 
            if not url.startswith(('http://', 'https://')):
                url = urljoin(self.base_url, url)
                
            logger.info(f"Fetching page: {url}")
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            time.sleep(self.delay)
            
            return BeautifulSoup(response.content, 'html.parser')
        except requests.exceptiions.RequestException as e:
            logger.error(f"Error fetching page {url}")
            return None
            
    def scrape_guidance_page(self, url: str) -> Dict[str, Any]:
        """Scrape a single FDA guidance document page.

        Args:
            url: URL of the guidance page

        Returns:
            Dictionary containing the guidance document details. 
        """
        
        soup = self._get_page(url)
        if not soup:
            return {}
        
        result = {
            'url': url,
            'title': '',
            'content': '',
            'publication_date': '',
            'document_number': '',
        }
        
        title_elem = soup.find('h1')
        if title_elem:
            result['title'] = title_elem.get_text().strip()
            
        content_elem = soup.find('div', {'class': 'content'}) or soup.find('div', {'id': 'content'})
        if content_elem:
            for script in content_elem(["script", "style"]):
                script.decompose()
            result['content'] = content_elem.get_text(separator=' ', strip=True)
            
        date_elem = soup.find('time') or soup.find('div', {'class': 'date'})
        if date_elem:
            result['publication_date'] = date_elem.get_text().strip()
            
        doc_elem = soup.find(string=lambda text: text and 'Document Number' in text)
        if doc_elem and doc_elem.parent:
            result['document_number'] = doc_elem.parent.get_text().replace('Document Number:', '').strip()
        
        # Look for pdf links.    
        pdf_links = []
        for a in soup.find_all('a', href=True):
            if a['href'].endswith('.pdf'):
                pdf_url = urljoin(url, a['href'])
                pdf_links.append({
                    'url': pdf_url,
                    'text': a.get_text().strip() or os.path.basename(pdf_url)
                })
                
        result['pdf_links'] = pdf_links
        
        return result
    
    def scrape_guidance_listing(self, url: str) -> List[Dict[str, Any]]:
        """Scrape an FDA guidance listing page to find guidance documents.

        Args:
            url: URL of the guidance listing page

        Returns:
            List of dictionaries with guidance document URLs and titles
        """
        
        soup = self._get_page(url)
        if not soup:
            return []
        
        guidance_links = []
        
        for a in soup.find_all('a', href=True):
            href = a['href']
            link_text = a.get_text().strip()
            
            if (('guidance' in href.lower() or 'guidance' in link_text.lower()) and
                not href.endswith(('.pdf', '.doc', '.docx'))):
                
                full_url = urljoin(url, href)
                guidance_links.append({
                    'url': full_url,
                    'title': link_text
                })
                
        return guidance_links
    
    def find_pdf_links(self, url: str) -> List[Dict[str, str]]:
        """Find PDF links on a page.

        Args:
            url: URL to search for PDF links

        Returns:
            List of dictionaries with PDF URLs and link text
        """
        soup = self._get_page(url)
        if not soup:
            return []
        
        pdf_links = []
        
        for a in soup.find_all('a', href=True):
            if a['href'].lower().endswith('.pdf'):
                pdf_url = urljoin(url, a['href'])
                link_text = a.get_text().strip() or os.path.basename(pdf_url)
                
                pdf_links.append({
                    'url': pdf_url,
                    'text': link_text
                })
                
            return pdf_links    
        
    def scrape_device_classification(self, url: str) -> List[Dict[str, Any]]:
        """Scrape device classification information from FDA page.

        Args:
            url: URL of the classification page

        Returns:
            List of dictionaries with device classification information
        """
        
        soup = self._get_page(url)
        if not soup:
            return[]
        
        classifications = []
        
        tables = soup.find_all('table')
        for table in tables:
            headers = []
            rows = table.find_all('tr')
            
            header_row = rows[0] if rows else None
            if header_row:
                headers = [th.get_text().strip() for th in header_row.find_all(['th', 'td'])]
                
            for row in rows[1:]:
                cells = row.find_all(['td', 'th'])
                if len(cells) == len(headers) and len(headers) > 0:
                    classification = {}
                    for i, cell in enumerate(cells):
                        if i < len(headers):
                            classification[headers[i]] = cell.get_text().strip()
                    classifications.append(classification)
                    
        return classifications
    
    
if __name__ == "__main__":
    scraper = FDAWebScraper()
    
    guidance_page = "https://www.fda.gov/medical-devices/guidance-documents-medical-devices-and-radiation-emitting-products/draft-medical-device-guidance"
    guidance_links = scraper.scrape_guidance_listing(guidance_page)
    print(f"Found {len(guidance_links)} guidance links")
    
    pdf_links = scraper.find_pdf_links(guidance_page)
    print(f"Found {len(pdf_links)} PDF links")
    
    for i, link in enumerate(pdf_links[:3]):
        print(f"{i+1}. {link['text']}: {link['url']}")
    