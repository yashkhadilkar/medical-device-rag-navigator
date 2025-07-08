#!/usr/bin/env python
"""
CloudVectorStore using HuggingFace Transformers instead of SentenceTransformers.
This version is more reliable and less prone to hanging.
"""

import os
import json
import logging
import requests
import numpy as np
import pdfplumber
from typing import Dict, List, Any, Optional

from transformers import AutoTokenizer, AutoModel
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CloudVectorStore:
    """Vector store that loads PDFs directly from AWS S3 with HuggingFace Transformers."""
    
    def __init__(self, config):
        self.config = config
        self.s3_bucket = "medical-device-regulations-2025"
        self.s3_base_url = f"https://{self.s3_bucket}.s3.amazonaws.com"

        self.cache_dir = ".cache/s3_pdfs"
        os.makedirs(self.cache_dir, exist_ok=True)
        
        self.metadata_cache_path = os.path.join(self.cache_dir, "metadata.json")
        self.vectors_cache_path = os.path.join(self.cache_dir, "vectors.npy")

        self.doc_metadata = {}
        self.vectors = {}
        self.id_to_index = {}
        self.doc_count = 0

        self.tokenizer = None
        self.embedding_model = None

        self.pdf_files = []

        self._initialize_s3_dataset()
    
    def _get_embedding_model(self):
        """Initialize HuggingFace embedding model if not already loaded."""
        if self.embedding_model is None:
            model_name = 'sentence-transformers/all-MiniLM-L6-v2'
            logger.info(f"Loading HuggingFace embedding model: {model_name}")
            
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.embedding_model = AutoModel.from_pretrained(model_name)
                logger.info("✅ HuggingFace embedding model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load HuggingFace model: {e}")
                raise
        
        return self.embedding_model, self.tokenizer
    
    def _encode_text(self, text):
        """Encode text using HuggingFace Transformers."""
        model, tokenizer = self._get_embedding_model()

        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
        
        with torch.no_grad():
            outputs = model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)
        
        return embeddings.numpy().flatten()
    
    def _initialize_s3_dataset(self):
        """Initialize dataset by loading PDFs from S3."""
        logger.info("Initializing dataset with S3 PDF loading...")

        if self._load_local_cache():
            logger.info(f"Loaded {self.doc_count} documents from local cache")
            return

        self.pdf_files = self._discover_pdf_files()
        
        if not self.pdf_files:
            logger.error("No PDF files discovered! Check bucket access.")
            return
        
        logger.info(f"Discovered {len(self.pdf_files)} PDFs - processing...")

        self._download_and_process_pdfs()

        self._save_local_cache()
        
        logger.info(f"Initialized S3 vector store with {self.doc_count} documents")
    
    def _download_and_process_pdfs(self):
        """Download and process all PDFs from S3."""
        logger.info(f"Processing {len(self.pdf_files)} PDFs from S3...")
        
        processed_count = 0
        
        for i, pdf_path in enumerate(self.pdf_files):
            try:
                logger.info(f"Processing {i+1}/{len(self.pdf_files)}: {pdf_path}")
                
                # Download PDF
                local_pdf_path = self._download_pdf_fixed(pdf_path)
                if not local_pdf_path:
                    continue
                
                # Extract text
                text = self._extract_text_from_pdf(local_pdf_path)
                if len(text) < 100:
                    logger.warning(f"Very short text from {pdf_path}: {len(text)} chars")
                    continue
                
                # Extract metadata
                category = pdf_path.split('/')[0]
                filename = os.path.basename(pdf_path)
                title = filename[:-4].replace('_', ' ').title()

                doc_id = f"s3_pdf_{i}"
                doc_data = {
                    'id': doc_id,
                    'title': title,
                    'category': category,
                    'file_path': pdf_path,
                    'text': text,
                    'text_length': len(text),
                    's3_url': f"{self.s3_base_url}/{pdf_path}",
                    'extraction_method': 's3_huggingface_transformers'
                }

                embedding = self._encode_text(text)

                self.doc_metadata[doc_id] = doc_data
                self.vectors[doc_id] = embedding
                self.id_to_index[doc_id] = processed_count
                
                processed_count += 1
                logger.info(f"✅ Processed {title}: {len(text):,} characters")

                if os.path.exists(local_pdf_path):
                    os.remove(local_pdf_path)
                
            except Exception as e:
                logger.error(f"Error processing {pdf_path}: {e}")
                continue
        
        self.doc_count = processed_count
        logger.info(f"🎉 Successfully processed {processed_count} PDFs from S3")

        categories = {}
        for doc in self.doc_metadata.values():
            cat = doc['category']
            categories[cat] = categories.get(cat, 0) + 1
        logger.info(f"📂 Categories: {categories}")
    
    def _download_pdf_fixed(self, pdf_path: str) -> Optional[str]:
        """Fixed PDF download method that properly handles response streams."""
        s3_url = f"{self.s3_base_url}/{pdf_path}"
        local_path = os.path.join(self.cache_dir, pdf_path.replace('/', '_'))
        
        try:
            session = requests.Session()

            headers = {
                'User-Agent': 'Medical-Device-RAG/1.0',
                'Accept': 'application/pdf,*/*'
            }

            response = session.get(s3_url, headers=headers, stream=True, timeout=60)
            response.raise_for_status()

            total_size = 0
            with open(local_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:  
                        f.write(chunk)
                        total_size += len(chunk)

            response.close()
            session.close()

            if total_size == 0:
                logger.error(f"Downloaded 0 bytes for {pdf_path}")
                if os.path.exists(local_path):
                    os.remove(local_path)
                return None
            
            logger.debug(f"Downloaded {pdf_path} ({total_size:,} bytes)")
            return local_path
            
        except Exception as e:
            logger.error(f"Failed to download {pdf_path}: {e}")
            return None
    
    def _extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from a PDF file."""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                text = ''
                for page_num, page in enumerate(pdf.pages):
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + '\n\n'
                
                return text.strip()
                
        except Exception as e:
            logger.error(f"Error extracting text from {pdf_path}: {e}")
            return ''
    
    def _load_local_cache(self) -> bool:
        """Load documents and vectors from local cache."""
        try:
            if not (os.path.exists(self.metadata_cache_path) and 
                   os.path.exists(self.vectors_cache_path)):
                return False

            with open(self.metadata_cache_path, 'r') as f:
                cache_data = json.load(f)
                self.doc_metadata = cache_data['metadata']
                self.id_to_index = cache_data['id_to_index']
                self.doc_count = cache_data['doc_count']

            vectors_array = np.load(self.vectors_cache_path)
            self.vectors = {}
            for doc_id, index in self.id_to_index.items():
                if index < len(vectors_array):
                    self.vectors[doc_id] = vectors_array[index]
            
            return True
            
        except Exception as e:
            logger.warning(f"Could not load cache: {e}")
            return False
    
    def _save_local_cache(self):
        """Save documents and vectors to local cache."""
        try:
            cache_data = {
                'metadata': self.doc_metadata,
                'id_to_index': self.id_to_index,
                'doc_count': self.doc_count
            }
            with open(self.metadata_cache_path, 'w') as f:
                json.dump(cache_data, f, indent=2)

            if self.vectors:
                vectors_list = []
                for doc_id in sorted(self.id_to_index.keys(), 
                                   key=lambda x: self.id_to_index[x]):
                    if doc_id in self.vectors:
                        vectors_list.append(self.vectors[doc_id])
                
                if vectors_list:
                    vectors_array = np.array(vectors_list)
                    np.save(self.vectors_cache_path, vectors_array)
            
            logger.info("Saved documents and vectors to local cache")
            
        except Exception as e:
            logger.error(f"Error saving cache: {e}")
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search method using HuggingFace Transformers embeddings."""
        if not self.vectors:
            logger.warning("No vectors available for search")
            return []
        
        try:
            if isinstance(query, dict):
                actual_query = query.get('processed_query', query.get('original_query', str(query)))
                logger.debug(f"Extracted query from dict: '{actual_query}'")
            else:
                actual_query = str(query).strip()
            
            if not actual_query:
                logger.warning("Empty query after processing")
                return []

            query_embedding = self._encode_text(actual_query)
            
            logger.debug(f"Query embedding shape: {query_embedding.shape}")
            
            # Calculate similarities
            similarities = []
            for doc_id, doc_vector in self.vectors.items():
                try:
                    if not isinstance(doc_vector, np.ndarray):
                        doc_vector = np.array(doc_vector)
                    
                    doc_vector = doc_vector.flatten()
                    
                    # Check dimension compatibility
                    if query_embedding.shape != doc_vector.shape:
                        logger.warning(f"Dimension mismatch for {doc_id}: query {query_embedding.shape} vs doc {doc_vector.shape}")
                        continue
                    
                    # Calculate cosine similarity
                    query_norm = np.linalg.norm(query_embedding)
                    doc_norm = np.linalg.norm(doc_vector)
                    
                    if query_norm == 0 or doc_norm == 0:
                        similarity = 0.0
                    else:
                        dot_product = np.dot(query_embedding, doc_vector)
                        similarity = dot_product / (query_norm * doc_norm)

                    if isinstance(similarity, np.ndarray):
                        similarity = similarity.item()
                    
                    similarities.append((doc_id, float(similarity)))
                    
                except Exception as e:
                    logger.error(f"Error calculating similarity for {doc_id}: {e}")
                    continue
            
            if not similarities:
                logger.warning("No valid similarities calculated")
                return []
            
            # Sort by similarity and return top results
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            results = []
            for doc_id, similarity in similarities[:top_k]:
                if doc_id in self.doc_metadata:
                    doc_data = self.doc_metadata[doc_id].copy()
                    doc_data['similarity'] = similarity
                    doc_data['score'] = similarity  
                    results.append(doc_data)
            
            logger.info(f"Search completed: {len(results)} results for '{actual_query}'")
            return results
            
        except Exception as e:
            logger.error(f"Error during search: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def get_cache_size_mb(self) -> float:
        """Get the size of local cache in MB."""
        total_size = 0
        
        if os.path.exists(self.metadata_cache_path):
            total_size += os.path.getsize(self.metadata_cache_path)
        
        if os.path.exists(self.vectors_cache_path):
            total_size += os.path.getsize(self.vectors_cache_path)
        
        return total_size / (1024 * 1024)
    
    def get(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get a document by ID."""
        return self.doc_metadata.get(doc_id)
    
    def clear_cache(self) -> bool:
        """Clear the local cache."""
        try:
            if os.path.exists(self.metadata_cache_path):
                os.remove(self.metadata_cache_path)
            
            if os.path.exists(self.vectors_cache_path):
                os.remove(self.vectors_cache_path)
            
            # Clear temporary PDF files
            for pdf_file in self.pdf_files:
                temp_path = os.path.join(self.cache_dir, pdf_file.replace('/', '_'))
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            
            self.doc_metadata = {}
            self.vectors = {}
            self.id_to_index = {}
            self.doc_count = 0
            
            logger.info("Cleared local cache")
            return True
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            return False
        
    def _discover_pdf_files(self):
        """
        Main PDF discovery method that works with 403 bucket listing restrictions.
        """
        logger.info("Starting file-based PDF discovery...")
        logger.info("(This method works even when bucket listing is disabled)")

        pdfs = self._discover_pdfs_by_file_testing()
        
        if not pdfs:
            logger.error("No PDF files discovered! Check file names and bucket access.")
            return []
        
        logger.info(f"Successfully discovered {len(pdfs)} accessible PDFs")
        
        # Log summary by category
        categories = {}
        for pdf_path in pdfs:
            cat = pdf_path.split('/')[0] if '/' in pdf_path else 'root'
            categories[cat] = categories.get(cat, 0) + 1
        
        for cat, count in categories.items():
            logger.info(f"    {cat}: {count} PDFs")
        
        return pdfs

    def _discover_pdfs_by_file_testing(self):
        """
        Discover PDFs by testing known file patterns directly.
        Works even when bucket listing is disabled (403).
        """
        import urllib.parse
        
        accessible_files = []
        
        # Known file patterns based on S3 bucket listing
        known_files = {
            'classification': [
                'Classification-of-Products-as-Drugs-and-Devices-and-Additional-Product-Classification-Issues---Guidance-for-Industry-and-FDA-Staff-.pdf',
                'Medical_Device_Classification_Product_Codes.pdf',
                'Novel_device_classification_guidance.pdf',
                'classification_procedures.pdf',
                'classification_products_drugs_devices.pdf',
                'feedback_meetings_submissions.pdf'
            ],
            'compliance': [
                '21_CFR_Part_820_(Quality System Regulation).pdf',
                'Corrective_and_Preventive_Actions_(CAPA)_guidance.pdf',
                'Establishment Registration and Device Listing guidance.pdf',
                'medical_device_reporting.pdf',
                'postmarket_surveillance.pdf',
                'udi.pdf'
            ],
            'software': [
                'AI:ML_Machine_learning_Software_as_Medical_Device_guidance.pdf',
                'IEC_62304.pdf',
                'Using_Artificial_Intelligence_&_Machine_Learning_in_the_Development_of_Drug_and_Biological_Products.pdf',
                'clinical_decision_support.pdf',
                'cybersecurity.pdf',
                'software_functions_mobile_apps.pdf',
                'software_submissions.pdf'
            ],
            'submission': [
                'Humanitarian Device Exemption (HDE) guidance.pdf',
                'IDE (Investigational Device Exemption) guidance.pdf',
                'Q-Sub (Q-Submission) program guidance.pdf',
                'de_novo_process.pdf',
                'pma_content.pdf',
                'traditional_510k_format.pdf',
                'when_to_submit_510k.pdf'
            ],
            'testing': [
                'Applying-Human-Factors-and-Usability-Engineering-to-Medical-Devices---Guidance-for-Industry-and-Food-and-Drug-Administration-Staff.pdf',
                'FDA-2015-D-3787-0016_attachment_1.pdf',
                'ISO 14971 (Risk Management).pdf',
                'Shelf-Life-of-Medical-Devices.pdf',
                'benefit_risk_factors.pdf',
                'consensus_standards.pdf',
                'home_use_devices.pdf',
                'iso_10993.pdf',
                'software_validation.pdf'
            ]
        }
        
        total_files_to_test = sum(len(files) for files in known_files.values())
        logger.info(f"Testing {total_files_to_test} known files...")
        
        # Test each file individually
        for category, filenames in known_files.items():
            logger.info(f"📂 Testing {category} ({len(filenames)} files)...")
            
            category_found = 0
            for filename in filenames:
                file_path = f"{category}/{filename}"
                
                if self._test_file_accessibility(file_path):
                    accessible_files.append(file_path)
                    category_found += 1
                    logger.info(f"  ✅ {filename}")
                else:
                    logger.debug(f"  ❌ {filename}")
            
            logger.info(f"  📊 {category}: {category_found}/{len(filenames)} files accessible")
        
        logger.info(f"🎯 File testing complete: {len(accessible_files)}/{total_files_to_test} files accessible")
        
        return sorted(accessible_files)

    def _test_file_accessibility(self, file_path):
        """
        Test if a specific file is accessible via HEAD request.
        
        Args:
            file_path: File path to test (e.g., "classification/file.pdf")
            
        Returns:
            True if file is accessible, False otherwise
        """
        try:
            import urllib.parse

            encoded_path = urllib.parse.quote(file_path, safe='/')
            test_url = f"{self.s3_base_url}/{encoded_path}"

            response = requests.head(test_url, timeout=10)
            
            if response.status_code == 200:
                return True
            else:
                logger.debug(f"    File not accessible: {file_path} (HTTP {response.status_code})")
                return False
                
        except Exception as e:
            logger.debug(f"    Error testing {file_path}: {e}")
            return False