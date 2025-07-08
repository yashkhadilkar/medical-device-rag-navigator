import os 
import logging
import json 
from typing import List, Dict, Any, Optional, Set, Tuple
from pathlib import Path
import hashlib
import re
import pandas as pd
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ResourceManager:
    """Manage regulatory document resources with prioritization and lifecyle management. 
    Focuses on storage efficiency and resource optimization
    """
    
    def __init__(self, base_dir: str = "data", 
                max_storage_mb: int = 200,
                cache_ttl_days: int = 30,
                priority_documents: Optional[List[str]] = None):
        """
        Initialize the resource manager.
        
        Args:
            base_dir: Base directory for data storage
            max_storage_mb: Maximum storage to use in MB
            cache_ttl_days: Time-to-live for cached data in days
            priority_documents: List of high-priority document names/IDs
        """
        self.base_dir = base_dir
        self.max_storage_bytes = max_storage_mb * 1024 * 1024
        self.cache_ttl_days = cache_ttl_days
        
        # Default priority documents (key FDA regulations)
        self.priority_documents = priority_documents or [
            # 510(k) related
            "510k_program",
            "510k_substantial_equivalence",
            "510k_when_to_submit",
            "510k_refuse_to_accept",
            
            # Classification and risk
            "device_classification",
            "class_determination",
            "benefit_risk_factors",
            
            # Software and digital health
            "software_medical_device",
            "digital_health_policies",
            "clinical_decision_support_software",
            
            # Quality and manufacturing
            "quality_system_regulation",
            "manufacturing_practices",
            
            # Safety and compliance
            "medical_device_reporting",
            "postmarket_surveillance"
        ]
        
        self.raw_dir = os.path.join(base_dir, "raw")
        self.processed_dir = os.path.join(base_dir, "processed")
        self.metadata_dir = os.path.join(base_dir, "metadata")
        self.cache_dir = os.path.join(base_dir, "cache")
        
        for directory in [self.raw_dir, self.processed_dir, self.metadata_dir, self.cache_dir]:
            os.makedirs(directory, exist_ok=True)
            
        self.metadata_index_path = os.path.join(self.metadata_dir, "document_index.json")
        self.document_index = self._load_document_index()
        
        self.update_storage_stats()
        
    def _load_document_index(self) -> Dict[str, Any]:
        """Load document index from disk.

        Returns:
            Document index dictionary
        """
        
        if os.path.exists(self.metadata_index_path):
            try:
                with open(self.metadata_index_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Error loading document index: {e}")
                
        return {
            "documents": {},
            "last_update": datetime.now().isoformat(),
            "total_count": 0,
            "total_size_bytes": 0
        }

    def _save_document_index(self) -> None:
        """
        Save document index to disk.
        """
        self.document_index["last_update"] = datetime.now().isoformat()
        
        try:
            with open(self.metadata_index_path, 'w', encoding='utf-8') as f:
                json.dump(self.document_index, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving document index: {e}")
            
    def update_storage_stats(self) -> Dict[str, Any]:
        """
        Update storage statistics.
        
        Returns:
            Dictionary with storage statistics
        """
        stats = {
            "raw_size_bytes": self._get_directory_size(self.raw_dir),
            "processed_size_bytes": self._get_directory_size(self.processed_dir),
            "metadata_size_bytes": self._get_directory_size(self.metadata_dir),
            "cache_size_bytes": self._get_directory_size(self.cache_dir),
            "timestamp": datetime.now().isoformat()
        }
        
        stats["total_size_bytes"] = (
            stats["raw_size_bytes"] + 
            stats["processed_size_bytes"] + 
            stats["metadata_size_bytes"] + 
            stats["cache_size_bytes"]
        )
        
        stats["total_size_mb"] = round(stats["total_size_bytes"] / (1024 * 1024), 2)
        stats["available_space_mb"] = round(
            (self.max_storage_bytes - stats["total_size_bytes"]) / (1024 * 1024), 2
        )
        
        stats_path = os.path.join(self.metadata_dir, "storage_stats.json")
        try:
            with open(stats_path, 'w', encoding='utf-8') as f:
                json.dump(stats, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving storage stats: {e}")
        
        return stats
    
    def _get_directory_size(self, directory: str) -> int:
        """Calculate total size of a directory in bytes.

        Args:
            directory: Directory path

        Returns:
            Size in bytes
        """
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(directory):
            for filename in filenames:
                file_path = os.path.join(dirpath, filename)
                total_size += os.path.getsize(file_path)
                
        return total_size
    
    def is_priority_document(self, document_name: str) -> bool:
        """
        Check if a document is in the priority list.
        Uses fuzzy matching to identify priority documents.
        
        Args:
            document_name: Document name or ID
            
        Returns:
            True if document is a priority
        """
        document_name_lower = document_name.lower()
        
        if document_name_lower in [p.lower() for p in self.priority_documents]:
            return True
        
        for priority in self.priority_documents:
            if priority.lower() in document_name_lower:
                return True
                
        return False
    
    def register_document(self, document_path: str, metadata: Dict[str, Any]) -> str:
        """Register a document in the system.

        Args:
            document_path: PAth to the document
            metadata: Document metadata

        Returns:
            Document ID
        """

        if "document_id" not in metadata:
            if "document_number" in metadata:
                doc_id = metadata["document_number"].replace(" ", "_").lower()
            elif "title" in metadata:
                title = metadata["title"][:50].lower()
                doc_id = re.sub(r'[^a-z0-9]', '_', title)
            else:
                doc_id = hashlib.md5(os.path.basename(document_path).encode()).hexdigest()[:12]
            
            metadata["document_id"] = doc_id
        else:
            doc_id = metadata["document_id"]
        
        document_info = {
            "id": doc_id,
            "path": document_path,
            "size_bytes": os.path.getsize(document_path),
            "registration_time": datetime.now().isoformat(),
            "metadata": metadata,
            "is_priority": self.is_priority_document(doc_id) or 
                           self.is_priority_document(metadata.get("title", ""))
        }
        
        self.document_index["documents"][doc_id] = document_info
        self.document_index["total_count"] = len(self.document_index["documents"])
        self.document_index["total_size_bytes"] = sum(
            doc["size_bytes"] for doc in self.document_index["documents"].values()
        )
        
        self._save_document_index()
        
        return doc_id
        
    def remove_document(self, document_id: str, delete_files: bool = True) -> bool:
        """
        Remove a document from the system.
        
        Args:
            document_id: Document ID
            delete_files: Whether to delete associated files
            
        Returns:
            True if successful
        """
        if document_id not in self.document_index["documents"]:
            logger.warning(f"Document not found: {document_id}")
            return False
        
        document_info = self.document_index["documents"][document_id]
        
        if delete_files:
            document_path = document_info["path"]
            if os.path.exists(document_path):
                try:
                    os.remove(document_path)
                    logger.info(f"Deleted document file: {document_path}")
                except Exception as e:
                    logger.error(f"Error deleting document: {e}")
                    return False
                
                
        del self.document_index["documents"][document_id]
        self.document_index["total_count"] = len(self.document_index["documents"])
        self.document_index["total_size_bytes"] = sum(
            doc["size_bytes"] for doc in self.document_index["documents"].values()
        )
        
        self._save_document_index()
        
        self.update_storage_stats()
        
        return True
    
    def get_document_info(self, document_id: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a document.
        
        Args:
            document_id: Document ID
            
        Returns:
            Document information
        """
        return self.document_index["documents"].get(document_id)
    
    def get_all_documents(self, priority_only: bool = False) -> List[Dict[str, Any]]:
        """
        Get all registered documents.
        
        Args:
            priority_only: Whether to get only priority documents
            
        Returns:
            List of document information
        """
        if priority_only:
            return [
                doc for doc in self.document_index["documents"].values()
                if doc.get("is_priority", False)
            ]
        else:
            return list(self.document_index["documents"].values())
        
    def cleanup_cache(self, force: bool = False) -> int:
        """
        Clean up cache files.
        
        Args:
            force: Whether to force cleanup regardless of TTL
            
        Returns:
            Number of files deleted
        """
        deleted_count = 0
        cutoff_date = datetime.now() - timedelta(days=self.cache_ttl_days)
        
        for root, dirs, files in os.walk(self.cache_dir):
            for file in files:
                file_path = os.path.join(root, file)

                file_mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
                
                if force or file_mtime < cutoff_date:
                    try:
                        os.remove(file_path)
                        deleted_count += 1
                    except Exception as e:
                        logger.error(f"Error deleting cache file {file_path}: {e}")
        
        logger.info(f"Deleted {deleted_count} cache files")
        
        self.update_storage_stats()
        
        return deleted_count
    
    def optimize_storage(self) -> Dict[str, Any]:
        """
        Optimize storage usage by removing non-priority items if needed.
        
        Returns:
            Results of optimization
        """
        results = {
            "before_size_mb": 0,
            "after_size_mb": 0,
            "removed_documents": 0,
            "cleared_cache_files": 0
        }
        
        stats = self.update_storage_stats()
        results["before_size_mb"] = stats["total_size_mb"]

        if stats["total_size_bytes"] <= self.max_storage_bytes:
            logger.info("Storage within limits, no optimization needed")
            results["after_size_mb"] = results["before_size_mb"]
            return results

        results["cleared_cache_files"] = self.cleanup_cache(force=True)

        if self.check_storage_limits():
            stats = self.update_storage_stats()
            results["after_size_mb"] = stats["total_size_mb"]
            return results

        non_priority_docs = [
            doc for doc in self.document_index["documents"].values()
            if not doc.get("is_priority", False)
        ]

        non_priority_docs.sort(key=lambda x: x["size_bytes"], reverse=True)
        
        for doc in non_priority_docs:
            if self.check_storage_limits():
                break
                
            if self.remove_document(doc["id"]):
                results["removed_documents"] += 1

        stats = self.update_storage_stats()
        results["after_size_mb"] = stats["total_size_mb"]
        
        return results
    
    def export_document_catalog(self, output_path: Optional[str] = None) -> str:
        """
        Export catalog of documents as CSV or JSON.
        
        Args:
            output_path: Output file path
            
        Returns:
            Path to exported file
        """
        if not output_path:
            output_path = os.path.join(self.metadata_dir, f"document_catalog_{datetime.now().strftime('%Y%m%d')}.csv")
        
        # Create catalog data
        catalog_data = []
        for doc_id, doc_info in self.document_index["documents"].items():
            metadata = doc_info.get("metadata", {})
            
            catalog_entry = {
                "document_id": doc_id,
                "title": metadata.get("title", ""),
                "document_number": metadata.get("document_number", ""),
                "document_date": metadata.get("document_date", ""),
                "document_type": metadata.get("document_type", ""),
                "is_priority": doc_info.get("is_priority", False),
                "size_bytes": doc_info.get("size_bytes", 0),
                "size_mb": round(doc_info.get("size_bytes", 0) / (1024 * 1024), 2),
                "registration_time": doc_info.get("registration_time", "")
            }
            
            catalog_data.append(catalog_entry)
        
        # Save as CSV or JSON
        if output_path.endswith('.csv'):
            df = pd.DataFrame(catalog_data)
            df.to_csv(output_path, index=False)
        else:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(catalog_data, f, indent=2)
        
        logger.info(f"Exported document catalog to {output_path}")
        return output_path
    
    def import_documents(self, directory: str, recursive: bool = True, 
                       registration_func: Optional[callable] = None) -> int:
        """
        Import documents from a directory.
        
        Args:
            directory: Directory containing documents
            recursive: Whether to search subdirectories
            registration_func: Function to extract metadata and register
            
        Returns:
            Number of documents imported
        """
        import_count = 0

        if registration_func is None:
            def default_registration(file_path):
                metadata = {
                    "title": os.path.basename(file_path),
                    "import_time": datetime.now().isoformat()
                }
                return self.register_document(file_path, metadata)
            
            registration_func = default_registration

        pattern = "**/*" if recursive else "*"
        for file_path in Path(directory).glob(pattern):
            if file_path.is_file():
                try:
                    doc_id = registration_func(str(file_path))
                    if doc_id:
                        import_count += 1
                except Exception as e:
                    logger.error(f"Error importing {file_path}: {e}")

        self.update_storage_stats()
        
        return import_count