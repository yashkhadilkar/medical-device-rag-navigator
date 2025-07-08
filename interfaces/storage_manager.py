import time 
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional
import datetime
import logging
import os
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StorageManager:
    """
    Manage local storage for the Medical Device Regulation Navigator.
    Provides functions for tracking, optimizing, and cleaning up storage.
    """
    
    def __init__(self, cache_dir: str = ".cache", max_storage_mb: int = 200):
        """
        Initialize the storage manager.
        
        Args:
            cache_dir: Base directory for cache storage
            max_storage_mb: Maximum storage size in MB
        """
        self.cache_dir = cache_dir
        self.max_storage_bytes = max_storage_mb * 1024 * 1024
        
        os.makedirs(cache_dir, exist_ok=True)

        self.subdirs = [
            "embeddings",
            "vector_store",
            "retriever",
            "llm",
            "queries",
            "pipeline",
            "terminology",
            "prompts"
        ]
        
        for subdir in self.subdirs:
            os.makedirs(os.path.join(cache_dir, subdir), exist_ok=True)
            
        self.stats_file = os.path.join(cache_dir, "storage_stats.json")

        self.update_storage_stats()
        logger.info(f"Initialized storage manager with {max_storage_mb}MB max storage")
        
    def get_directory_size(self, path: str) -> int:
        """
        Calculate the size of a directory in bytes.
        
        Args:
            path: Directory path
            
        Returns:
            Size in bytes
        """
        total_size = 0
        for entry in Path(path).rglob('*'):
            if entry.is_file() and not entry.is_symlink():
                total_size += entry.stat().st_size
                
        return total_size
    
    def update_storage_stats(self) -> Dict[str, Any]:
        """
        Update storage statistics.
        
        Returns:
            Dictionary of storage statistics
        """
        stats = {
            "timestamp": datetime.datetime.now().isoformat(),
            "max_storage_mb": self.max_storage_bytes / (1024 * 1024),
            "subdirectories": {}
        }
        
        total_size = 0
        
        for subdir in self.subdirs:
            subdir_path = os.path.join(self.cache_dir, subdir)
            if os.path.exists(subdir_path):
                size_bytes = self.get_directory_size(subdir_path)
                stats["subdirectories"][subdir] = {
                    "path": subdir_path,
                    "size_bytes": size_bytes,
                    "size_mb": size_bytes / (1024 * 1024)
                }
                total_size += size_bytes
            
        stats["total_size_bytes"] = total_size
        stats["total_size_mb"] = total_size / (1024 * 1024)
        stats["available_space_mb"] = (self.max_storage_bytes - total_size) / (1024 * 1024)
        stats["usage_percent"] = (total_size / self.max_storage_bytes) * 100 if self.max_storage_bytes > 0 else 0

        try:
            with open(self.stats_file, 'w', encoding='utf-8') as f:
                json.dump(stats, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving storage stats: {e}")
            
        return stats
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """
        Get current storage statistics.
        
        Returns:
            Dictionary of storage statistics
        """
        if os.path.exists(self.stats_file):
            try:
                with open(self.stats_file, 'r', encoding='utf-8') as f:
                    stats = json.load(f)

                timestamp = datetime.datetime.fromisoformat(stats["timestamp"])
                now = datetime.datetime.now()

                if (now - timestamp).total_seconds() > 300:
                    stats = self.update_storage_stats()
                    
                return stats
            except Exception as e:
                logger.error(f"Error reading storage stats: {e}")
                
        return self.update_storage_stats()
    
    def check_storage_limits(self) -> bool:
        """Check if storage is within limits.
        
        Returns:
            True if storage is within limits, False otherwise
        """
        stats = self.get_storage_stats()
        return stats["total_size_bytes"] <= self.max_storage_bytes
    
    def clear_subdir(self, subdir: str, age_days: Optional[int] = None) -> int:
        """
        Clear files from a subdirectory.
        
        Args:
            subdir: Subdirectory name
            age_days: Only clear files older than this many days (None for all files)
            
        Returns:
            Number of files cleared
        """
        subdir_path = os.path.join(self.cache_dir, subdir)
        if not os.path.exists(subdir_path):
            return 0
            
        file_count = 0
        cutoff_time = time.time() - (age_days * 86400) if age_days else 0
        
        for path in Path(subdir_path).rglob('*'):
            if path.is_file() and not path.is_symlink():
                if path.name == "storage_stats.json":
                    continue
                    
                if age_days is None or path.stat().st_mtime < cutoff_time:
                    try:
                        path.unlink()
                        file_count += 1
                    except Exception as e:
                        logger.error(f"Error deleting file {path}: {e}")
                        
        logger.info(f"Cleared {file_count} files from {subdir}")
        return file_count
    
    def clear_cache(self, age_days: Optional[int] = None) -> Dict[str, int]:
        """Clear all cache directories.
        
        Args:
            age_days: Only clear files older than this many days (None for all files)
            
        Returns:
            Dictionary with number of files cleared per subdirectory
        """
        cleared = {}
        
        for subdir in self.subdirs:
            cleared[subdir] = self.clear_subdir(subdir, age_days)
            
        self.update_storage_stats()
        
        return cleared
    
    def optimize_storage(self) -> Dict[str, Any]:
        """
        Optimize storage usage by removing old files if needed.
        
        Returns:
            Results of optimization
        """
        results = {
            "before_size_mb": 0,
            "after_size_mb": 0,
            "cleared_files": {}
        }
        
        stats = self.get_storage_stats()
        results["before_size_mb"] = stats["total_size_mb"]

        if stats["usage_percent"] < 90:
            results["action"] = "none"
            results["after_size_mb"] = stats["total_size_mb"]
            return results

        cleared = self.clear_cache(age_days=30)
        results["cleared_files"]["older_than_30_days"] = sum(cleared.values())

        if self.check_storage_limits():
            stats = self.update_storage_stats()
            if stats["usage_percent"] < 90:
                results["action"] = "cleared_old"
                results["after_size_mb"] = stats["total_size_mb"]
                return results

        cleared = self.clear_cache(age_days=7)
        results["cleared_files"]["older_than_7_days"] = sum(cleared.values())

        if self.check_storage_limits():
            stats = self.update_storage_stats()
            if stats["usage_percent"] < 90:
                results["action"] = "cleared_recent"
                results["after_size_mb"] = stats["total_size_mb"]
                return results

        for subdir in ["queries", "llm", "retriever"]:
            cleared = self.clear_subdir(subdir)
            results["cleared_files"][subdir] = cleared

            if self.check_storage_limits():
                stats = self.update_storage_stats()
                if stats["usage_percent"] < 90:
                    results["action"] = f"cleared_{subdir}"
                    results["after_size_mb"] = stats["total_size_mb"]
                    return results

        for subdir in self.subdirs:
            if subdir != "vector_store":
                cleared = self.clear_subdir(subdir)
                results["cleared_files"][subdir] = cleared
        
        results["action"] = "cleared_most"
        stats = self.update_storage_stats()
        results["after_size_mb"] = stats["total_size_mb"]
        
        return results
    
    def export_storage_report(self, output_path: Optional[str] = None) -> str:
        """
        Export a detailed storage report.
        
        Args:
            output_path: Output file path (None for default)
            
        Returns:
            Path to the exported report
        """
        if output_path is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(self.cache_dir, f"storage_report_{timestamp}.json")
        
        stats = self.get_storage_stats()

        for subdir_name, subdir_info in stats["subdirectories"].items():
            subdir_path = subdir_info["path"]
            file_count = 0
            oldest_file = None
            newest_file = None
            
            if os.path.exists(subdir_path):
                for path in Path(subdir_path).rglob('*'):
                    if path.is_file() and not path.is_symlink():
                        file_count += 1
                        
                        mtime = path.stat().st_mtime
                        if oldest_file is None or mtime < oldest_file:
                            oldest_file = mtime
                        if newest_file is None or mtime > newest_file:
                            newest_file = mtime
            
            subdir_info["file_count"] = file_count
            
            if oldest_file:
                subdir_info["oldest_file"] = datetime.datetime.fromtimestamp(oldest_file).isoformat()
            if newest_file:
                subdir_info["newest_file"] = datetime.datetime.fromtimestamp(newest_file).isoformat()

        stats["system_info"] = {
            "platform": os.name,
            "python_version": os.sys.version,
            "report_time": datetime.datetime.now().isoformat()
        }

        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(stats, f, indent=2)
            logger.info(f"Exported storage report to {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Error exporting storage report: {e}")
            return ""
        
    def cleanup_old_files(self, max_age_days: int = 90) -> int:
        """
        Clean up files older than a certain age.
        
        Args:
            max_age_days: Maximum age of files in days
            
        Returns:
            Number of files removed
        """
        return sum(self.clear_cache(age_days=max_age_days).values())
    
    def get_oldest_files(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get the oldest files in the cache.
        
        Args:
            limit: Maximum number of files to return
            
        Returns:
            List of file information dictionaries
        """
        files = []
        
        for subdir in self.subdirs:
            subdir_path = os.path.join(self.cache_dir, subdir)
            if not os.path.exists(subdir_path):
                continue
                
            for path in Path(subdir_path).rglob('*'):
                if path.is_file() and not path.is_symlink():
                    if path.name == "storage_stats.json":
                        continue
                        
                    mtime = path.stat().st_mtime
                    size = path.stat().st_size
                    
                    files.append({
                        "path": str(path),
                        "subdir": subdir,
                        "modified_time": mtime,
                        "modified_date": datetime.datetime.fromtimestamp(mtime).isoformat(),
                        "size_bytes": size,
                        "size_mb": size / (1024 * 1024)
                    })

        files.sort(key=lambda x: x["modified_time"])
        
        return files[:limit]
    
    def get_largest_files(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get the largest files in the cache.
        
        Args:
            limit: Maximum number of files to return
            
        Returns:
            List of file information dictionaries
        """
        files = []
        
        for subdir in self.subdirs:
            subdir_path = os.path.join(self.cache_dir, subdir)
            if not os.path.exists(subdir_path):
                continue
                
            for path in Path(subdir_path).rglob('*'):
                if path.is_file() and not path.is_symlink():
                    if path.name == "storage_stats.json":
                        continue
                        
                    mtime = path.stat().st_mtime
                    size = path.stat().st_size
                    
                    files.append({
                        "path": str(path),
                        "subdir": subdir,
                        "modified_time": mtime,
                        "modified_date": datetime.datetime.fromtimestamp(mtime).isoformat(),
                        "size_bytes": size,
                        "size_mb": size / (1024 * 1024)
                    })

        files.sort(key=lambda x: x["size_bytes"], reverse=True)
        
        return files[:limit]