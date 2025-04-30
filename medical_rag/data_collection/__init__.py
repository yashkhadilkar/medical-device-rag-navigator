"""
Data collection package for Medical Device RAG System

This package provides modules for collecting regulatory documents and 
data from FDA and other sources in a storage-efficient manner.
"""
from .web_scraper import FDAWebScraper
from .text_extractor import TextExtractor
from .resource_manager import ResourceManager

__all__ = ['FDAWebScraper', 'TextExtractor', 'ResourceManager']
