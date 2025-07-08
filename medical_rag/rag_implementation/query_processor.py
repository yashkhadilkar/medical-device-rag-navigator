import re 
import logging
import json
from typing import List, Dict, Any, Optional, Tuple, Set
from pathlib import Path
import os
import time
import hashlib
from enum import Enum

try:
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    NLTK_AVAILABLE = True
    
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt_tab', quiet=True)
        
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)
        
except ImportError:
    NLTK_AVAILABLE = False
    logging.warning("NLTK not available. Install with: pip install nltk")

try:
    from spellchecker import SpellChecker
    SPELL_CHECKER_AVAILABLE = True
except ImportError:
    SPELL_CHECKER_AVAILABLE = False
    logging.warning("SpellChecker not available. Install with: pip install pyspellchecker")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class QueryType(Enum):
    """Types of medical device regulatory queries.
    """
    CLASSIFICATION = "classification"
    SUBMISSION = "submission"
    COMPLIANCE = "compliance"
    SOFTWARE = "software"
    TESTING = "testing"
    GENERAL = "general"
    
class QueryProcessor:
    """Process and optimize queries for the medical device RAG System.
    Improves retrieval by understanding common regulatory questions.
    """
    
    def __init__(self, 
                 domain_terms_file: Optional[str] = None,
                 cache_dir: str = ".cache/queries",
                 enable_advanced_nlp: bool = False,
                 enable_spell_check: bool = True,
                 medical_terms_file: Optional[str] = None):
        """
        Initialize the query processor.
        
        Args:
            domain_terms_file: Path to JSON file with domain terms
            cache_dir: Directory for caching query results
            enable_advanced_nlp: Whether to use advanced NLP (more resource-intensive)
            enable_spell_check: Whether to enable spell checking for queries
            medical_terms_file: Path to file with medical terminology for spell checking
        """
        self.enable_advanced_nlp = enable_advanced_nlp and NLTK_AVAILABLE
        self.enable_spell_check = enable_spell_check and SPELL_CHECKER_AVAILABLE
        self.domain_terms = self._load_domain_terms(domain_terms_file)
        self.query_history = []
        
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            self.cache_dir = cache_dir
        else:
            self.cache_dir = None
            
        if self.enable_advanced_nlp:
            self.stop_words = set(stopwords.words('english'))
        else:
            self.stop_words = set()
            
        self.domain_patterns = {
            QueryType.CLASSIFICATION: [
                r'class\s+(i|ii|iii)',
                r'device\s+classification',
                r'(classify|categorize)\s+my\s+device'
            ],
            QueryType.SUBMISSION: [
                r'510\(k\)',
                r'premarket\s+(approval|notification)',
                r'pma\b',
                r'de\s+novo',
                r'how\s+to\s+submit'
            ],
            QueryType.COMPLIANCE: [
                r'(compliance|conform)',
                r'regulation',
                r'quality\s+system',
                r'qsr\b',
                r'(report|reporting)'
            ],
            QueryType.SOFTWARE: [
                r'software',
                r'samd\b',
                r'(mobile|digital)\s+(app|health)',
                r'cybersecurity'
            ],
            QueryType.TESTING: [
                r'(test|testing)',
                r'(clinical|validation)\s+(trial|study)',
                r'bench\s+testing',
                r'biocompatibility'
            ]
        }

        if self.enable_spell_check:
            self.spell_checker = SpellChecker()
            self._load_medical_terms(medical_terms_file)
            logger.info("Spell checker initialized")
        
        logger.info(f"Initialized query processor with advanced NLP: {self.enable_advanced_nlp}, spell checking: {self.enable_spell_check}")
        
    def _load_domain_terms(self, filename: Optional[str]) -> Dict[str, List[str]]:
        """Load domain-specific terms for query enhancement."""
        default_terms = {
            "device_types": [
                "implant", "diagnostic", "therapeutic", "monitor", "surgical", "wearable"
            ],
            "regulatory_paths": [
                "510(k)", "PMA", "De Novo", "HDE", "Emergency Use Authorization"  
            ],
            "regulations": [
                "QSR", "MDR", "UDI", "GUDID", "CFR", "Part 820"
            ],
            "common_concerns": [
                "biocompatibility", "sterilization", "software validation", "risk management"
            ],
            "agencies": [
                "FDA", "CDRH", "EMA", "Health Canada", "PMDA", "NMPA"
            ]
        }
        
        if filename and os.path.exists(filename):
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    loaded_terms = json.load(f)
                    
                logger.info(f"Loaded domain terms from {filename}")
                return loaded_terms
            except Exception as e:
                logger.error(f"Error loading domain terms: {e}")
                
        return default_terms
    
    def _load_medical_terms(self, filename: Optional[str]) -> None:
        """
        Load medical and regulatory terminology for spell checking.
        
        Args:
            filename: Path to the medical terms file
        """
        if not self.enable_spell_check:
            return

        default_terms = [
            "510k", "510(k)", "PMA", "De Novo", "HDE", "CFR", "MDR", "UDI", "QSR", 
            "GUDID", "FDA", "CDRH", "SaMD", "biocompatibility", "premarket",

            "implantable", "implant", "pacemaker", "catheter", "stent", "diagnostic",
            "therapeutic", "monitoring", "surgical", "wearable", "glucose", "insulin",
            "defibrillator", "ventilator", "orthopedic", "dental", "ophthalmic",
            "cardiovascular", "neurological", "radiology", "anesthesiology",

            "Class I", "Class II", "Class III", "exempt", "non-exempt", "substantial",
            "equivalence", "predicate", "special controls", "general controls"
        ]

        for term in default_terms:
            self.spell_checker.word_frequency.add(term)

            self.spell_checker.word_frequency.add(term.lower())

            if " " in term:
                self.spell_checker.word_frequency.add(term.replace(" ", ""))

        if filename and os.path.exists(filename):
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    custom_terms = [line.strip() for line in f if line.strip()]
                
                for term in custom_terms:
                    self.spell_checker.word_frequency.add(term)
                
                logger.info(f"Loaded {len(custom_terms)} medical terms from {filename}")
            except Exception as e:
                logger.error(f"Error loading medical terms: {e}")
    
    def correct_spelling(self, query: str) -> str:
        """
        Correct spelling errors in the query.
        
        Args:
            query: Original query with potential typos
            
        Returns:
            Corrected query
        """
        if not self.enable_spell_check:
            return query
            
        try:
            words = re.findall(r'\b[\w\']+\b', query)
            corrected_words = []
            
            for word in words:
                if (word.isdigit() or 
                    len(word) <= 2 or 
                    word.isupper() or
                    any(char.isdigit() for char in word)):
                    corrected_words.append(word)
                    continue

                if self.spell_checker.unknown([word]):
                    correction = self.spell_checker.correction(word)

                    if correction and self._edit_distance(word, correction) <= 2:
                        corrected_words.append(correction)
                    else:
                        corrected_words.append(word)
                else:
                    corrected_words.append(word)

            corrected_query = query
            for original, corrected in zip(words, corrected_words):
                if original != corrected:
                    pattern = r'\b' + re.escape(original) + r'\b'
                    corrected_query = re.sub(pattern, corrected, corrected_query)

            if corrected_query != query:
                logger.info(f"Corrected query: '{query}' -> '{corrected_query}'")
                
            return corrected_query
            
        except Exception as e:
            logger.warning(f"Error in spell checking: {e}")
            return query
            
    def _edit_distance(self, s1: str, s2: str) -> int:
        """
        Calculate the Levenshtein edit distance between two strings.
        
        Args:
            s1: First string
            s2: Second string
            
        Returns:
            Edit distance (number of character changes needed)
        """
        if len(s1) < len(s2):
            return self._edit_distance(s2, s1)
            
        if len(s2) == 0:
            return len(s1)
            
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
            
        return previous_row[-1]
    
    def preprocess_query(self, query: str) -> str:
        """
        Preprocess a user query for better retrieval.
        
        Args:
            query: Original user query
            
        Returns:
            Preprocessed query
        """
        if self.enable_spell_check:
            query = self.correct_spelling(query)
            
        processed = query.strip()
        processed = re.sub(r'\s+', ' ', processed)

        abbreviations = {
            r'\b510k\b': '510(k)',
            r'\bpma\b': 'PMA',
            r'\bsw\b': 'software',
            r'\bqa\b': 'quality assurance',
            r'\bqc\b': 'quality control',
            r'\bfda\b': 'FDA',
            r'\bcdr\b': 'Class Determination Request',
            r'\bcdrh\b': 'CDRH',
            r'\bcfr\b': 'CFR',
            r'\budi\b': 'UDI',
            r'\bmdr\b': 'MDR',
            r'\bqsr\b': 'QSR'
        }
        
        for pattern, replacement in abbreviations.items():
            processed = re.sub(pattern, replacement, processed, flags=re.IGNORECASE)
            
        return processed
    
    def identify_query_type(self, query: str) -> QueryType:
        """
        Identify the type of medical device regulatory query.
        
        Args:
            query: User query
            
        Returns:
            Query type
        """
        query_lower = query.lower()
        
        for query_type, patterns in self.domain_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query_lower, re.IGNORECASE):
                    return query_type
        
        return QueryType.GENERAL
    
    def expand_query(self, query: str) -> str:
        """
        Expand a query with domain-specific terms.
        
        Args:
            query: Original query
            
        Returns:
            Expanded query
        """
        query_type = self.identify_query_type(query)
        expanded = query

        if query_type == QueryType.CLASSIFICATION:
            expanded += " device classification regulatory class"
        elif query_type == QueryType.SUBMISSION:
            expanded += " submission FDA regulatory approval clearance"
        elif query_type == QueryType.COMPLIANCE:
            expanded += " compliance regulatory requirements regulations"
        elif query_type == QueryType.SOFTWARE:
            expanded += " software medical device digital health SaMD"
        elif query_type == QueryType.TESTING:
            expanded += " testing validation verification"
            
        return expanded
    
    def extract_keywords(self, query: str) -> List[str]:
        """
        Extract keywords from a query.
        
        Args:
            query: User query
            
        Returns:
            List of keywords
        """
        if not self.enable_advanced_nlp:
            words = re.findall(r'\b\w+\b', query.lower())
            keywords = [word for word in words if len(word) > 2 and word not in self.stop_words]
            return keywords[:10] 

        tokens = word_tokenize(query)
        keywords = [word.lower() for word in tokens 
                    if word.lower() not in self.stop_words 
                    and word.isalnum() 
                    and len(word) > 2]
        
        return keywords[:10] 
    
    def generate_search_queries(self, query: str, max_queries: int = 3) -> List[str]:
        """
        Generate search queries from the user query.
        
        Args:
            query: User query
            max_queries: Maximum number of queries to generate
            
        Returns:
            List of search queries
        """
        preprocessed = self.preprocess_query(query)
        queries = [preprocessed]

        expanded = self.expand_query(preprocessed)
        if expanded != preprocessed:
            queries.append(expanded)

        query_type = self.identify_query_type(query)
        
        if query_type == QueryType.CLASSIFICATION:
            keywords = self.extract_keywords(query)
            device_keywords = [k for k in keywords if k not in ["class", "classify", "classification"]]
            if device_keywords:
                device_terms = " ".join(device_keywords[:3])
                queries.append(f"{device_terms} FDA classification")
                
        elif query_type == QueryType.SUBMISSION:
            if "510(k)" in query:
                queries.append("510(k) submission requirements FDA")
            elif "PMA" in query or "premarket approval" in query.lower():
                queries.append("PMA requirements FDA")

        return queries[:max_queries]
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """
        Process a query and prepare it for retrieval.
        
        Args:
            query: User query
            
        Returns:
            Processed query information
        """
        corrected_query = self.correct_spelling(query) if self.enable_spell_check else query
        
        processed_info = {
            "original_query": query,
            "corrected_query": corrected_query,
            "processed_query": self.preprocess_query(corrected_query),
            "query_type": self.identify_query_type(corrected_query).value,
            "keywords": self.extract_keywords(corrected_query),
            "search_queries": self.generate_search_queries(corrected_query),
            "timestamp": time.time()
        }

        self.query_history.append({
            "query": query,
            "corrected": corrected_query if query != corrected_query else None,
            "processed": processed_info["processed_query"],
            "type": processed_info["query_type"],
            "timestamp": processed_info["timestamp"]
        })

        if len(self.query_history) > 100:
            self.query_history = self.query_history[-100:]
            
        return processed_info
    
    def save_query_stats(self, query_info: Dict[str, Any], results_count: int) -> None:
        """
        Save query statistics for later analysis.
        
        Args:
            query_info: Processed query information
            results_count: Number of results returned
        """
        if not self.cache_dir:
            return
            
        stats = query_info.copy()
        stats["results_count"] = results_count

        query_hash = hashlib.md5(query_info["original_query"].encode()).hexdigest()[:10]
        stats_path = os.path.join(self.cache_dir, f"query_{query_hash}.json")
        
        try:
            with open(stats_path, 'w', encoding='utf-8') as f:
                json.dump(stats, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"Error saving query stats: {e}")
            
    def analyze_query_trends(self, days: int = 30) -> Dict[str, Any]:
        """
        Analyze query trends from history.
        
        Args:
            days: Number of days to analyze
            
        Returns:
            Analysis of query trends
        """
        if not self.query_history:
            return {"error": "No query history available"}
            
        cutoff_time = time.time() - (days * 24 * 60 * 60)
        recent_queries = [q for q in self.query_history if q["timestamp"] >= cutoff_time]
        
        if not recent_queries:
            return {"error": f"No queries in the last {days} days"}

        type_counts = {}
        for query in recent_queries:
            query_type = query["type"]
            type_counts[query_type] = type_counts.get(query_type, 0) + 1

        all_keywords = []
        for query in recent_queries:
            keywords = self.extract_keywords(query["query"])
            all_keywords.extend(keywords)
            
        keyword_counts = {}
        for keyword in all_keywords:
            keyword_counts[keyword] = keyword_counts.get(keyword, 0) + 1
            
        top_keywords = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)[:10]

        corrections_made = sum(1 for q in recent_queries if q.get("corrected") is not None)
        
        return {
            "total_queries": len(recent_queries),
            "query_types": type_counts,
            "top_keywords": dict(top_keywords),
            "corrections_made": corrections_made,
            "correction_rate": corrections_made / len(recent_queries) if recent_queries else 0,
            "days_analyzed": days
        }
        
    def format_response(self, results: List[Dict[str, Any]], query_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format retrieval results for better presentation.
        
        Args:
            results: List of retrieval results
            query_info: Processed query information
            
        Returns:
            Formatted response
        """
        formatted = {
            "query": query_info["original_query"],
            "query_type": query_info["query_type"],
            "results_count": len(results),
            "results": []
        }

        if "corrected_query" in query_info and query_info["original_query"] != query_info["corrected_query"]:
            formatted["corrected_query"] = query_info["corrected_query"]

        docs_map = {}
        for result in results:
            doc_id = result.get("doc_id", result.get("id", "unknown"))
            if doc_id in docs_map:
                docs_map[doc_id]["chunks"].append(result)
            else:
                metadata = result.get("metadata", {})
                docs_map[doc_id] = {
                    "doc_id": doc_id,
                    "title": metadata.get("title", "Untitled Document"),
                    "source": metadata.get("source", "Unknown Source"),
                    "chunks": [result]
                }

        for doc_id, doc_info in docs_map.items():
            chunks = doc_info["chunks"]
            chunks.sort(key=lambda x: x.get("score", 0), reverse=True)
            
            formatted_doc = {
                "doc_id": doc_id,
                "title": doc_info["title"],
                "source": doc_info["source"],
                "relevance": max([chunk.get("score", 0) for chunk in chunks]),
                "excerpts": [self._format_excerpt(chunk) for chunk in chunks[:3]]  # Top 3 chunks
            }
            
            formatted["results"].append(formatted_doc)

        formatted["results"].sort(key=lambda x: x["relevance"], reverse=True)
        
        return formatted
    
    def _format_excerpt(self, chunk: Dict[str, Any]) -> str:
        """Format a chunk as an excerpt."""
        text = chunk.get("text", "")
        if not text:
            return ""

        max_length = 200
        if len(text) > max_length:
            text = text[:max_length] + "..."
            
        return text