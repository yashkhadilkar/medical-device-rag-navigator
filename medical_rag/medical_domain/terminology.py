import os 
import re 
import logging
import json
from typing import List, Dict, Any, Optional, Set, Tuple
from enum import Enum, auto
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DeviceType(Enum):
    """Types of medical devices based on their function and use."""
    DIAGNOSTIC = auto()  # For diagnosis of disease or conditions
    THERAPEUTIC = auto()  # For treatment of disease or conditions
    MONITORING = auto()  # For monitoring patient conditions
    SURGICAL = auto()  # Used in surgical procedures
    IMPLANTABLE = auto()  # Devices meant to be implanted in the body
    WEARABLE = auto()  # Wearable devices
    SOFTWARE = auto()  # Software as a Medical Device
    IVD = auto()  # In vitro diagnostic devices
    GENERAL = auto()  # General purpose or other devices
    
class RegulatoryPath(Enum):
    """Regulatory submission types and paths."""
    PRE_510K = auto()  # Pre-submission for 510(k)
    TRADITIONAL_510K = auto()  # Traditional 510(k)
    SPECIAL_510K = auto()  # Special 510(k)
    ABBREVIATED_510K = auto()  # Abbreviated 510(k)
    DE_NOVO = auto()  # De Novo Classification
    PMA = auto()  # Premarket Approval
    HDE = auto()  # Humanitarian Device Exemption
    EMERGENCY_USE = auto()  # Emergency Use Authorization
    EXEMPT = auto()  # Exempt from premarket submission
    GENERAL = auto()  # General or unknown regulatory path
    
class MedicalTerminologyProcessor:
    """Process medical device terminology, optimize regulatory queries, and
    improve retrieval with domain-specific knowledge. 
    Storage-efficient implementation focused on key regulatory terms. 
    """
    
    def __init__(self, terminology_file: Optional[str] = None, cache_dir: Optional[str] = ".cache/terminology", use_minimal_set: bool = True):
        """Initialize the medical terminology processor.

        Args:
            terminology_file: Path to JSON file with terminology
            cache_dir : Directory for caching processed terms
            use_minimal_set: Whether to use minimal terminology to save storage.
        """
        
        self.use_minimal_set = use_minimal_set

        self.device_types = {}
        self.regulatory_paths = {}
        self.device_classifications = {}
        self.regulatory_bodies = {}
        self.common_standards = {}

        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            self.cache_dir = cache_dir
        else:
            self.cache_dir = None

        self.terminology = self._load_terminology(terminology_file)

        self.patterns = self._generate_patterns()
        
        logger.info(f"Initialized medical terminology processor with {len(self.terminology)} term categories")

        self._build_classification_maps()
        
    def _load_terminology(self, filename: Optional[str]) -> Dict[str, List[str]]:
        """Load medical device terminology from file or use minimal default set.
        
        Args:
            filename: Path to JSON file with terminology
            
        Returns:
            Dictionary of term categories and terms
        """
        default_terms = {
            "device_types": [
                "implant", "pacemaker", "catheter", "stent", "diagnostic", 
                "imaging", "monitor", "surgical", "defibrillator", "ventilator",
                "software", "wearable", "insulin pump", "blood glucose", "IVD"
            ],
            "regulatory_paths": [
                "510(k)", "PMA", "De Novo", "HDE", "EUA", "exempt",
                "premarket notification", "premarket approval", "humanitarian device"
            ],
            "device_classifications": [
                "Class I", "Class II", "Class III", 
                "low risk", "moderate risk", "high risk"
            ],
            "regulatory_bodies": [
                "FDA", "CDRH", "OHT", "OPEQ", "CBER"
            ],
            "common_standards": [
                "ISO 13485", "ISO 14971", "IEC 60601", "IEC 62304", "ISO 10993"
            ],
            "quality_system": [
                "QSR", "QMS", "21 CFR 820", "MDSAP", "design control", 
                "CAPA", "change control", "verification", "validation"
            ],
            "submission_requirements": [
                "substantial equivalence", "predicate", "intended use",
                "special controls", "clinical data", "biocompatibility",
                "sterilization", "shelf life", "performance testing"
            ]
        }

        if self.use_minimal_set:
            for category in default_terms:
                if len(default_terms[category]) > 10:
                    default_terms[category] = default_terms[category][:10]

        if filename and os.path.exists(filename):
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    loaded_terms = json.load(f)
                logger.info(f"Loaded terminology from {filename}")
                return loaded_terms
            except Exception as e:
                logger.warning(f"Error loading terminology file: {e}. Using defaults.")
                
        return default_terms
    
    def _generate_patterns(self) -> Dict[str, List[re.Pattern]]:
        """Generate regex patterns for efficient terminology matching.
        
        Returns:
            Dictionary of term categories and compiled regex patterns
        """
        patterns = {}
        
        for category, terms in self.terminology.items():
            category_patterns = []
            for term in terms:
                escaped_term = re.escape(term)
                pattern = re.compile(r'\b' + escaped_term + r'\b', re.IGNORECASE)
                category_patterns.append(pattern)
            patterns[category] = category_patterns
            
        return patterns
    
    def _build_classification_maps(self) -> None:
        """Build lookup maps for device classification."""
        self.device_type_map = {
            "implant": DeviceType.IMPLANTABLE,
            "pacemaker": DeviceType.IMPLANTABLE,
            "stent": DeviceType.IMPLANTABLE,
            "catheter": DeviceType.THERAPEUTIC,
            "diagnostic": DeviceType.DIAGNOSTIC,
            "imaging": DeviceType.DIAGNOSTIC,
            "monitor": DeviceType.MONITORING,
            "surgical": DeviceType.SURGICAL,
            "defibrillator": DeviceType.THERAPEUTIC,
            "ventilator": DeviceType.THERAPEUTIC,
            "software": DeviceType.SOFTWARE,
            "wearable": DeviceType.WEARABLE,
            "blood glucose": DeviceType.DIAGNOSTIC,
            "ivd": DeviceType.IVD,
            "in vitro diagnostic": DeviceType.IVD
        }

        self.regulatory_path_map = {
            "510(k)": RegulatoryPath.TRADITIONAL_510K,
            "traditional 510(k)": RegulatoryPath.TRADITIONAL_510K,
            "special 510(k)": RegulatoryPath.SPECIAL_510K,
            "abbreviated 510(k)": RegulatoryPath.ABBREVIATED_510K,
            "de novo": RegulatoryPath.DE_NOVO,
            "pma": RegulatoryPath.PMA,
            "premarket approval": RegulatoryPath.PMA,
            "hde": RegulatoryPath.HDE,
            "humanitarian device": RegulatoryPath.HDE,
            "eua": RegulatoryPath.EMERGENCY_USE,
            "emergency use": RegulatoryPath.EMERGENCY_USE,
            "exempt": RegulatoryPath.EXEMPT
        }
        
    def extract_terms(self, text: str) -> Dict[str, List[str]]:
        """Extract medical device terminology from text.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of term categories and matched terms
        """
        if not text:
            return {}
            
        results = {}
        
        for category, patterns in self.patterns.items():
            matches = []
            for pattern in patterns:
                matches.extend(pattern.findall(text.lower()))
            
            if matches:
                unique_matches = []
                for match in matches:
                    if match not in unique_matches:
                        unique_matches.append(match)
                
                results[category] = unique_matches
                
        return results
    
    def identify_device_type(self, text: str) -> DeviceType:
        """Identify the type of medical device mentioned in text.
        
        Args:
            text: Input text
            
        Returns:
            DeviceType enum value
        """
        text_lower = text.lower()
        
        for term, device_type in self.device_type_map.items():
            if term in text_lower:
                return device_type
                
        return DeviceType.GENERAL
    
    def identify_regulatory_path(self, text: str) -> RegulatoryPath:
        """Identify the regulatory path mentioned in text.
        
        Args:
            text: Input text
            
        Returns:
            RegulatoryPath enum value
        """
        text_lower = text.lower()
        
        for term, reg_path in self.regulatory_path_map.items():
            if term in text_lower:
                return reg_path
                
        return RegulatoryPath.GENERAL
    
    def extract_device_class(self, text: str) -> Optional[int]:
        """Extract device classification (Class I, II, or III) from text.
        
        Args:
            text: Input text
            
        Returns:
            Device class as integer (1, 2, 3) or None if not found
        """
        class_match = re.search(r'\bclass\s+(i|ii|iii|1|2|3)\b', text, re.IGNORECASE)
        if class_match:
            class_str = class_match.group(1).lower()
            if class_str in ('i', '1'):
                return 1
            elif class_str in ('ii', '2'):
                return 2
            elif class_str in ('iii', '3'):
                return 3

        risk_match = re.search(r'\b(low|moderate|high)\s+risk\b', text, re.IGNORECASE)
        if risk_match:
            risk_str = risk_match.group(1).lower()
            if risk_str == 'low':
                return 1
            elif risk_str == 'moderate':
                return 2
            elif risk_str == 'high':
                return 3
                
        return None
    
    def enhance_query(self, query: str) -> str:
        """Enhance a query with relevant medical device terminology.
        
        Args:
            query: Original query
            
        Returns:
            Enhanced query
        """
        extracted_terms = self.extract_terms(query)

        if not extracted_terms:
            return query

        device_type = self.identify_device_type(query)
        regulatory_path = self.identify_regulatory_path(query)
        device_class = self.extract_device_class(query)

        enhancements = []

        if device_type != DeviceType.GENERAL:
            if device_type == DeviceType.IMPLANTABLE:
                enhancements.append("implantable medical device")
            elif device_type == DeviceType.DIAGNOSTIC:
                enhancements.append("diagnostic medical device")
            elif device_type == DeviceType.THERAPEUTIC:
                enhancements.append("therapeutic medical device")
            elif device_type == DeviceType.SOFTWARE:
                enhancements.append("software as a medical device SaMD")
            elif device_type == DeviceType.IVD:
                enhancements.append("in vitro diagnostic IVD")

        if regulatory_path != RegulatoryPath.GENERAL:
            if regulatory_path == RegulatoryPath.TRADITIONAL_510K:
                enhancements.append("510(k) submission")
            elif regulatory_path == RegulatoryPath.PMA:
                enhancements.append("premarket approval PMA")
            elif regulatory_path == RegulatoryPath.DE_NOVO:
                enhancements.append("De Novo classification")

        if device_class:
            enhancements.append(f"Class {device_class} medical device")

        if not enhancements:
            if "classification" in query.lower():
                enhancements.append("device classification FDA regulatory class")
            elif "submission" in query.lower():
                enhancements.append("FDA submission regulatory 510k PMA")
            elif "software" in query.lower():
                enhancements.append("medical device software SaMD")
            elif "testing" in query.lower():
                enhancements.append("medical device testing requirements validation")

        enhanced_query = query
        if enhancements:
            enhancement_str = " " + " ".join(enhancements)
            enhanced_query = f"{query} {enhancement_str}"
        
        return enhanced_query
    
    def generate_search_queries(self, query: str, max_queries: int = 2) -> List[str]:
        """Generate search queries from a user query, incorporating domain knowledge.
        
        Args:
            query: Original user query
            max_queries: Maximum number of queries to generate
            
        Returns:
            List of search queries
        """
        queries = [query]  

        terms = self.extract_terms(query)

        device_type = self.identify_device_type(query)
        regulatory_path = self.identify_regulatory_path(query)
        device_class = self.extract_device_class(query)

        if "class" in query.lower() and device_type != DeviceType.GENERAL:
            device_type_str = device_type.name.lower()
            queries.append(f"{device_type_str} device classification FDA")
            
        elif regulatory_path != RegulatoryPath.GENERAL:
            if regulatory_path == RegulatoryPath.TRADITIONAL_510K:
                queries.append("510(k) submission requirements FDA")
            elif regulatory_path == RegulatoryPath.PMA:
                queries.append("premarket approval PMA requirements FDA")
            elif regulatory_path == RegulatoryPath.DE_NOVO:
                queries.append("De Novo classification process FDA")
                
        elif device_class and "requirements" in query.lower():
            queries.append(f"Class {device_class} medical device requirements FDA")
            
        elif "software" in query.lower():
            queries.append("software as a medical device FDA guidance")

        return queries[:max_queries]
    
    def analyze_document(self, text: str, title: Optional[str] = None) -> Dict[str, Any]:
        """Analyze document content for device classification and regulatory paths.
        
        Args:
            text: Document text
            title: Document title
            
        Returns:
            Dictionary with analysis results
        """
        results = {
            "device_types": [],
            "regulatory_paths": [],
            "device_class": None,
            "standards_referenced": [],
            "submission_requirements": []
        }
        
        extracted_terms = self.extract_terms(text)

        if "device_types" in extracted_terms:
            for term in extracted_terms["device_types"]:
                if term in self.device_type_map:
                    device_type = self.device_type_map[term]
                    if device_type.name not in results["device_types"]:
                        results["device_types"].append(device_type.name)

        if "regulatory_paths" in extracted_terms:
            for term in extracted_terms["regulatory_paths"]:
                if term in self.regulatory_path_map:
                    reg_path = self.regulatory_path_map[term]
                    if reg_path.name not in results["regulatory_paths"]:
                        results["regulatory_paths"].append(reg_path.name)

        results["device_class"] = self.extract_device_class(text)

        if "common_standards" in extracted_terms:
            results["standards_referenced"] = extracted_terms["common_standards"]

        if "submission_requirements" in extracted_terms:
            results["submission_requirements"] = extracted_terms["submission_requirements"]
            
        return results
    
    def get_keyword_dictionary(self) -> Dict[str, List[str]]:
        """Get a dictionary of keywords for enhancing retrieval.
        
        Returns:
            Dictionary of keyword categories and terms
        """
        return dict(self.terminology)
    
    def save_terminology(self, filepath: str) -> bool:
        """Save the current terminology to a file.
        
        Args:
            filepath: Path to save the terminology
            
        Returns:
            True if successful, False otherwise
        """
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.terminology, f, indent=2)
            logger.info(f"Saved terminology to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error saving terminology: {e}")
            return False
        
    def add_term(self, category: str, term: str) -> bool:
        """Add a term to the terminology.
        
        Args:
            category: Term category
            term: Term to add
            
        Returns:
            True if added, False otherwise
        """
        if category not in self.terminology:
            self.terminology[category] = []
            
        if term.lower() not in [t.lower() for t in self.terminology[category]]:
            self.terminology[category].append(term)

            escaped_term = re.escape(term)
            pattern = re.compile(r'\b' + escaped_term + r'\b', re.IGNORECASE)
            
            if category not in self.patterns:
                self.patterns[category] = []
                
            self.patterns[category].append(pattern)
            
            return True
        return False
        
    def expand_abbreviations(self, text: str) -> str:
        """Expand common medical device abbreviations in text.
        
        Args:
            text: Input text
            
        Returns:
            Text with expanded abbreviations
        """
        abbreviations = {
            r'\b510k\b': '510(k)',
            r'\bpma\b': 'Premarket Approval',
            r'\bmdr\b': 'Medical Device Reporting',
            r'\bqsr\b': 'Quality System Regulation',
            r'\budi\b': 'Unique Device Identification',
            r'\bcfr\b': 'Code of Federal Regulations',
            r'\bsaMD\b': 'Software as a Medical Device',
            r'\bivd\b': 'In Vitro Diagnostic',
            r'\bcapa\b': 'Corrective and Preventive Action'
        }
        
        expanded = text
        for abbr, expansion in abbreviations.items():
            expanded = re.sub(abbr, expansion, expanded, flags=re.IGNORECASE)
            
        return expanded