import os 
import logging
from typing import List, Dict, Any, Optional, Union
from enum import Enum
import re 
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class QueryDomain(Enum):
    """Medical device query domain categories for specialized prompting."""
    CLASSIFICATION = "classification"     # Device classification queries
    SUBMISSION = "submission"             # Regulatory submission queries
    COMPLIANCE = "compliance"             # Compliance and QSR queries
    SOFTWARE = "software"                 # Software and SaMD queries
    TESTING = "testing"                   # Testing requirements queries
    GENERAL = "general"                   # General medical device queries
    
class PromptGenerator:
    """Generate domain-specific prompts for medical device regulatory queries.OPtimized for storage efficiency
    with minimal template storage.
    """
    def __init__(self, prompts_file: Optional[str] = None, cache_dir: Optional[str] = None):
        """Initialize the prompt generator.
        
        Args:
            prompts_file: Path to JSON file with prompt templates
            cache_dir: Directory for caching generated prompts
        """
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            self.cache_dir = cache_dir
        else:
            self.cache_dir = None
            
        self.prompt_templates = self._load_prompt_templates(prompts_file)
        
        logger.info(f"Initialized prompt generator with {len(self.prompt_templates)} templates")
        
    def _load_prompt_templates(self, filename: Optional[str]) -> Dict[str, Dict[str, str]]:
        """Load prompt templates from file or use minimal default set.
        
        Args:
            filename: Path to JSON file with prompt templates
            
        Returns:
            Dictionary of prompt templates by domain
        """
        default_templates = {
            "system": {
                "default": """
You are a Medical Device Regulatory Assistant that helps people understand FDA regulations.
Answer questions based on the provided context information from regulatory documents.
If the context doesn't contain enough information to fully answer the question, say so clearly.
Use a helpful, professional tone and cite your sources.
""",
                "classification": """
You are a Medical Device Regulatory Assistant specializing in device classification.
When discussing device classification, be precise about the class (I, II, or III) and explain the regulatory implications.
If there's any ambiguity, mention that proper classification requires review of the specific device details by regulatory professionals.
Base your answers on the regulatory context provided and cite your sources.
""",
                "submission": """
You are a Medical Device Regulatory Assistant specializing in submission pathways.
When discussing submission pathways, be clear about the requirements for 510(k), PMA, De Novo, or other submission types.
Emphasize that submission strategies should be determined by regulatory professionals based on the specific device.
Base your answers on the regulatory context provided and cite your sources.
""",
                "compliance": """
You are a Medical Device Regulatory Assistant specializing in compliance requirements.
When discussing compliance, refer to specific regulations and guidance documents.
Emphasize that compliance strategies should be developed with regulatory professionals.
Base your answers on the regulatory context provided and cite your sources.
""",
                "software": """
You are a Medical Device Regulatory Assistant specializing in software and digital health.
When discussing software and digital health, refer to the latest FDA guidance on Software as a Medical Device (SaMD).
Note that software regulations are evolving, and checking for the most recent guidance is important.
Base your answers on the regulatory context provided and cite your sources.
""",
                "testing": """
You are a Medical Device Regulatory Assistant specializing in testing requirements.
When discussing testing requirements, be specific about the types of tests needed for different device types and risk classes.
Emphasize that testing strategies should be determined by regulatory professionals based on the specific device.
Base your answers on the regulatory context provided and cite your sources.
"""
            },
            "user": {
                "default": """
Please answer the following question based on the provided context information.
If the context doesn't contain the answer, say so without making up information.

Context:
{context}

Question: {query}
""",
                "classification": """
Please answer the following question about medical device classification based on the provided regulatory information.
If the context doesn't contain enough information to determine the classification, mention that proper classification requires review of the specific device details by regulatory professionals.

Context:
{context}

Question about device classification: {query}
""",
                "submission": """
Please answer the following question about medical device submission pathways based on the provided regulatory information.
Provide specific documentation requirements where available, and emphasize that submission strategies should be determined by regulatory professionals.

Context:
{context}

Question about submission pathway: {query}
""",
                "software": """
Please answer the following question about medical device software requirements based on the provided regulatory information.
Software regulations are evolving, so note if the guidance may not reflect the most recent changes.

Context:
{context}

Question about medical device software: {query}
"""
            }
        }

        if filename and os.path.exists(filename):
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    loaded_templates = json.load(f)
                logger.info(f"Loaded prompt templates from {filename}")
                return loaded_templates
            except Exception as e:
                logger.warning(f"Error loading prompt templates file: {e}. Using defaults.")
                
        return default_templates
    
    def identify_query_domain(self, query: str) -> QueryDomain:
        """
        Identify the domain of a medical device query.
        
        Args:
            query: User query
            
        Returns:
            QueryDomain enum value
        """
        query_lower = query.lower()

        classification_patterns = [
            r'class\s+(i|ii|iii)',
            r'device\s+classification',
            r'(classify|categorize)\s+my\s+device',
            r'risk\s+based\s+classification'
        ]

        submission_patterns = [
            r'510\(k\)',
            r'premarket\s+(approval|notification)',
            r'pma\b',
            r'de\s+novo',
            r'how\s+to\s+submit',
            r'submission\s+requirement'
        ]

        compliance_patterns = [
            r'(compliance|conform)',
            r'regulation',
            r'quality\s+system',
            r'qsr\b',
            r'(report|reporting)',
            r'(recall|adverse\s+event)'
        ]

        software_patterns = [
            r'software',
            r'samd\b',
            r'(mobile|digital)\s+(app|health)',
            r'cybersecurity',
            r'software\s+validation'
        ]

        testing_patterns = [
            r'(test|testing)',
            r'(clinical|validation)\s+(trial|study)',
            r'bench\s+testing',
            r'biocompatibility',
            r'sterilization'
        ]

        for pattern in classification_patterns:
            if re.search(pattern, query_lower):
                return QueryDomain.CLASSIFICATION
                
        for pattern in submission_patterns:
            if re.search(pattern, query_lower):
                return QueryDomain.SUBMISSION
                
        for pattern in compliance_patterns:
            if re.search(pattern, query_lower):
                return QueryDomain.COMPLIANCE
                
        for pattern in software_patterns:
            if re.search(pattern, query_lower):
                return QueryDomain.SOFTWARE
                
        for pattern in testing_patterns:
            if re.search(pattern, query_lower):
                return QueryDomain.TESTING
                
        return QueryDomain.GENERAL
    
    def generate_system_prompt(self, query: str, domain: Optional[QueryDomain] = None) -> str:
        """Generate a system prompt for a given query and domain.
        
        Args:
            query: User query
            domain: Query domain (if None, will be identified from query)
            
        Returns:
            System prompt
        """
        if domain is None:
            domain = self.identify_query_domain(query)
            
        domain_name = domain.value

        template = self.prompt_templates.get("system", {}).get(domain_name)
        if not template:
            template = self.prompt_templates.get("system", {}).get("default", "")

        return template
    
    def generate_user_prompt(self, query: str, context: str, domain: Optional[QueryDomain] = None) -> str:
        """Generate a user prompt for a given query, context, and domain.
        
        Args:
            query: User query
            context: Context information from retrieved documents
            domain: Query domain (if None, will be identified from query)
            
        Returns:
            User prompt
        """
        if domain is None:
            domain = self.identify_query_domain(query)
            
        domain_name = domain.value

        template = self.prompt_templates.get("user", {}).get(domain_name)
        if not template:
            template = self.prompt_templates.get("user", {}).get("default", "")

        filled_template = template.replace("{query}", query).replace("{context}", context)
        
        return filled_template
    
    def generate_full_prompt(self, query: str, context: str, domain: Optional[QueryDomain] = None) -> Dict[str, str]:
        """Generate both system and user prompts for a given query.
        
        Args:
            query: User query
            context: Context information from retrieved documents
            domain: Query domain (if None, will be identified from query)
            
        Returns:
            Dictionary with system and user prompts
        """
        if domain is None:
            domain = self.identify_query_domain(query)
            
        system_prompt = self.generate_system_prompt(query, domain)
        user_prompt = self.generate_user_prompt(query, context, domain)
        
        return {
            "system": system_prompt,
            "user": user_prompt,
            "domain": domain.value
        }
        
    def enhance_prompts_for_device_type(self, prompts: Dict[str, str], device_type: str) -> Dict[str, str]:
        """Enhance prompts with device-specific information.
        
        Args:
            prompts: Dictionary with system and user prompts
            device_type: Type of medical device
            
        Returns:
            Enhanced prompts
        """
        enhanced = dict(prompts)
        
        if device_type and "system" in enhanced:
            device_info = f"\nThe query appears to be about a {device_type} device. "
            
            if device_type.lower() == "implantable":
                device_info += "Implantable devices often require more rigorous testing and may be Class III devices requiring PMA."
            elif device_type.lower() == "software":
                device_info += "Software as Medical Device (SaMD) has specific regulatory considerations outlined in FDA guidance documents."
            elif device_type.lower() == "ivd":
                device_info += "In vitro diagnostic devices have specific regulatory considerations under FDA guidelines."
                
            enhanced["system"] = enhanced["system"] + device_info
            
        return enhanced
    
    def format_response(self, answer: str, sources: List[Dict[str, Any]], domain: QueryDomain) -> str:
        """Format the response with appropriate domain-specific elements.
        
        Args:
            answer: Generated answer
            sources: List of source documents
            domain: Query domain
            
        Returns:
            Formatted response
        """
        formatted = answer

        if domain == QueryDomain.CLASSIFICATION:
            class_match = re.search(r'class\s+(i|ii|iii)', answer, re.IGNORECASE)
            if class_match:
                class_str = class_match.group(1).upper()
                formatted = re.sub(
                    r'(class\s+)(i|ii|iii)', 
                    r'\1**\2**', 
                    formatted, 
                    flags=re.IGNORECASE
                )
                
        elif domain == QueryDomain.SUBMISSION:
            submission_terms = [
                r'510\(k\)', r'PMA', r'De Novo', r'HDE', 
                r'premarket approval', r'premarket notification'
            ]
            for term in submission_terms:
                formatted = re.sub(
                    r'\b(' + term + r')\b', 
                    r'**\1**', 
                    formatted, 
                    flags=re.IGNORECASE
                )

        if sources and "source" not in formatted.lower():
            formatted += "\n\n**Sources:**\n"
            for i, source in enumerate(sources[:3]):
                title = source.get("title", "Untitled Document")
                source_type = source.get("source", "Unknown Source")
                formatted += f"{i+1}. {title} ({source_type})\n"
                
        return formatted
    
    def add_prompt_template(self, prompt_type: str, domain: str, template: str) -> bool:
        """Add a new prompt template.
        
        Args:
            prompt_type: Type of prompt (system or user)
            domain: Domain name
            template: Prompt template
            
        Returns:
            True if added successfully, False otherwise
        """
        if prompt_type not in self.prompt_templates:
            self.prompt_templates[prompt_type] = {}
            
        self.prompt_templates[prompt_type][domain] = template

        if self.cache_dir:
            try:
                cache_path = os.path.join(self.cache_dir, "prompt_templates.json")
                with open(cache_path, 'w', encoding='utf-8') as f:
                    json.dump(self.prompt_templates, f, indent=2)
                logger.info(f"Saved updated prompt templates to {cache_path}")
            except Exception as e:
                logger.warning(f"Error saving prompt templates: {e}")
                
        return True
    
    def save_templates(self, filepath: str) -> bool:
        """Save prompt templates to a file.
        
        Args:
            filepath: Path to save templates
            
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.prompt_templates, f, indent=2)
            logger.info(f"Saved prompt templates to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error saving prompt templates: {e}")
            return False