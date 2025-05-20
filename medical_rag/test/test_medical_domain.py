#!/usr/bin/env python
"""
Test script for the Medical Domain Specialization modules.
This script tests the functionality of the terminology processor and prompt generator.
"""

import os
import logging
import sys
import json
from pathlib import Path

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)

from medical_rag.medical_domain.terminology import MedicalTerminologyProcessor, DeviceType, RegulatoryPath
from medical_rag.medical_domain.prompts import PromptGenerator, QueryDomain

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

class TestMedicalDomain:
    """Test class for Medical Domain Specialization modules."""
    
    def __init__(self, output_dir: str = "test_results"):
        """
        Initialize the test class.
        
        Args:
            output_dir: Directory to save test results
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize the components to test
        self.terminology_processor = MedicalTerminologyProcessor(use_minimal_set=True)
        self.prompt_generator = PromptGenerator()
        
        # Sample data for testing
        self.sample_queries = [
            "What is the classification of a blood glucose monitor?",
            "What is required in a 510(k) submission?",
            "How does the FDA regulate medical device software?",
            "What biocompatibility testing is needed for implantable devices?",
            "What are the labeling requirements for Class II devices?"
        ]
        
        self.sample_documents = [
            """
            Class II medical devices are subject to special controls. Special controls 
            include special labeling requirements, mandatory performance standards, 
            and postmarket surveillance. For Class II devices, the 510(k) premarket 
            notification process is usually required before marketing.
            """,
            """
            Software as a Medical Device (SaMD) is defined as software intended to be 
            used for one or more medical purposes that perform these purposes without 
            being part of a hardware medical device. FDA regulates these based on their 
            intended use and risk level.
            """,
            """
            For implantable devices, biocompatibility testing according to ISO 10993 
            series is required. This includes cytotoxicity, sensitization, irritation, 
            acute systemic toxicity, and implantation tests. Additional tests may be 
            required based on the device's specific characteristics.
            """
        ]
        
        logger.info("Initialized TestMedicalDomain")
    
    def test_terminology_processor(self):
        """Test the MedicalTerminologyProcessor."""
        results = {
            "extract_terms": [],
            "identify_device_type": [],
            "identify_regulatory_path": [],
            "extract_device_class": [],
            "enhance_query": [],
            "generate_search_queries": []
        }

        for doc in self.sample_documents:
            terms = self.terminology_processor.extract_terms(doc)
            results["extract_terms"].append({
                "document_preview": doc[:100] + "...",
                "extracted_terms": terms
            })

        for query in self.sample_queries:
            device_type = self.terminology_processor.identify_device_type(query)
            results["identify_device_type"].append({
                "query": query,
                "device_type": device_type.name
            })

        for query in self.sample_queries:
            reg_path = self.terminology_processor.identify_regulatory_path(query)
            results["identify_regulatory_path"].append({
                "query": query,
                "regulatory_path": reg_path.name
            })

        test_texts = [
            "This device is Class I and exempt from 510(k)",
            "Class II devices require special controls",
            "High risk devices are typically Class III"
        ]
        for text in test_texts:
            device_class = self.terminology_processor.extract_device_class(text)
            results["extract_device_class"].append({
                "text": text,
                "device_class": device_class
            })

        for query in self.sample_queries:
            enhanced = self.terminology_processor.enhance_query(query)
            results["enhance_query"].append({
                "original_query": query,
                "enhanced_query": enhanced
            })

        for query in self.sample_queries:
            search_queries = self.terminology_processor.generate_search_queries(query)
            results["generate_search_queries"].append({
                "original_query": query,
                "search_queries": search_queries
            })

        with open(os.path.join(self.output_dir, "terminology_processor_results.json"), "w") as f:
            json.dump(results, f, indent=2)
            
        logger.info("Completed terminology processor tests")
        return results
    
    def test_prompt_generator(self):
        """Test the PromptGenerator."""
        results = {
            "identify_query_domain": [],
            "generate_system_prompt": [],
            "generate_user_prompt": [],
            "generate_full_prompt": [],
            "format_response": []
        }

        for query in self.sample_queries:
            domain = self.prompt_generator.identify_query_domain(query)
            results["identify_query_domain"].append({
                "query": query,
                "domain": domain.value
            })

        for query in self.sample_queries:
            system_prompt = self.prompt_generator.generate_system_prompt(query)
            results["generate_system_prompt"].append({
                "query": query,
                "system_prompt": system_prompt[:200] + "..." if len(system_prompt) > 200 else system_prompt
            })

        for query in self.sample_queries:
            context = "Sample context information for testing"
            user_prompt = self.prompt_generator.generate_user_prompt(query, context)
            results["generate_user_prompt"].append({
                "query": query,
                "user_prompt": user_prompt[:200] + "..." if len(user_prompt) > 200 else user_prompt
            })

        for query in self.sample_queries:
            context = "Sample context information for testing"
            prompts = self.prompt_generator.generate_full_prompt(query, context)
            results["generate_full_prompt"].append({
                "query": query,
                "prompts": {
                    "system": prompts["system"][:100] + "..." if len(prompts["system"]) > 100 else prompts["system"],
                    "user": prompts["user"][:100] + "..." if len(prompts["user"]) > 100 else prompts["user"],
                    "domain": prompts["domain"]
                }
            })

        sample_answers = [
            "Blood glucose monitors are typically classified as Class II medical devices.",
            "A 510(k) submission requires substantial equivalence to a predicate device.",
            "The FDA regulates medical device software based on its intended use and risk level."
        ]
        
        for i, answer in enumerate(sample_answers):
            query = self.sample_queries[i]
            domain = self.prompt_generator.identify_query_domain(query) 
            sources = [
                {"title": "FDA Guidance Document", "source": "FDA"},
                {"title": "Medical Device Classification Manual", "source": "Regulatory Database"}
            ]
            formatted = self.prompt_generator.format_response(answer, sources, domain)
            results["format_response"].append({
                "original_answer": answer,
                "formatted_answer": formatted,
                "domain": domain.value 
            })

        results["add_prompt_template"] = []
        test_domain = "custom_domain"
        test_template = "This is a test prompt template for {domain}."
        success = self.prompt_generator.add_prompt_template(
            prompt_type="system", 
            domain=test_domain, 
            template=test_template
        )
        results["add_prompt_template"].append({
            "domain": test_domain,
            "success": success,
            "template_exists": test_domain in self.prompt_generator.prompt_templates.get("system", {})
        })

        results["save_templates"] = []
        temp_file = os.path.join(self.output_dir, "test_templates.json")
        success = self.prompt_generator.save_templates(temp_file)
        file_exists = os.path.exists(temp_file)
        results["save_templates"].append({
            "filepath": temp_file,
            "success": success,
            "file_exists": file_exists
        })

        with open(os.path.join(self.output_dir, "prompt_generator_results.json"), "w") as f:
            json.dump(results, f, indent=2)
            
        logger.info("Completed prompt generator tests")
        return results
    
    def test_integration(self):
        """Test the integration of terminology processor and prompt generator."""
        results = []
        
        for query in self.sample_queries:
            terms = self.terminology_processor.extract_terms(query)

            try:
                domain = self.prompt_generator.identify_query_domain(query)
                device_type = self.terminology_processor.identify_device_type(query)
                regulatory_path = self.terminology_processor.identify_regulatory_path(query)
            except Exception as e:
                logger.warning(f"Error in identification: {e}")
                domain = QueryDomain.GENERAL
                device_type = DeviceType.GENERAL
                regulatory_path = RegulatoryPath.GENERAL

            enhanced_query = self.terminology_processor.enhance_query(query)

            search_queries = self.terminology_processor.generate_search_queries(query)

            context = "Sample context from retrieved documents"
            try:
                prompts = self.prompt_generator.generate_full_prompt(query, context, domain)

                if device_type != DeviceType.GENERAL:
                    enhanced_prompts = self.prompt_generator.enhance_prompts_for_device_type(
                        prompts, device_type.name
                    )
                    device_specific = enhanced_prompts["system"] != prompts["system"]
                else:
                    device_specific = False
            except Exception as e:
                logger.warning(f"Error generating prompts: {e}")
                prompts = {
                    "system": "Default system prompt",
                    "user": "Default user prompt",
                    "domain": domain.value
                }
                device_specific = False

            sample_answer = "This is a sample answer about medical devices and FDA regulations."
            sources = [
                {"title": "FDA Guidance Document", "source": "FDA"}
            ]
            try:
                formatted_answer = self.prompt_generator.format_response(sample_answer, sources, domain)
                formatting_successful = formatted_answer != sample_answer
            except Exception as e:
                logger.warning(f"Error formatting response: {e}")
                formatted_answer = sample_answer
                formatting_successful = False

            result = {
                "query": query,
                "extracted_terms": terms,
                "domain": domain.value,
                "device_type": device_type.name,
                "regulatory_path": regulatory_path.name,
                "enhanced_query": enhanced_query,
                "search_queries": search_queries,
                "system_prompt": prompts["system"][:100] + "..." if len(prompts["system"]) > 100 else prompts["system"],
                "user_prompt": prompts["user"][:100] + "..." if len(prompts["user"]) > 100 else prompts["user"],
                "device_specific_enhancement": device_specific,
                "formatting_successful": formatting_successful
            }
            
            results.append(result)

        with open(os.path.join(self.output_dir, "integration_results.json"), "w") as f:
            json.dump(results, f, indent=2)
            
        logger.info("Completed integration tests")
        return results
    
    def run_all_tests(self):
        """Run all tests."""
        logger.info("Starting medical domain tests")
        
        terminology_results = self.test_terminology_processor()
        prompt_results = self.test_prompt_generator()
        integration_results = self.test_integration()

        summary = {
            "terminology_processor_tests": len(terminology_results["extract_terms"]),
            "prompt_generator_tests": len(prompt_results["identify_query_domain"]),
            "integration_tests": len(integration_results),
            "status": "completed"
        }
        
        with open(os.path.join(self.output_dir, "test_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
            
        logger.info("All tests completed successfully")
        return summary

def main():
    """Run the tests."""
    test = TestMedicalDomain()
    summary = test.run_all_tests()
    
    print("\nTest Summary:")
    print(f"Terminology Processor Tests: {summary['terminology_processor_tests']}")
    print(f"Prompt Generator Tests: {summary['prompt_generator_tests']}")
    print(f"Integration Tests: {summary['integration_tests']}")
    print(f"Status: {summary['status']}")
    print(f"\nResults saved to: {test.output_dir}/")

if __name__ == "__main__":
    main()
