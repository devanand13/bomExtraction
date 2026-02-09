"""
BOM Extraction Script
Works with PyPDF2 and OpenAI - no special libraries needed
"""

import os
import PyPDF2
import openai
import json
import pandas as pd
from typing import Dict, Any
from dotenv import load_dotenv

load_dotenv()


class BOMExtractor:
    """Extract structured BOM data from PDFs using LLMs"""
    
    def __init__(self, model: str = "gpt-4o-mini"):
        self.api_key = os.getenv('OPENAI_API_KEY')
        
        if not self.api_key:
            raise ValueError("API key not found. Set OPENAI_API_KEY in .env file")
        
        self.client = openai.OpenAI(api_key=self.api_key)
        self.model = model
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract all text from PDF"""
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page_num, page in enumerate(pdf_reader.pages, 1):
                text += f"\n--- Page {page_num} ---\n"
                text += page.extract_text()
        return text
    
    def extract_bom_data(self, pdf_path: str, bom_type: str = "engineering") -> Dict[str, Any]:
        """
        Extract BOM data from PDF
        
        Args:
            pdf_path: Path to PDF file
            bom_type: Type of BOM - "engineering" or "simple"
        
        Returns:
            Dictionary with extracted BOM items
        """
        # Extract text
        text = self.extract_text_from_pdf(pdf_path)
        
        # Define schema based on BOM type
        if bom_type == "engineering":
            schema = {
                "item_number": "Line item number (e.g., '1', '2')",
                "quantity": "Quantity needed",
                "substitution_code": "Substitution code (S column, e.g., 6, 10)",
                "manufacturer": "Manufacturer name",
                "part_number": "Manufacturer part number",
                "description": "Component description",
                "reference_designator": "Reference designator (REF column, e.g., 'C1, C2', 'U1')",
                "package": "Package type if specified (e.g., '0603', 'SOIC8')"
            }
        else:  # simple
            schema = {
                "category": "Category (e.g., STRUCTURE, ELECTRONICS, OTHER)",
                "where": "Source/location",
                "item": "Item description",
                "quantity": "Quantity",
                "unit_price": "Unit price",
                "total": "Total cost"
            }
        
        # Create prompt
        prompt = f"""
Extract ALL BOM (Bill of Materials) line items from this document.

Document contains a BOM table with these expected fields:
{json.dumps(schema, indent=2)}

CRITICAL INSTRUCTIONS:
1. Extract EVERY line item - do not skip any rows
2. Preserve exact values from the document
3. If a field is empty or not present, use null
4. Return ONLY valid JSON with no markdown formatting
5. For engineering BOMs: Pay attention to substitution codes and reference designators
6. For cost BOMs: Ensure calculations match (quantity × unit_price = total)

Return format:
{{
  "document_title": "extracted title if present",
  "bom_type": "{bom_type}",
  "total_items": <count>,
  "items": [
    {{"field1": "value1", "field2": "value2", ...}},
    ...
  ]
}}

Document text:
{text}
"""
        
        # Call LLM
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a precise BOM data extraction expert. Extract data exactly as it appears."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            response_format={"type": "json_object"}
        )
        
        # Parse response
        result = json.loads(response.choices[0].message.content)
        return result
    
    def save_to_csv(self, bom_data: Dict[str, Any], output_path: str):
        """Save BOM data to CSV"""
        df = pd.DataFrame(bom_data['items'])
        df.to_csv(output_path, index=False)
        print(f"✅ Saved {len(df)} items to {output_path}")
    
    def save_to_json(self, bom_data: Dict[str, Any], output_path: str):
        """Save BOM data to JSON"""
        with open(output_path, 'w') as f:
            json.dump(bom_data, f, indent=2)
        print(f"✅ Saved to {output_path}")
    
    def print_summary(self, bom_data: Dict[str, Any]):
        """Print BOM summary"""
        print(f"\n{'='*60}")
        print(f"📄 Document: {bom_data.get('document_title', 'Unknown')}")
        print(f"📦 BOM Type: {bom_data.get('bom_type', 'Unknown')}")
        print(f"📊 Total Items: {bom_data.get('total_items', len(bom_data['items']))}")
        print(f"{'='*60}\n")
        
        # Show first few items
        df = pd.DataFrame(bom_data['items'])
        print(df.head(10))
        print(f"\n... ({len(df)} total items)")


def main():
    """Example usage"""
    
    # Setup
    extractor = BOMExtractor()
    
    # Process Sample BOM 3 (complex engineering BOM)
    print("\n🔧 Processing Sample BOM 3 (Engineering BOM)...")
    bom3 = extractor.extract_bom_data(
        pdf_path="Sample_BOM_3.pdf",
        bom_type="engineering"
    )
    extractor.print_summary(bom3)
    extractor.save_to_csv(bom3, "extracted_bom_3.csv")
    extractor.save_to_json(bom3, "extracted_bom_3.json")
    
    # Process Sample BOM 4 (simple cost BOM)
    print("\n💰 Processing Sample BOM 4 (Cost BOM)...")
    bom4 = extractor.extract_bom_data(
        pdf_path="Sample_BOM_4.pdf",
        bom_type="simple"
    )
    extractor.print_summary(bom4)
    extractor.save_to_csv(bom4, "extracted_bom_4.csv")
    extractor.save_to_json(bom4, "extracted_bom_4.json")


if __name__ == "__main__":
    main()