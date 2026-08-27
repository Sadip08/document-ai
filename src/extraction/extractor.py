import re

def extract_information(doc_type, full_text):
    """
    Extracts semantic fields based on the predicted document type.
    Uses deterministic regex patterns as required by Phase 9.
    """
    extracted_data = {"document_type": doc_type}
    
    # Combine the text into a single string for easier regex searching
    text = " ".join(full_text)
    
    if doc_type == "invoice":
        # Invoice Number: (INV-1234 or Invoice #1234)
        inv_match = re.search(r'(?:INV[-#]?|Invoice\s*(?:No|Number|#)\s*[:\-]?\s*)([A-Z0-9\-]+)', text, re.IGNORECASE)
        if inv_match:
            extracted_data["invoice_number"] = inv_match.group(1)
            
        # Date: (Date: 2023-08-12 or 08/12/2023)
        date_match = re.search(r'(?:Date\s*[:\-]?\s*)(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}|\d{4}[\/\-]\d{1,2}[\/\-]\d{1,2})', text, re.IGNORECASE)
        if date_match:
            extracted_data["date"] = date_match.group(1)
            
        # Total: (Total: $1,240.00)
        total_match = re.search(r'(?:Total\s*(?:Amount)?\s*[:\-]?\s*)(\$?\d{1,3}(?:,\d{3})*(?:\.\d{2})?)', text, re.IGNORECASE)
        if total_match:
            extracted_data["total"] = total_match.group(1)

    elif doc_type == "resume":
        # Email: standard email regex
        email_match = re.search(r'[\w\.-]+@[\w\.-]+', text)
        if email_match:
            extracted_data["email"] = email_match.group(0)
            
        # Phone: (123-456-7890 or (123) 456-7890)
        phone_match = re.search(r'(\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4})', text)
        if phone_match:
            extracted_data["phone"] = phone_match.group(1)
            
    elif doc_type == "paper":
        # ArXiv ID or DOI
        arxiv_match = re.search(r'(arXiv:\d{4}\.\d{4})', text, re.IGNORECASE)
        if arxiv_match:
            extracted_data["arxiv_id"] = arxiv_match.group(1)
            
    return extracted_data