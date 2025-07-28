import sys
import torch
import json
from tqdm import tqdm
import fitz  # PyMuPDF
from pathlib import Path
from transformers import BertTokenizer, BertForSequenceClassification
import re
import os

# CONFIG
MAX_PAGES = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABELS = ['title', 'h1', 'h2', 'h3', 'other']

# Load trained model and tokenizer
print("Loading BERT model...")
model = BertForSequenceClassification.from_pretrained('Priyanshu958/bert-heading-extractor')
tokenizer = BertTokenizer.from_pretrained('Priyanshu958/bert-heading-extractor')
model.eval()
model.to(DEVICE)

def extract_pdf_lines(pdf_path):
    """Extract lines from PDF using PyMuPDF, preserving page and line info"""
    doc = fitz.open(str(pdf_path))
    page_data = []
    for i, page in enumerate(doc):
        blocks = page.get_text("blocks")
        for block in blocks:
            text = block[4].strip()
            if text:
                # Optionally, you can split by lines if block contains multiple lines
                for line in text.split("\n"):
                    clean_line = line.strip()
                    if clean_line:
                        page_data.append((clean_line, i + 1))  # (text, page)
    return page_data

def preprocess_text_for_bert(text, page=1, x=0, y=0, font_size=12, is_bold=0):
    """Format text with features for BERT input"""
    return f"[FONTSIZE={font_size}] [BOLD={is_bold}] [PAGE={page}] [X={x}] [Y={y}] {text}"

def predict_label(text, page=1):
    """Predict label for a line of text"""
    # For inference, we don't have font features, so use defaults
    bert_input = preprocess_text_for_bert(text, page=page)
    
    # Tokenize
    inputs = tokenizer(
        bert_input,
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors="pt"
    )
    
    # Move to device
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    
    # Predict
    with torch.no_grad():
        outputs = model(**inputs)
        pred = torch.argmax(outputs.logits, dim=1).item()
    
    return LABELS[pred]

def extract_structure(pdf_path):
    """Extract structured headings from PDF"""
    lines = extract_pdf_lines(pdf_path)
    result = {
        "title": None,
        "outline": []
    }
    
    for text, page in tqdm(lines, desc="Processing"):
        label = predict_label(text, page)
        
        if label == "title" and not result["title"]:
            result["title"] = text
        elif label in {"h1", "h2", "h3"}:
            result["outline"].append({
                "level": label.upper(),
                "text": text,
                "page": page
            })
    
    return result

def process_pdf(pdf_path, output_dir):
    """Process a single PDF and save JSON output"""
    print(f"📄 Processing: {pdf_path.name}")
    
    try:
        structured_data = extract_structure(pdf_path)
        
        # Output JSON
        output_path = output_dir / f"{pdf_path.stem}.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(structured_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ JSON saved to: {output_path}")
        print(f"📊 Results: Title found: {structured_data['title'] is not None}, Headings found: {len(structured_data['outline'])}")
        
    except Exception as e:
        print(f"❌ Error processing {pdf_path.name}: {str(e)}")

def main():
    """Main function to process all PDFs in input directory"""
    input_dir = Path("/app/input")
    output_dir = Path("/app/output")
    
    # Ensure output directory exists
    output_dir.mkdir(exist_ok=True)
    
    # Find all PDF files in input directory
    pdf_files = list(input_dir.glob("*.pdf"))
    
    if not pdf_files:
        print("❌ No PDF files found in /app/input directory")
        return
    
    print(f"📁 Found {len(pdf_files)} PDF file(s) to process")
    
    # Process each PDF
    for pdf_file in pdf_files:
        process_pdf(pdf_file, output_dir)
        print("-" * 50)
    
    print(f"🎉 Processing complete! Check /app/output for JSON files.")

if __name__ == "__main__":
    main() 
