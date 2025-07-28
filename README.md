# PDF Heading Extraction with BERT

## Overview

This project implements a BERT-based solution for extracting structured headings from PDF documents. The model is trained to classify text lines as Title, H1, H2, H3, or Other, enabling automatic generation of document outlines.

## Approach

### Model Architecture
- **Base Model**: BERT (bert-base-uncased) fine-tuned for sequence classification
- **Input Format**: Text lines with embedded features as special tokens
- **Features Used**: Font size, bold formatting, page number, x/y coordinates
- **Output**: 5-class classification (title, h1, h2, h3, other)

### Training Data
- Dataset: 100 PDFs with manually labeled text lines
- Features: Text content, font size, bold formatting, position (x,y), page number
- Class Distribution: Balanced dataset with downsampled "other" class
- Input Format: `[FONTSIZE=18] [BOLD=1] [PAGE=1] [X=100] [Y=50] Heading Text`

### Model Performance
- Successfully extracts titles and headings from various PDF formats
- Handles class imbalance through weighted loss function
- Processes PDFs using PyMuPDF for robust text extraction

## Dependencies

### Core Libraries
- **torch**: PyTorch for deep learning
- **transformers**: HuggingFace Transformers for BERT
- **pymupdf**: PDF text extraction
- **scikit-learn**: Class weight computation
- **pandas**: Data manipulation
- **tqdm**: Progress bars

### System Requirements
- CPU: AMD64 architecture
- Memory: 16GB RAM (as per hackathon constraints)
- Model Size: <200MB (BERT base model)
- Runtime: ≤10 seconds per 50-page PDF

## Docker Setup

### Build the Image
```bash
docker build --platform linux/amd64 -t heading-extractor:latest .
```

### Run the Container
```bash
docker run --rm \
  -v $(pwd)/input:/app/input \
  -v $(pwd)/output:/app/output \
  --network none \
  heading-extractor:latest
```

### Input/Output
- **Input**: Place PDF files in the `input/` directory
- **Output**: JSON files are generated in the `output/` directory
- **Format**: Each PDF generates a corresponding `filename.json`

## JSON Output Format

```json
{
  "title": "Document Title",
  "outline": [
    {
      "level": "H1",
      "text": "Main Heading",
      "page": 1
    },
    {
      "level": "H2", 
      "text": "Sub Heading",
      "page": 2
    }
  ]
}
```

## Model Training

### Data Preparation
```bash
python prepare_bert_data.py
```

### Training
```bash
python train_bert_heading.py
```

### Model Files
- Trained model: `saved/bert_heading_model/`
- Tokenizer: `saved/bert_heading_model/`
- Training data: `bert_training_data.csv`

## Performance

### Constraints Compliance
- ✅ CPU-only execution (no GPU required)
- ✅ Model size <200MB
- ✅ Offline operation (no internet calls)
- ✅ AMD64 architecture support
- ✅ ≤10 seconds processing time for 50-page PDFs

### Accuracy
- Successfully extracts titles and headings from various PDF formats
- Handles different document layouts and formatting styles
- Robust to variations in font sizes and positioning

## Usage Examples

### Process Single PDF
```bash
python bert_inference.py path/to/document.pdf
```

### Process Multiple PDFs (Docker)
```bash
# Copy PDFs to input directory
cp *.pdf input/

# Run container
docker run --rm -v $(pwd)/input:/app/input -v $(pwd)/output:/app/output heading-extractor:latest

# Check results
ls output/
```

## Technical Details

### Feature Engineering
- Font size, bold formatting, and position information are embedded as special tokens
- BERT learns to use these visual cues for heading classification
- Default values used for inference when features are unavailable

### Text Extraction
- Uses PyMuPDF for robust PDF text extraction
- Preserves page information and text positioning
- Handles various PDF formats and layouts

### Model Architecture
- BERT base model (110M parameters)
- Fine-tuned classification head for 5 classes
- Weighted loss function to handle class imbalance

## Limitations

- Requires training data with manual annotations
- Performance depends on PDF quality and formatting consistency
- May struggle with complex layouts or non-standard formatting

## Future Improvements

- Add support for more heading levels (H4, H5, etc.)
- Implement confidence scoring for predictions
- Add post-processing rules for cleaner output
- Support for multilingual documents 