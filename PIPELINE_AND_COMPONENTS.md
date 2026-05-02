# Pipeline Report

## Abstract
This report describes the full DocSearch pipeline from file discovery to evaluation, explaining the tools used at each stage and how data flows through the system.

## Input sources
- Local directories selected by the user.
- Supported files: PDF, DOCX, XLSX, text, and images.

## Step 1: File discovery
The system scans the selected directory recursively and filters files by extension. This ensures only supported formats proceed to parsing.

## Step 2: Parsing and OCR
### PDFs
PDF pages are rendered into images. If extracted text is small, OCR runs on the rendered page. This allows scanned PDFs to be indexed.

### Images
Images are preprocessed (grayscale, contrast, sharpen) and sent to Tesseract OCR to extract text.

### DOCX and Excel
Structured document libraries extract text from Word and spreadsheet files.

### Text files
Plain text is read with encoding detection and normalized.

## Step 3: Filename enrichment
To handle coverage gaps, normalized filename tokens and mapped tags are appended to extracted text. This raises recall for documents where key terms exist only in filenames.

## Step 4: Preprocessing
Tokens are normalized for English, Hindi, and Telugu scripts. Stopwords and length filters reduce noise.

## Step 5: Indexing
Three indices are built:
1) Inverted index for positional term search.
2) TF-IDF for keyword relevance.
3) FAISS vector index for semantic similarity.

## Step 6: Search and ranking
The search stage supports keyword, semantic, and hybrid modes. Hybrid search fuses keyword and semantic scores. Optional re-ranking refines top candidates.

## Step 7: Storage
Index artifacts are persisted locally to allow fast reuse without reprocessing.

## Step 8: Evaluation
A fixed query set with ground-truth relevant files is used to compute Precision, Recall, F1, MRR, and MAP. Filename normalization prevents evaluation mismatches caused by formatting differences.

## How to run the pipeline
Indexing:
```powershell
python -m src.main --index IR_DOCUMNETS
```

Evaluation:
```powershell
python -m src.main --evaluate
```
