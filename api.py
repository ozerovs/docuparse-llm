"""
FastAPI Application for Document Parsing with Donut

This module provides a REST API for document parsing using the Donut model.
It allows users to upload document files (images or PDFs) and receive
structured information extracted from those documents.
"""

import os
import io
import json
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from PIL import Image
import fitz  # PyMuPDF for PDF processing (imported from pymupdf package)
from transformers import DonutProcessor, VisionEncoderDecoderModel

# Import the document processing function from main.py
from main import load_model_and_processor, process_document

# Create FastAPI app
app = FastAPI(
    title="Document Parser API",
    description="API for parsing documents using the Donut model",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and processor at startup
model = None
processor = None
device = None

@app.on_event("startup")
async def startup_event():
    """Load the Donut model and processor when the API starts."""
    global model, processor, device
    print("Loading Donut model and processor...")
    model, processor, device = load_model_and_processor()
    print("Model loaded successfully!")

# Response models
class DocumentField(BaseModel):
    """Model for a field extracted from a document."""
    name: str
    value: str

class ParsedDocument(BaseModel):
    """Model for the parsed document response."""
    document_type: str
    currency: Optional[str] = None
    language: str = "en"
    total_amount: Optional[float] = None
    fields: List[DocumentField]
    raw_text: str

@app.post("/parse", response_model=ParsedDocument, 
          summary="Parse a document",
          description="Upload a document file (image or PDF) and receive structured information extracted from it.")
async def parse_document(file: UploadFile = File(...)):
    """
    Parse a document file and extract structured information.

    Args:
        file: The document file to parse (image or PDF)

    Returns:
        Structured information extracted from the document
    """
    global model, processor, device

    if not model or not processor:
        raise HTTPException(status_code=500, detail="Model not loaded. Please try again later.")

    # Check file extension
    file_ext = os.path.splitext(file.filename)[1].lower()

    try:
        # Process the file based on its type
        if file_ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']:
            # Process image file
            contents = await file.read()
            image = Image.open(io.BytesIO(contents)).convert("RGB")
            result = process_document_image(image, model, processor, device)
        elif file_ext == '.pdf':
            # Process PDF file
            contents = await file.read()
            result = process_pdf_document(contents, model, processor, device)
        else:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file format: {file_ext}. Supported formats: jpg, jpeg, png, bmp, tiff, pdf"
            )

        # Parse the result into structured data
        parsed_data = parse_result_to_structured_data(result, file_ext)
        return parsed_data

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")

def process_document_image(image, model, processor, device, task_prompt="<s_cord-v2>"):
    """Process a document image using the Donut model."""
    # Prepare the image for the model
    pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)

    # Generate text from the image
    task_prompt = processor.tokenizer.encode(task_prompt, add_special_tokens=False, return_tensors="pt").to(device)

    # Set the decoder input IDs to the task prompt
    model.config.decoder_start_token_id = int(task_prompt[0][0])

    # Generate the output
    outputs = model.generate(
        pixel_values,
        decoder_input_ids=task_prompt,
        max_length=model.config.decoder.max_position_embeddings,
        early_stopping=False,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
        use_cache=True,
        num_beams=1,
        bad_words_ids=[[processor.tokenizer.unk_token_id]],
        return_dict_in_generate=True,
    )

    # Decode the generated IDs to text
    sequence = processor.batch_decode(outputs.sequences)[0]
    sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")

    # Remove the task prompt from the generated text
    sequence = sequence.replace(processor.tokenizer.decode(task_prompt[0], skip_special_tokens=True), "")

    return sequence

def process_pdf_document(pdf_bytes, model, processor, device):
    """Process a PDF document using the Donut model."""
    # Open the PDF file
    pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")

    # Process the first page of the PDF
    # In a real application, you might want to process all pages or let the user select a page
    if pdf_document.page_count > 0:
        page = pdf_document[0]

        # Convert PDF page to image
        pix = page.get_pixmap()
        img_data = pix.tobytes("png")
        image = Image.open(io.BytesIO(img_data)).convert("RGB")

        # Process the image
        result = process_document_image(image, model, processor, device)
        return result
    else:
        raise ValueError("PDF document has no pages")

def parse_result_to_structured_data(result: str, file_ext: str) -> ParsedDocument:
    """
    Parse the raw result from the Donut model into structured data.

    Args:
        result: The raw text result from the Donut model
        file_ext: The file extension of the processed document

    Returns:
        Structured data extracted from the document
    """
    # Try to parse the result as JSON
    try:
        # The Donut model for CORD-v2 returns a JSON-like string
        # Clean up the result to make it valid JSON
        result = result.replace("'", '"').replace("<", '"<').replace(">", '>"')
        data = json.loads(result)
    except json.JSONDecodeError:
        # If the result is not valid JSON, create a simple structure
        data = {"raw_text": result}

    # Determine document type based on content and file extension
    document_type = determine_document_type(data, file_ext)

    # Extract currency and total amount
    currency, total_amount = extract_currency_and_total(data)

    # Extract fields
    fields = extract_fields(data)

    return ParsedDocument(
        document_type=document_type,
        currency=currency,
        language="en",  # Default to English
        total_amount=total_amount,
        fields=fields,
        raw_text=result
    )

def determine_document_type(data: Dict[str, Any], file_ext: str) -> str:
    """
    Determine the type of document based on its content and file extension.

    Args:
        data: The parsed data from the document
        file_ext: The file extension of the document

    Returns:
        The determined document type
    """
    # Check for receipt-specific keywords
    if isinstance(data, dict) and "menu" in data:
        return "receipt"

    # Check for invoice-specific keywords
    if isinstance(data, dict) and any(key in data for key in ["invoice", "bill"]):
        return "invoice"

    # Default to generic document type based on file extension
    if file_ext == '.pdf':
        return "pdf_document"
    else:
        return "image_document"

def extract_currency_and_total(data: Dict[str, Any]) -> tuple:
    """
    Extract currency and total amount from the parsed data.

    Args:
        data: The parsed data from the document

    Returns:
        A tuple containing the currency and total amount
    """
    currency = None
    total_amount = None

    # Try to extract from CORD-v2 format
    if isinstance(data, dict) and "menu" in data:
        # Look for total in the menu items
        for item in data.get("menu", []):
            if "total" in item.get("nm", "").lower():
                # Extract currency and amount
                price = item.get("price", "")
                if price:
                    # Extract currency symbol
                    if price.startswith("$"):
                        currency = "USD"
                    elif price.startswith("€"):
                        currency = "EUR"
                    elif price.startswith("£"):
                        currency = "GBP"
                    elif price.startswith("¥"):
                        currency = "JPY"

                    # Extract amount
                    try:
                        # Remove currency symbol and convert to float
                        amount_str = price.replace("$", "").replace("€", "").replace("£", "").replace("¥", "").strip()
                        total_amount = float(amount_str)
                    except ValueError:
                        pass

    return currency, total_amount

def extract_fields(data: Dict[str, Any]) -> List[DocumentField]:
    """
    Extract fields from the parsed data.

    Args:
        data: The parsed data from the document

    Returns:
        A list of DocumentField objects
    """
    fields = []

    # Extract fields from CORD-v2 format
    if isinstance(data, dict):
        # Add menu items
        if "menu" in data:
            for item in data.get("menu", []):
                name = item.get("nm", "")
                price = item.get("price", "")
                if name and price:
                    fields.append(DocumentField(name=name, value=price))

        # Add other fields
        for key, value in data.items():
            if key not in ["menu", "raw_text"] and isinstance(value, str):
                fields.append(DocumentField(name=key, value=value))

    return fields

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
