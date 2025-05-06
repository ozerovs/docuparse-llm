# Donut Model Local Implementation

This repository contains a simple implementation of the [Donut model](https://huggingface.co/docs/transformers/main/en/model_doc/donut) from Hugging Face for document understanding tasks, with both a command-line interface and a REST API.

## About Donut

Donut (Document Understanding Transformer) is a model designed for document understanding tasks such as:
- Document classification
- Document parsing (extracting structured information)
- Visual question answering on documents

The model uses a vision encoder (Swin Transformer) and a text decoder (BART) to process document images and generate text outputs.

## Setup

1. Clone this repository:
```bash
git clone <repository-url>
cd <repository-directory>
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

This will install:
- transformers (for the Donut model)
- torch (PyTorch backend)
- pillow (for image processing)
- datasets (for handling datasets if needed)
- sentencepiece (for tokenization)
- requests (for downloading images from URLs)
- fastapi (for the REST API)
- uvicorn (ASGI server for running the API)
- python-multipart (for handling file uploads)
- pymupdf (for processing PDF documents)

## Usage

### Running the Example

The main.py file contains a complete example that:
1. Loads the Donut model and processor
2. Downloads a sample receipt image
3. Processes the image to extract structured information
4. Prints the extracted information

To run the example:
```bash
python main.py
```

### Using with Your Own Documents

To use the model with your own document images:

1. Uncomment and modify the following lines in main.py:
```python
# local_image_path = "path/to/your/document.jpg"
# result = process_document(local_image_path, model, processor, device)
# print(result)
```

2. Replace "path/to/your/document.jpg" with the path to your document image.

3. Run the script:
```bash
python main.py
```

## Customizing the Model

The default implementation uses the "naver-clova-ix/donut-base-finetuned-cord-v2" model, which is fine-tuned for receipt parsing. If you want to use a different Donut model variant:

1. Change the model and processor initialization in the `load_model_and_processor()` function:
```python
processor = DonutProcessor.from_pretrained("your-preferred-model")
model = VisionEncoderDecoderModel.from_pretrained("your-preferred-model")
```

2. Update the task prompt in the `process_document()` function call to match your chosen model.

## Available Donut Models

Some available Donut models include:
- "naver-clova-ix/donut-base-finetuned-cord-v2" (for receipt parsing)
- "naver-clova-ix/donut-base-finetuned-docvqa" (for document visual QA)
- "naver-clova-ix/donut-base" (base model)

## API Usage

The repository includes a REST API built with FastAPI that allows you to process documents via HTTP requests.

### Running the API Server

To start the API server:

```bash
python api.py
```

This will start the server at http://localhost:8000.

Alternatively, you can use uvicorn directly:

```bash
uvicorn api:app --reload
```

### API Documentation

Once the server is running, you can access the interactive API documentation at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

These interfaces provide detailed information about the available endpoints, request parameters, and response formats.

### Making API Requests

To parse a document using the API, send a POST request to the `/parse` endpoint with the document file:

```bash
curl -X 'POST' \
  'http://localhost:8000/parse' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@/path/to/your/document.jpg'
```

You can also use tools like Postman or the interactive documentation to make requests.

### API Response Format

The API returns a JSON response with the following structure:

```json
{
  "document_type": "receipt",
  "currency": "USD",
  "language": "en",
  "total_amount": 33.21,
  "fields": [
    {
      "name": "Item 1",
      "value": "$10.00"
    },
    {
      "name": "Item 2",
      "value": "$15.50"
    },
    {
      "name": "Item 3",
      "value": "$5.25"
    },
    {
      "name": "Subtotal",
      "value": "$30.75"
    },
    {
      "name": "Tax",
      "value": "$2.46"
    },
    {
      "name": "Total",
      "value": "$33.21"
    }
  ],
  "raw_text": "..."
}
```

### Supported File Formats

The API supports the following file formats:
- Images: JPG, JPEG, PNG, BMP, TIFF
- Documents: PDF

## Resources

- [Hugging Face Donut Documentation](https://huggingface.co/docs/transformers/main/en/model_doc/donut)
- [Donut Paper](https://arxiv.org/abs/2111.15664)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
