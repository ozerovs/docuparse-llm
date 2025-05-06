# Donut Model Local Implementation

This repository contains a simple implementation of the [Donut model](https://huggingface.co/docs/transformers/main/en/model_doc/donut) from Hugging Face for document understanding tasks.

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
- protobuf (required for the XLM-RoBERTa tokenizer used by Donut)

## Usage

### Running the Example

The main.py file contains a complete example that:
1. Loads the Donut model and processor
2. Creates a sample receipt image locally (no internet connection required)
3. Processes the image to extract structured information
4. Prints the extracted information

To run the example:
```bash
python main.py
```

The script will:
- Generate a test receipt image with sample data
- Save it as "test_receipt.png" in the current directory
- Process this image with the Donut model
- Display the extracted structured information

### Using with Your Own Documents

To use the model with your own document images:

1. Modify the main() function in main.py to use your own image:
```python
def main():
    print("Loading Donut model and processor...")
    model, processor, device = load_model_and_processor()

    # Replace this path with the path to your document
    local_image_path = "path/to/your/document.jpg"
    print(f"Processing document: {local_image_path}")

    result = process_document(local_image_path, model, processor, device)
    print("\nExtracted information:")
    print(result)
```

2. Run the script:
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

## Resources

- [Hugging Face Donut Documentation](https://huggingface.co/docs/transformers/main/en/model_doc/donut)
- [Donut Paper](https://arxiv.org/abs/2111.15664)
