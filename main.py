"""
Donut Model Implementation Example

This script demonstrates how to use the Donut model from Hugging Face for document understanding.
Donut (Document Understanding Transformer) can be used for various document processing tasks
such as document classification, document parsing, and visual question answering.

Requirements:
- Install dependencies: pip install -r requirements.txt
- Download a sample document image for testing
"""

import torch
from transformers import DonutProcessor, VisionEncoderDecoderModel
from PIL import Image
import requests
from io import BytesIO

def download_image(url):
    """Download an image from a URL."""
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Error downloading image: Status code {response.status_code}")
        print(f"Response content: {response.text[:100]}...")
        raise Exception(f"Failed to download image from {url}")

    try:
        return Image.open(BytesIO(response.content))
    except Exception as e:
        print(f"Error opening image: {e}")
        print(f"Content length: {len(response.content)} bytes")
        raise

def load_model_and_processor():
    """Load the Donut model and processor."""
    processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")
    model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")

    # Set the device to GPU if available, otherwise use CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    return model, processor, device

def process_document(image_path, model, processor, device, task_prompt="<s_cord-v2>"):
    """
    Process a document image using the Donut model.

    Args:
        image_path: Path to the image file or URL
        model: The loaded Donut model
        processor: The loaded Donut processor
        device: The device to run inference on (cuda or cpu)
        task_prompt: The prompt for the specific task

    Returns:
        The generated text from the document
    """
    # Load the image
    if image_path.startswith(('http://', 'https://')):
        image = download_image(image_path)
    else:
        image = Image.open(image_path).convert("RGB")

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
        early_stopping=False,  # Changed to False since num_beams=1
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

def create_test_image():
    """Create a simple test image with text for demonstration purposes."""
    from PIL import Image, ImageDraw, ImageFont
    import os

    # Create a white image
    img = Image.new('RGB', (800, 600), color='white')
    draw = ImageDraw.Draw(img)

    # Try to use a default font, or use the default PIL font if not available
    try:
        # Try to find a system font
        font_path = None
        system_fonts = [
            '/System/Library/Fonts/Helvetica.ttc',  # macOS
            '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',  # Linux
            'C:\\Windows\\Fonts\\arial.ttf'  # Windows
        ]
        for path in system_fonts:
            if os.path.exists(path):
                font_path = path
                break

        font = ImageFont.truetype(font_path, 24) if font_path else None
    except Exception:
        font = None  # Use default font

    # Add some text to the image
    text = [
        "SAMPLE RECEIPT",
        "------------------------",
        "Item 1          $10.00",
        "Item 2          $15.50",
        "Item 3           $5.25",
        "------------------------",
        "Subtotal        $30.75",
        "Tax (8%)         $2.46",
        "------------------------",
        "Total           $33.21",
        "------------------------",
        "Thank you for your business!"
    ]

    y_position = 50
    for line in text:
        draw.text((50, y_position), line, fill='black', font=font)
        y_position += 40

    # Save the image
    test_image_path = "test_receipt.png"
    img.save(test_image_path)
    print(f"Created test image at: {test_image_path}")

    return test_image_path

def main():
    """Main function to demonstrate the Donut model."""
    print("Loading Donut model and processor...")
    model, processor, device = load_model_and_processor()

    # Create a test image since we had issues with remote URLs
    local_image_path = create_test_image()
    print(f"Processing local test image: {local_image_path}")

    result = process_document(local_image_path, model, processor, device)
    print("\nExtracted information:")
    print(result)

if __name__ == "__main__":
    main()
