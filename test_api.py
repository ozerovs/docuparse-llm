"""
Test Script for Document Parser API

This script demonstrates how to test the Document Parser API by sending
requests with different document types and verifying the responses.
"""

import requests
import json
import os
import time

# API endpoint URL (update if your server runs on a different host/port)
API_URL = "http://localhost:8000/parse"

def test_image_document(image_path):
    """Test the API with an image document."""
    print(f"\n=== Testing with image: {image_path} ===")
    
    # Check if the file exists
    if not os.path.exists(image_path):
        print(f"Error: File {image_path} not found")
        return
    
    # Prepare the file for upload
    with open(image_path, "rb") as file:
        files = {"file": (os.path.basename(image_path), file, "image/png")}
        
        # Send the request
        print("Sending request...")
        response = requests.post(API_URL, files=files)
        
        # Process the response
        if response.status_code == 200:
            print("Request successful!")
            result = response.json()
            print_formatted_result(result)
        else:
            print(f"Error: {response.status_code}")
            print(response.text)

def test_pdf_document(pdf_path):
    """Test the API with a PDF document."""
    print(f"\n=== Testing with PDF: {pdf_path} ===")
    
    # Check if the file exists
    if not os.path.exists(pdf_path):
        print(f"Error: File {pdf_path} not found")
        return
    
    # Prepare the file for upload
    with open(pdf_path, "rb") as file:
        files = {"file": (os.path.basename(pdf_path), file, "application/pdf")}
        
        # Send the request
        print("Sending request...")
        response = requests.post(API_URL, files=files)
        
        # Process the response
        if response.status_code == 200:
            print("Request successful!")
            result = response.json()
            print_formatted_result(result)
        else:
            print(f"Error: {response.status_code}")
            print(response.text)

def print_formatted_result(result):
    """Print the API response in a formatted way."""
    print("\nParsed Document Information:")
    print(f"Document Type: {result.get('document_type', 'N/A')}")
    print(f"Currency: {result.get('currency', 'N/A')}")
    print(f"Language: {result.get('language', 'N/A')}")
    print(f"Total Amount: {result.get('total_amount', 'N/A')}")
    
    print("\nExtracted Fields:")
    for field in result.get('fields', []):
        print(f"  {field.get('name', 'N/A')}: {field.get('value', 'N/A')}")
    
    print("\nRaw Text (first 100 chars):")
    raw_text = result.get('raw_text', '')
    print(f"  {raw_text[:100]}..." if len(raw_text) > 100 else f"  {raw_text}")

def main():
    """Main function to run the tests."""
    print("=== Document Parser API Test ===")
    
    # Wait for the API server to be ready
    print("Checking if API server is running...")
    max_retries = 5
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            # Try to connect to the API server
            response = requests.get("http://localhost:8000/docs")
            if response.status_code == 200:
                print("API server is running!")
                break
        except requests.exceptions.ConnectionError:
            print(f"API server not ready. Retrying in 2 seconds... ({retry_count + 1}/{max_retries})")
            time.sleep(2)
            retry_count += 1
    
    if retry_count == max_retries:
        print("Error: Could not connect to the API server. Make sure it's running with 'python api.py'")
        return
    
    # Test with the sample receipt image
    test_image_document("test_receipt.png")
    
    # If you have a PDF document, uncomment the following line and provide the path
    # test_pdf_document("path/to/your/document.pdf")
    
    print("\n=== Test Completed ===")

if __name__ == "__main__":
    main()