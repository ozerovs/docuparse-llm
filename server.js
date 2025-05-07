const express = require('express');
const axios = require('axios');
const cors = require('cors');
const multer = require('multer');
const pdfParse = require('pdf-parse');
const { createWorker } = require('tesseract.js');
const franc = require('franc');
const fs = require('fs');
const path = require('path');
const { fileTypeFromBuffer } = require('file-type');

const app = express();
const PORT = 3000;

// Set up multer for file uploads
const upload = multer({ 
  dest: 'uploads/',
  limits: { fileSize: 10 * 1024 * 1024 } // 10MB limit
});

// Create uploads directory if it doesn't exist
if (!fs.existsSync('uploads')) {
  fs.mkdirSync('uploads');
}

app.use(cors());
app.use(express.json());

app.post('/ocr-parse', async (req, res) => {
  const { ocrText } = req.body;

  if (!ocrText) {
    return res.status(400).json({ error: 'Missing ocrText in request body.' });
  }

  console.log('ocrText:', ocrText);

  const prompt = `
This is OCR output from a scanned document or receipt:
""" 
${ocrText}
"""
Extract structured data in JSON format. Include: date, merchant name, list of items (with name, quantity, price), and total amount.`;

  try {
    const response = await axios.post('http://localhost:11434/api/generate', {
      model: 'mistral',
      prompt,
      stream: false
    });

    const result = response.data.response;
    return res.json({ data: result });
  } catch (err) {
    console.error('Error querying Ollama:', err.message);
    return res.status(500).json({ error: 'Failed to get response from Ollama.' });
  }
});

// Function to extract text from PDF
async function extractTextFromPDF(filePath) {
  try {
    const dataBuffer = fs.readFileSync(filePath);
    const data = await pdfParse(dataBuffer);
    return data.text;
  } catch (error) {
    console.error('Error extracting text from PDF:', error);
    throw new Error('Failed to extract text from PDF');
  }
}

// Function to perform OCR on image
async function performOCR(filePath) {
  try {
    const worker = await createWorker();
    await worker.loadLanguage('eng');
    await worker.initialize('eng');
    const { data } = await worker.recognize(filePath);
    await worker.terminate();
    return data.text;
  } catch (error) {
    console.error('Error performing OCR:', error);
    throw new Error('Failed to perform OCR on image');
  }
}

// Function to detect language
function detectLanguage(text) {
  console.log(text, 'le text')
  try {
    const langCode = franc(text);
    return langCode !== 'und' ? langCode : 'unknown';
  } catch (error) {
    console.error('Error detecting language:', error);
    return 'unknown';
  }
}

// New endpoint for file upload and processing
app.post('/process-document', upload.single('file'), async (req, res) => {
  if (!req.file) {
    return res.status(400).json({ error: 'No file uploaded' });
  }

  try {
    const filePath = req.file.path;
    const fileBuffer = fs.readFileSync(filePath);
    const fileType = await fileTypeFromBuffer(fileBuffer);

    let extractedText = '';
    let fileTypeInfo = 'unknown';

    // Determine file type and extract text
    if (fileType) {
      if (fileType.mime === 'application/pdf') {
        extractedText = await extractTextFromPDF(filePath);
        fileTypeInfo = 'pdf';
      } else if (fileType.mime.startsWith('image/')) {
        extractedText = await performOCR(filePath);
        fileTypeInfo = 'image';
      } else {
        return res.status(400).json({ error: 'Unsupported file type. Please upload a PDF or image.' });
      }
    } else {
      // Try to determine file type by extension
      const ext = path.extname(req.file.originalname).toLowerCase();
      if (ext === '.pdf') {
        extractedText = await extractTextFromPDF(filePath);
        fileTypeInfo = 'pdf';
      } else if (['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff'].includes(ext)) {
        extractedText = await performOCR(filePath);
        fileTypeInfo = 'image';
      } else {
        return res.status(400).json({ error: 'Unsupported file type. Please upload a PDF or image.' });
      }
    }

    // Check if text was extracted
    if (!extractedText || extractedText.trim() === '') {
      return res.status(400).json({ error: 'No text found in the document' });
    }

    // Detect language
    const language = detectLanguage(extractedText);

    // Process with LLM
    const prompt = `
This is text extracted from a ${fileTypeInfo} document in ${language} language:
""" 
${extractedText}
"""
Extract structured data in JSON format. Include relevant fields based on the document type.`;

    const response = await axios.post('http://localhost:11434/api/generate', {
      model: 'mistral',
      prompt,
      stream: false
    });

    const result = response.data.response;

    // Clean up the uploaded file
    fs.unlinkSync(filePath);

    return res.json({ 
      data: result,
      metadata: {
        fileType: fileTypeInfo,
        language: language,
        textLength: extractedText.length
      }
    });
  } catch (err) {
    console.error('Error processing document:', err);
    return res.status(500).json({ error: 'Failed to process document' });
  }
});

app.listen(PORT, () => {
  console.log(`✅ Server is running on http://localhost:${PORT}`);
});
