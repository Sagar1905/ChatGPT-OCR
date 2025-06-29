# ChatGPT OCR App

This project provides a simple web interface for extracting text from scanned
images using the OpenAI API. Uploaded images are converted to text while
attempting to retain layout, then saved as `.docx` files which can be
downloaded from the browser.

## Features

- Upload single or multiple image files
- Text extraction powered by OpenAI's vision model
- Basic layout retention using Markdown to DOCX conversion
- Preview of the first extracted document in the browser

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Set your OpenAI API key in the `OPENAI_API_KEY` environment variable.
3. Start the Flask server:
   ```bash
   python app.py
   ```
4. Open your browser to `http://localhost:5000` and upload your documents.

## Notes

The layout retention relies on GPT-4 Vision producing Markdown with layout
information. You may need to experiment with prompts for optimal results.
