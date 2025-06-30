# ChatGPT OCR App

This project provides a simple web interface for extracting text from scanned
images using the OpenAI API. Uploaded images are converted to text while
attempting to retain layout, then saved as `.docx` files which can be
downloaded from the browser.

## Features

- Upload single or multiple image files
- Text extraction powered by OpenAI's vision model
- Advanced multi-strategy extraction with fallback mechanisms
- Professional Word document-like interface
- Basic layout retention using Markdown to DOCX conversion
- Preview of extracted documents in the browser
- High-accuracy mode with 2-pass verification
- Automatic document type detection (tables, forms, receipts, documents)

## Quick Setup for Windows Users 🚀

**Option 1: Automated Batch Script (Recommended)**
1. Download all project files to a folder
2. Double-click `setup_and_run_windows.bat`
3. Follow the prompts to enter your OpenAI API key
4. The application will automatically open in your browser!

**Option 2: PowerShell Script (Advanced)**
1. Right-click and "Run with PowerShell" on `setup_and_run_windows.ps1`
2. Or open PowerShell and run: `.\setup_and_run_windows.ps1`
3. Use `.\setup_and_run_windows.ps1 -Help` for more options

## Manual Setup (All Platforms)

### Prerequisites
- Python 3.7 or higher
- OpenAI API key (get one from [OpenAI Platform](https://platform.openai.com/api-keys))

### Installation Steps

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set your OpenAI API key:**
   
   Create a `.env` file in the project directory:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```
   
   Or set as environment variable:
   ```bash
   # Windows
   set OPENAI_API_KEY=your_api_key_here
   
   # macOS/Linux
   export OPENAI_API_KEY=your_api_key_here
   ```

3. **Start the Flask server:**
   ```bash
   python app.py
   ```

4. **Open your browser to `http://localhost:5000` and upload your documents.**

## Usage Tips

- **High-resolution images** (300 DPI or higher) work best
- **Good lighting** and minimal shadows improve accuracy
- **Flat, properly aligned documents** give better results
- **Enable High Accuracy Mode** for legal/financial documents
- **Multiple extraction strategies** automatically handle difficult images

## Troubleshooting

### Windows Users
- If the batch script fails, try running as Administrator
- Ensure Python is installed and added to PATH
- Check that your OpenAI API key is valid

### All Platforms
- Make sure you have a stable internet connection
- Verify your OpenAI API key has sufficient credits
- Check the console output for detailed error messages

## Technical Details

The application uses:
- **Multi-strategy extraction**: 6 different approaches with automatic fallbacks
- **Document type detection**: Specialized handling for tables, forms, receipts
- **Image preprocessing**: OpenCV-based enhancement for better OCR
- **Smart failure detection**: Distinguishes between refusals and actual content
- **Professional UI**: Modern, responsive design with drag-and-drop

## Notes

The layout retention relies on GPT-4 Vision producing structured text with layout
information. The system automatically tries multiple extraction strategies to
handle challenging documents that might initially fail.
