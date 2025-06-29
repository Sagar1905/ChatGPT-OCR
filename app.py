import os
from flask import Flask, request, render_template, send_file, redirect, url_for
from werkzeug.utils import secure_filename
from docx import Document
import openai
from PIL import Image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Replace with your OpenAI API key or set environment variable OPENAI_API_KEY
openai.api_key = os.getenv('OPENAI_API_KEY', 'sk-...')


def extract_text_with_layout(image_path: str) -> str:
    """Use OpenAI to extract text from an image.

    This implementation calls the GPT-4 Vision API to read text and keep layout.
    Adjust the model or parameters as necessary.
    """
    with open(image_path, 'rb') as img_file:
        response = openai.ChatCompletion.create(
            model="gpt-4-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Extract the text from this document and preserve layout in markdown.",
                        },
                        {"type": "image", "image": img_file.read()}]
                }],
            max_tokens=4096,
        )
    # The content is expected to be markdown representing the layout
    return response.choices[0].message.content


def save_to_docx(markdown_text: str, output_path: str):
    """Save markdown text as a docx document."""
    doc = Document()
    for line in markdown_text.splitlines():
        doc.add_paragraph(line)
    doc.save(output_path)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        files = request.files.getlist('documents')
        doc_paths = []
        for file in files:
            if not file.filename:
                continue
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Extract text and layout using OpenAI
            markdown_text = extract_text_with_layout(file_path)
            output_docx = os.path.splitext(file_path)[0] + '.docx'
            save_to_docx(markdown_text, output_docx)
            doc_paths.append(output_docx)

        # Display only first document text for preview
        preview_text = ""
        if doc_paths:
            with open(doc_paths[0], 'rb') as doc_file:
                # Convert docx text to plain text for preview
                from docx import Document as DocReader
                doc_reader = DocReader(doc_file)
                preview_text = "\n".join([p.text for p in doc_reader.paragraphs])
        return render_template('index.html', docs=doc_paths, preview=preview_text)
    return render_template('index.html', docs=None)


@app.route('/download/<path:filename>')
def download(filename):
    return send_file(filename, as_attachment=True)


if __name__ == '__main__':
    app.run(debug=True)
