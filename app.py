from flask import Flask, request, render_template, jsonify
import PyPDF2
import os
import logging
from google.generativeai import GenerativeModel
import google.generativeai as genai
from dotenv import load_dotenv
import secrets
import re
from pdf2image import convert_from_path
import pytesseract

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

load_dotenv()

app = Flask(__name__)

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env file")
genai.configure(api_key=GEMINI_API_KEY)

model = GenerativeModel('gemini-1.5-flash')

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = {'pdf'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def sanitize_filename(filename):
    """Sanitize the filename to remove problematic characters and ensure uniqueness."""
    safe_filename = re.sub(r'[^a-zA-Z0-9._-]', '_', filename)
    random_suffix = secrets.token_hex(4)
    name, ext = os.path.splitext(safe_filename)
    return f"{name}_{random_suffix}{ext}"

def extract_pdf_content(file_path):
    """Extract text from a PDF file using PyPDF2, with OCR fallback for image-based PDFs."""
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            
            if pdf_reader.is_encrypted:
                logger.error("PDF is encrypted and cannot be processed")
                return "Error: PDF is encrypted and cannot be processed without a password"
            
            text = ''
            for page_num, page in enumerate(pdf_reader.pages, 1):
                page_text = page.extract_text() or ''
                text += page_text
                logger.debug(f"PyPDF2 - Page {page_num}: Extracted {len(page_text)} characters")
            
            if text.strip():
                logger.info(f"Successfully extracted {len(text)} characters using PyPDF2")
                return text
            else:
                logger.warning("No text extracted with PyPDF2. Falling back to OCR.")
    
    except PyPDF2.errors.PdfReadError as e:
        logger.error(f"PdfReadError during PyPDF2 extraction: {str(e)}")
        return f"Error: Failed to read PDF file. It may be corrupted or invalid: {str(e)}"
    except Exception as e:
        logger.error(f"Unexpected error during PyPDF2 extraction: {str(e)}")

    try:
        images = convert_from_path(file_path)
        logger.info(f"Converted PDF to {len(images)} images for OCR")
        
        text = ''
        for page_num, image in enumerate(images, 1):
            page_text = pytesseract.image_to_string(image)
            text += page_text
            logger.debug(f"OCR - Page {page_num}: Extracted {len(page_text)} characters")
        
        if not text.strip():
            logger.error("No text extracted from PDF using OCR. It may be empty or unreadable.")
            return "Error: No text could be extracted from the PDF. It may be empty or unreadable."
        
        logger.info(f"Successfully extracted {len(text)} characters using OCR")
        return text
    
    except Exception as e:
        logger.error(f"Error during OCR extraction: {str(e)}")
        return f"Error: Failed to extract text using OCR. It may be an issue with the PDF or OCR setup: {str(e)}"

def analyze_content_with_gemini(resume_content, job_description):
    """Analyze resume content with Gemini API."""
    try:
        if not resume_content or not resume_content.strip():
            logger.error("Resume content is empty or invalid")
            return 0, "Error: Resume content is empty or invalid", []
        
        prompt = f"""
        You are a strict and practical career advisor evaluating a resume based on a provided job description. Perform the following tasks:
        1. Assign a score from 0 to 100 based on how well the resume matches the job description, considering relevance of skills, experience, clarity, and coherence.
        2. Provide a brief explanation for the score, highlighting strengths and weaknesses.
        3. Suggest 2-3 specific tips for improving the resume to better align with the job description.

        Job Description: {job_description}
        Resume Content: {resume_content}

        Return the response in the following format:
        Score: [number]
        Explanation: [text]
        Improvement Tips:
        - [Tip 1]
        - [Tip 2]
        - [Tip 3]
        """
        response = model.generate_content(prompt)
        response_text = response.text

        score = int(response_text.split('Score: ')[1].split('\n')[0])
        explanation = response_text.split('Explanation: ')[1].split('Improvement Tips:')[0].strip()
        tips = response_text.split('Improvement Tips:')[1].strip().split('\n- ')[1:]  # Extract tips as a list
        tips = [tip.strip() for tip in tips if tip.strip()]  # Clean up tips

        logger.info(f"Gemini analysis completed. Score: {score}")
        return score, explanation, tips
    except Exception as e:
        logger.error(f"Error analyzing content with Gemini: {str(e)}")
        return 0, f"Error analyzing content with Gemini: {str(e)}", []

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files or 'job_description' not in request.form:
        logger.warning("Missing file or job description in request")
        return jsonify({'error': 'Missing file or job description'}), 400

    file = request.files['file']
    job_description = request.form['job_description']

    if file.filename == '':
        logger.warning("No file selected")
        return jsonify({'error': 'No selected file'}), 400
    if not job_description:
        logger.warning("No job description provided")
        return jsonify({'error': 'No job description provided'}), 400
    if file and allowed_file(file.filename):
        filename = sanitize_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        try:
            file.save(file_path)
            logger.info(f"File saved: {file_path}")
        except Exception as e:
            logger.error(f"Error saving file: {str(e)}")
            return jsonify({'error': f"Error saving file: {str(e)}"}), 500
        
        content = extract_pdf_content(file_path)
        if content.startswith('Error'):
            logger.error(content)
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    logger.info(f"File deleted: {file_path}")
            except Exception as e:
                logger.warning(f"Error deleting file {file_path}: {str(e)}")
            return jsonify({'error': content}), 400
        
        score, explanation, tips = analyze_content_with_gemini(content, job_description)
        if score == 0 and explanation.startswith('Error'):
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    logger.info(f"File deleted: {file_path}")
            except Exception as e:
                logger.warning(f"Error deleting file {file_path}: {str(e)}")
            return jsonify({'error': explanation}), 400
        
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"File deleted: {file_path}")
            else:
                logger.warning(f"File not found for deletion: {file_path}")
        except Exception as e:
            logger.warning(f"Error deleting file {file_path}: {str(e)}")
        
        return jsonify({
            'content': content[:1000],  
            'score': score,
            'explanation': explanation,
            'improvement_tips': tips
        })
    logger.warning("Invalid file type")
    return jsonify({'error': 'Invalid file type'}), 400

if __name__ == '__main__':
    app.run(debug=True)