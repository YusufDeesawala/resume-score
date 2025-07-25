from flask import Flask, request, render_template, jsonify
import PyPDF2
import os
from google.generativeai import GenerativeModel
import google.generativeai as genai
from dotenv import load_dotenv

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

def extract_pdf_content(file_path):
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ''
            for page in pdf_reader.pages:
                text += page.extract_text() or ''
            return text
    except Exception as e:
        return f"Error extracting PDF content: {str(e)}"

def analyze_content_with_gemini(resume_content, job_description):
    try:
        prompt = f"""
        You are a career advisor evaluating a resume based on a provided job description. Perform the following tasks:
        1. Assign a score from 0 to 100 based on how well the resume matches the job description, considering relevance of skills, experience, clarity, and coherence.
        2. Provide a brief explanation for the score, highlighting strengths and weaknesses.
        3. Suggest 2-3 specific tips for improving the resume to better align with the job description.

        Job Description: {job_description[:2000]}
        Resume Content: {resume_content[:2000]}

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

        # Parse the response
        score = int(response_text.split('Score: ')[1].split('\n')[0])
        explanation = response_text.split('Explanation: ')[1].split('Improvement Tips:')[0].strip()
        tips = response_text.split('Improvement Tips:')[1].strip().split('\n- ')[1:]  # Extract tips as a list
        tips = [tip.strip() for tip in tips if tip.strip()]  # Clean up tips

        return score, explanation, tips
    except Exception as e:
        return 0, f"Error analyzing content with Gemini: {str(e)}", []

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files or 'job_description' not in request.form:
        return jsonify({'error': 'Missing file or job description'}), 400

    file = request.files['file']
    job_description = request.form['job_description']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if not job_description:
        return jsonify({'error': 'No job description provided'}), 400
    if file and allowed_file(file.filename):
        filename = file.filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        content = extract_pdf_content(file_path)
        if content.startswith('Error'):
            return jsonify({'error': content}), 500
        
        score, explanation, tips = analyze_content_with_gemini(content, job_description)
        
        os.remove(file_path)
        
        return jsonify({
            'content': content[:1000],  # Limit content for response size
            'score': score,
            'explanation': explanation,
            'improvement_tips': tips
        })
    return jsonify({'error': 'Invalid file type'}), 400

if __name__ == '__main__':
    app.run(debug=True)