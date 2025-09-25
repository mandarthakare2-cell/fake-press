
# app.py - Flask backend for Fake Prescription Detector
import os
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
from PIL import Image
from model_utils import predict_image
from ocr_utils import run_ocr_improved

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
ALLOWED_EXT = {'png','jpg','jpeg','webp'}

app = Flask(__name__, static_folder='../frontend', static_url_path='/')
CORS(app)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 12 * 1024 * 1024  # 12 MB

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.',1)[1].lower() in ALLOWED_EXT

@app.route('/')
def serve_frontend():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error':'no file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error':'no filename'}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(path)
        try:
            pil_img = Image.open(path).convert('RGB')
        except Exception as e:
            return jsonify({'error': 'cannot open image', 'details': str(e)}), 400

        cnn_res = predict_image(pil_img)
        ocr_text, ocr_details = run_ocr_improved(pil_img)

        result = {
            'cnn': cnn_res,
            'ocr_text': ocr_text,
            'ocr_details': ocr_details
        }
        return jsonify(result)
    else:
        return jsonify({'error':'invalid file type'}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)
