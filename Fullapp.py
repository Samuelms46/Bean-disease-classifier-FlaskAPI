from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
from PIL import Image
import torch
from torchvision import transforms, models
import io
import os
from werkzeug.utils import secure_filename
import logging

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define allowed extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Model paths 
MODEL_PATHS = {
    'mobilenet_v2': 'models/best_model.pt',
    'vit': 'models/vit_model.pt',
    'resnet18': 'models/resnet_model.pth'
}

# Class labels 
CLASS_LABELS = ['angular_leaf_spot', 'bean_rust', 'healthy']

# Load models with custom state_dict handling
def load_models():
    try:
        # Initialize models from torchvision.models
        mobilenet_v2 = models.mobilenet_v2(weights=None)
        vit = models.vit_b_16(weights=None)
        resnet18 = models.resnet18(weights=None)
        
        # Modify final layers for 3-class classification
        mobilenet_v2.classifier[1] = torch.nn.Linear(mobilenet_v2.classifier[1].in_features, 3)
        vit.heads.head = torch.nn.Linear(vit.heads.head.in_features, 3)
        resnet18.fc = torch.nn.Linear(resnet18.fc.in_features, 3)
        
        # Load weights with custom handling
        for model_name, model in [('mobilenet_v2', mobilenet_v2), ('vit', vit), ('resnet18', resnet18)]:
            if os.path.exists(MODEL_PATHS[model_name]):
                state_dict = torch.load(MODEL_PATHS[model_name], map_location=torch.device('cpu'))
                # Handle custom state_dict keys (e.g., 'model_state_dict' or DataParallel prefixes)
                if 'model_state_dict' in state_dict:
                    state_dict = state_dict['model_state_dict']
                # Remove 'module.' prefix if present (from DataParallel)
                state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
                # Filter out unexpected keys
                model_dict = model.state_dict()
                state_dict = {k: v for k, v in state_dict.items() if k in model_dict}
                model_dict.update(state_dict)
                model.load_state_dict(model_dict)
            else:
                logger.warning(f"Model file not found: {MODEL_PATHS[model_name]}. Using random weights.")
            model.eval()
        
        return {
            'mobilenet_v2': mobilenet_v2,
            'vit': vit,
            'resnet18': resnet18
        }
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        raise

# Image transformation pipeline (standard ImageNet preprocessing)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load models globally
try:
    models_dict = load_models()
except Exception as e:
    logger.error(f"Failed to initialize models: {str(e)}")
    models_dict = {}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_image(image, model):
    try:
        # Apply transformations
        img_tensor = transform(image).unsqueeze(0)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            
        return {
            'class': CLASS_LABELS[predicted_class],
            'probabilities': probabilities[0].tolist()
        }
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        return None

# HTML template embedded in Python
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Bean Leaf Classifier</title>
    <style>
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }
        body {
            font-family: Arial, sans-serif;
            background-color: #f0f5f0;
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
        }
        .container {
            width: 100%;
            max-width: 1024px;
            padding: 24px;
            background-color: #ffffff;
            border-radius: 8px;
            box-shadow: 0 10px 15px rgba(0, 0, 0, 0.1), 0 4px 6px rgba(0, 0, 0, 0.05);
        }
        h1 {
            font-size: 30px;
            font-weight: bold;
            text-align: center;
            color: #1f2937;
            margin-bottom: 16px;
        }
        .subtitle {
            text-align: center;
            color: #4b5563;
            margin-bottom: 32px;
            font-size: 16px;
        }
        .dropzone {
            border: 2px dashed #4b8a3e;
            background-color: #e6efe6;
            padding: 32px;
            border-radius: 8px;
            text-align: center;
            margin-bottom: 16px;
            transition: all 0.3s ease;
        }
        .dropzone.dragover {
            background-color: #c6e2c0;
            border-color: #2f6b26;
        }
        .dropzone p {
            color: #4b5563;
            margin-bottom: 8px;
        }
        .dropzone .file-note {
            color: #6b7280;
            font-size: 14px;
        }
        .preview {
            display: none;
            margin-bottom: 16px;
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 16px;
        }
        @media (min-width: 640px) {
            .preview {
                grid-template-columns: repeat(3, 1fr);
            }
        }
        .preview img {
            width: 100%;
            height: 128px;
            object-fit: cover;
            border-radius: 8px;
        }
        .preview p {
            font-size: 14px;
            color: #4b5563;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .button-group {
            display: flex;
            justify-content: space-between;
            margin-bottom: 32px;
        }
        .btn {
            padding: 8px 16px;
            border-radius: 8px;
            font-size: 14px;
            cursor: pointer;
            transition: background-color 0.3s ease;
            border: none;
        }
        .btn-secondary {
            background-color: #6b7280;
            color: white;
        }
        .btn-secondary:hover {
            background-color: #4b5563;
        }
        .model-selection {
            margin-bottom: 32px;
        }
        .model-selection h2 {
            font-size: 20px;
            font-weight: 600;
            color: #1f2937;
            margin-bottom: 16px;
        }
        .model-grid {
            display: grid;
            grid-template-columns: 1fr;
            gap: 16px;
        }
        @media (min-width: 640px) {
            .model-grid {
                grid-template-columns: repeat(3, 1fr);
            }
        }
        .model-card {
            background-color: #e6efe6;
            border: 1px solid #4b8a3e;
            padding: 16px;
            border-radius: 8px;
        }
        .model-card label {
            display: flex;
            align-items: center;
            cursor: pointer;
        }
        .model-card input[type="checkbox"] {
            width: 20px;
            height: 20px;
            border: 1px solid #4b8a3e;
            border-radius: 4px;
            margin-right: 8px;
            cursor: pointer;
        }
        .model-card input[type="checkbox"]:checked {
            background-color: #4b8a3e;
            background-image: url("data:image/svg+xml,%3csvg viewBox='0 0 16 16' fill='white' xmlns='http://www.w3.org/2000/svg'%3e%3cpath d='M12.207 4.793a1 1 0 010 1.414l-5 5a1 1 0 01-1.414 0l-2-2a1 1 0 011.414-1.414L6.5 9.086l4.293-4.293a1 1 0 011.414 0z'/%3e%3c/svg%3e");
        }
        .model-card span {
            font-weight: 600;
            color: #1f2937;
        }
        .model-card p {
            color: #6b7280;
            font-size: 14px;
            margin-top: 4px;
        }
        .btn-primary {
            background-color: #4b8a3e;
            color: white;
            width: 100%;
            padding: 12px;
            font-size: 16px;
            border-radius: 8px;
            transition: background-color 0.3s ease;
            border: none;
        }
        .btn-primary:hover {
            background-color: #2f6b26;
        }
        .btn-primary:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }
        .btn-primary.loading::after {
            content: '';
            display: inline-block;
            width: 24px;
            height: 24px;
            border: 3px solid #ffffff;
            border-radius: 50%;
            border-top-color: #4b8a3e;
            animation: spin 1s ease-in-out infinite;
            margin-left: 8px;
            vertical-align: middle;
        }
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
        .error {
            color: #dc2626;
            text-align: center;
            margin-top: 16px;
            display: none;
        }
        .loading {
            text-align: center;
            margin-top: 16px;
            color: #4b5563;
            display: none;
        }
        .results {
            margin-top: 32px;
            display: none;
        }
        .results h2 {
            font-size: 20px;
            font-weight: 600;
            color: #1f2937;
            margin-bottom: 16px;
        }
        .result-card {
            background-color: #f9fafb;
            padding: 16px;
            border-radius: 8px;
            margin-bottom: 24px;
        }
        .result-card img {
            width: 128px;
            height: 128px;
            object-fit: cover;
            border-radius: 8px;
            margin-right: 16px;
        }
        .result-card h3 {
            font-size: 16px;
            font-weight: 600;
            color: #1f2937;
            margin-bottom: 8px;
        }
        .prediction {
            margin-bottom: 8px;
        }
        .prediction p {
            color: #4b5563;
            font-size: 14px;
        }
        .prediction .model-name {
            font-weight: 500;
        }
        .prediction .class {
            color: #4b8a3e;
        }
        @media (min-width: 640px) {
            .result-card {
                display: flex;
                align-items: flex-start;
            }
        }
        input[type="file"] {
            display: none;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🤖 Bean Leaf Classifier</h1>
        <p class="subtitle">Upload multiple images and classify them using advanced AI models</p>

        <div class="upload-section">
            <div id="dropzone" class="dropzone">
                <span style="font-size: 40px;">📤</span>
                <p>Drop multiple images here or click to browse</p>
                <p class="file-note">Supports JPG, PNG up to 10MB each</p>
                <input type="file" id="fileInput" accept=".jpg,.jpeg,.png" multiple>
            </div>

            <div id="preview" class="preview">
            </div>

            <div class="button-group">
                <button id="addMoreBtn" class="btn btn-secondary">+ Add More</button>
                <button id="clearAllBtn" class="btn btn-secondary">🗑️ Clear All</button>
            </div>
        </div>

        <div class="model-selection">
            <h2>🎯 Select Classification Model</h2>
            <div class="model-grid">
                <div class="model-card">
                    <label>
                        <input type="checkbox" value="mobilenet_v2" class="model-checkbox" checked>
                        <div>
                            <span>MobileNetV2</span>
                            <p>A lightweight model for mobile apps</p>
                        </div>
                    </label>
                </div>
                <div class="model-card">
                    <label>
                        <input type="checkbox" value="vit" class="model-checkbox" checked>
                        <div>
                            <span>Vision Transformer</span>
                            <p>A transformer-based model</p>
                        </div>
                    </label>
                </div>
                <div class="model-card">
                    <label>
                        <input type="checkbox" value="resnet18" class="model-checkbox" checked>
                        <div>
                            <span>ResNet18</span>
                            <p>An ImageNet model for classification</p>
                        </div>
                    </label>
                </div>
            </div>
        </div>

        <button id="classifyBtn" class="btn-primary" disabled>🔍 Classify All Images</button>

        <div id="loading" class="loading">Processing images...</div>
        <p id="error" class="error"></p>

        <div id="results" class="results">
            <h2>🎯 Classification Results</h2>
            <div id="predictions">
            </div>
        </div>
    </div>

    <script>
        const dropzone = document.getElementById('dropzone');
        const fileInput = document.getElementById('fileInput');
        const preview = document.getElementById('preview');
        const addMoreBtn = document.getElementById('addMoreBtn');
        const clearAllBtn = document.getElementById('clearAllBtn');
        const classifyBtn = document.getElementById('classifyBtn');
        const error = document.getElementById('error');
        const results = document.getElementById('results');
        const predictions = document.getElementById('predictions');
        const loading = document.getElementById('loading');
        const modelCheckboxes = document.querySelectorAll('.model-checkbox');
        let files = [];

        function updateClassifyButton() {
            const filesSelected = files.length > 0;
            const modelsSelected = Array.from(modelCheckboxes).some(checkbox => checkbox.checked);
            classifyBtn.disabled = !(filesSelected && modelsSelected);
            if (!modelsSelected) {
                showError('Please select at least one model.');
            } else if (!filesSelected) {
                showError('Please upload at least one image.');
            } else {
                error.style.display = 'none';
            }
        }

        modelCheckboxes.forEach(checkbox => {
            checkbox.addEventListener('change', updateClassifyButton);
        });

        dropzone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropzone.classList.add('dragover');
        });

        dropzone.addEventListener('dragleave', () => {
            dropzone.classList.remove('dragover');
        });

        dropzone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropzone.classList.remove('dragover');
            handleFiles(e.dataTransfer.files);
        });

        dropzone.addEventListener('click', () => fileInput.click());
        addMoreBtn.addEventListener('click', () => fileInput.click());

        fileInput.addEventListener('change', () => {
            handleFiles(fileInput.files);
            fileInput.value = '';
        });

        clearAllBtn.addEventListener('click', () => {
            files = [];
            preview.innerHTML = '';
            preview.style.display = 'none';
            results.style.display = 'none';
            updateClassifyButton();
        });

        function handleFiles(newFiles) {
            const validFiles = Array.from(newFiles).filter(file => 
                ['image/jpeg', 'image/png'].includes(file.type) && file.size <= 10 * 1024 * 1024
            );
            if (validFiles.length < newFiles.length) {
                showError('Some files were invalid. Only JPG/PNG up to 10MB are allowed.');
            }
            files = [...files, ...validFiles];
            updatePreview();
            updateClassifyButton();
        }

        function updatePreview() {
            preview.innerHTML = '';
            if (files.length > 0) {
                preview.style.display = 'grid';
                files.forEach((file, index) => {
                    const reader = new FileReader();
                    reader.onload = () => {
                        const div = document.createElement('div');
                        div.innerHTML = `
                            <img src="${reader.result}" alt="Preview ${index + 1}">
                            <p>${file.name}</p>
                        `;
                        preview.appendChild(div);
                    };
                    reader.readAsDataURL(file);
                });
            } else {
                preview.style.display = 'none';
            }
        }

        function showError(message) {
            error.textContent = message;
            error.style.display = 'block';
            results.style.display = 'none';
        }

        classifyBtn.addEventListener('click', async () => {
            if (classifyBtn.disabled) return;
            classifyBtn.disabled = true;
            classifyBtn.classList.add('loading');
            error.style.display = 'none';
            results.style.display = 'none';
            loading.style.display = 'block';

            const selectedModels = Array.from(modelCheckboxes)
                .filter(checkbox => checkbox.checked)
                .map(checkbox => checkbox.value);

            try {
                const allResults = [];
                
                // Process each file individually
                for (let i = 0; i < files.length; i++) {
                    const file = files[i];
                    const formData = new FormData();
                    formData.append('file', file);
                    formData.append('models', selectedModels.join(','));

                    const response = await fetch('/predict', {
                        method: 'POST',
                        body: formData
                    });

                    const data = await response.json();
                    
                    if (!response.ok) {
                        throw new Error(data.error || 'Failed to get predictions.');
                    }

                    allResults.push({
                        filename: file.name,
                        predictions: data.predictions,
                        fileIndex: i
                    });
                }

                // Display results
                predictions.innerHTML = '';
                allResults.forEach(result => {
                    const file = files[result.fileIndex];
                    const reader = new FileReader();
                    reader.onload = () => {
                        const imgSrc = reader.result;
                        let html = `
                            <div class="result-card">
                                <img src="${imgSrc}" alt="${result.filename}">
                                <div>
                                    <h3>Image: ${result.filename}</h3>
                                    <div>
                        `;
                        Object.entries(result.predictions).forEach(([model, pred]) => {
                            const probs = pred.probabilities
                                .map((prob, i) => `${['angular_leaf_spot', 'bean_rust', 'healthy'][i]}: ${(prob * 100).toFixed(2)}%`)
                                .join('<br>');
                            html += `
                                <div class="prediction">
                                    <p class="model-name">${model.replace('_', ' ').toUpperCase()}</p>
                                    <p>Class: <span class="class">${pred.class}</span></p>
                                    <p>Probabilities:<br>${probs}</p>
                                </div>
                            `;
                        });
                        html += '</div></div></div>';
                        predictions.innerHTML += html;
                    };
                    reader.readAsDataURL(file);
                });

                results.style.display = 'block';
                classifyBtn.classList.remove('loading');
                loading.style.display = 'none';
                classifyBtn.disabled = false;

            } catch (err) {
                showError('Error connecting to the server: ' + err.message);
                classifyBtn.classList.remove('loading');
                loading.style.display = 'none';
                classifyBtn.disabled = false;
            }
        });
    </script>
</body>
</html>
"""

@app.route('/')
def serve_ui():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if file is in the request
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        models_param = request.form.get('models', 'mobilenet_v2,vit,resnet18')
        selected_models = [m.strip() for m in models_param.split(',')]
        
        # Validate file
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
            
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Allowed: jpg, jpeg, png'}), 400
            
        # Read and process image
        image = Image.open(io.BytesIO(file.read())).convert('RGB')
        
        # Get predictions from selected models
        predictions = {}
        for model_name in selected_models:
            if model_name in models_dict:
                result = predict_image(image, models_dict[model_name])
                if result is None:
                    return jsonify({'error': f'Prediction failed for {model_name}'}), 500
                predictions[model_name] = result
            else:
                return jsonify({'error': f'Model {model_name} not available'}), 400
            
        return jsonify({
            'status': 'success',
            'predictions': predictions
        })
        
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'}), 200

if __name__ == '__main__':
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Install flask-cors if not already installed
    try:
        import flask_cors
    except ImportError:
        print("Installing flask-cors...")
        os.system("pip install flask-cors")
        import flask_cors
    
    app.run(debug=True, host='0.0.0.0', port=5000)