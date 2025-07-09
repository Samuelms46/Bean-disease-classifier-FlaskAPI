from flask import Flask, request, jsonify, render_template
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
        
        loaded_models = {}
        
        # Load weights with custom handling
        for model_name, model in [('mobilenet_v2', mobilenet_v2), ('vit', vit), ('resnet18', resnet18)]:
            if os.path.exists(MODEL_PATHS[model_name]):
                try:
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
                    logger.info(f"Successfully loaded model: {model_name}")
                except Exception as e:
                    logger.error(f"Error loading {model_name}: {str(e)}")
                    logger.warning(f"Using random weights for {model_name}")
            else:
                logger.warning(f"Model file not found: {MODEL_PATHS[model_name]}. Using random weights.")
            
            model.eval()
            loaded_models[model_name] = model
        
        return loaded_models
        
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
    logger.info(f"Loaded {len(models_dict)} models successfully")
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
    
@app.route('/')
def home():
    return render_template("index.html")

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
            'predictions': predictions,
            'filename': file.filename
        })
        
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/models', methods=['GET'])
def get_available_models():
    """Return list of available models"""
    return jsonify({
        'available_models': list(models_dict.keys()),
        'model_info': {
            'mobilenet_v2': 'A lightweight model for mobile apps',
            'vit': 'A transformer-based model',
            'resnet18': 'An ImageNet model for classification'
        }
    })

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'models_loaded': len(models_dict),
        'available_models': list(models_dict.keys())
    }), 200

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
    
    print("Starting Flask server...")
    print("Available endpoints:")
    print("  - POST /predict - Classify images")
    print("  - GET /models - Get available models")
    print("  - GET /health - Health check")
    print("\nServer running on http://localhost:5000")
    
    app.run(debug=True, host='0.0.0.0', port=5000)