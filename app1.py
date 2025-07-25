from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from pathlib import Path
from functools import lru_cache
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision import models
# Removed transformers import to reduce memory usage on free tier
import os
import base64
import gc
from io import BytesIO

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
MODEL_DIR = Path("models")
CLASS_NAMES = ['angular_leaf_spot', 'bean_rust', 'healthy']
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_num_threads(1)

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225])
])

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend connection
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB upload limit
app.config["UPLOAD_FOLDER"] = Path("static/uploads")

# Ensure upload directory exists
upload_dir = Path("static/uploads")
upload_dir.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------
# Model loading helpers - Optimized for low memory
# ----------------------------------------------------------------------

# Global variable to store the currently loaded model
_current_model = None
_current_model_name = None

def load_model_on_demand(model_name):
    """Load only one model at a time to save memory"""
    global _current_model, _current_model_name
    
    # If the requested model is already loaded, return it
    if _current_model is not None and _current_model_name == model_name:
        return _current_model
    
    # Clear previous model from memory
    if _current_model is not None:
        del _current_model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Load the requested model
    try:
        if model_name == "mobilenet_v2":
            model = models.mobilenet_v2()
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, 3)
            model_path = MODEL_DIR / "mobilenetv2_normalized_weights.pt"
            
            if model_path.exists():
                print(f"Loading {model_name} model...")
                sd = torch.load(model_path, map_location="cpu")  # Load to CPU first
                model.load_state_dict(sd, strict=False)
                model = model.to(DEVICE).eval()
                
                _current_model = model
                _current_model_name = model_name
                print(f"Successfully loaded {model_name}")
                return model
            else:
                print(f"Model file not found: {model_path}")
                return None
        
        elif model_name == "resnet18":
            model = models.resnet18()
            model.fc = nn.Linear(model.fc.in_features, 3)
            model_path = MODEL_DIR / "resnet_model.pth"
            
            if model_path.exists():
                print(f"Loading {model_name} model...")
                sd = torch.load(model_path, map_location="cpu")
                model.load_state_dict(sd)
                model = model.to(DEVICE).eval()
                
                _current_model = model
                _current_model_name = model_name
                print(f"Successfully loaded {model_name}")
                return model
            else:
                print(f"Model file not found: {model_path}")
                return None
        
        else:
            # For free tier, disabled ViT model as it's too large
            print(f"Model {model_name} disabled on free tier to save memory")
            return None
            
    except Exception as e:
        print(f"Error loading {model_name}: {e}")
        return None

# Legacy functions for compatibility - use on-demand loading
@lru_cache(maxsize=1)
def load_vit():
    return None  # Disabled for free tier

@lru_cache(maxsize=1) 
def load_resnet():
    return load_model_on_demand("resnet18")

@lru_cache(maxsize=1)
def load_mobilenet():
    return load_model_on_demand("mobilenet_v2")

# Simplified model registry for free tier - prioritizing lightweight models
MODEL_REGISTRY = {
    "mobilenet_v2": lambda: load_model_on_demand("mobilenet_v2"),
    "resnet18": lambda: load_model_on_demand("resnet18"),
}

MODEL_DESCRIPTIONS = {
    "mobilenet_v2": "MobileNet V2 - Lightweight model optimized for mobile devices",
    "resnet18": "ResNet-18 - Deep residual network with skip connections",
}

def get_available_models():
    """Check which models are actually available"""
    available = []
    for model_name, loader in MODEL_REGISTRY.items():
        try:
            model = loader()
            if model is not None:
                available.append(model_name)
        except Exception as e:
            print(f"Model {model_name} not available: {e}")
    return available

def predict_image(image, model_names):
    """Predict image class using specified models - optimized for memory"""
    results = {}
    
    # Preprocess image
    if isinstance(image, str):
        # If base64 string, decode it
        image_data = base64.b64decode(image.split(',')[1])
        image = Image.open(BytesIO(image_data)).convert("RGB")
    elif hasattr(image, 'read'):
        # If file-like object
        image = Image.open(image).convert("RGB")
    
    tensor = transform(image).unsqueeze(0).to(DEVICE)
    
    # For free tier, only process one model at a time to save memory
    for model_name in model_names:
        if model_name in MODEL_REGISTRY:
            try:
                print(f"Processing prediction with {model_name}...")
                model = MODEL_REGISTRY[model_name]()
                if model is not None:
                    with torch.no_grad():
                        output = model(tensor)
                        logits = output.logits if hasattr(output, "logits") else output
                        probs = torch.softmax(logits, dim=1)[0]
                        
                        pred_idx = probs.argmax().item()
                        pred_class = CLASS_NAMES[pred_idx]
                        
                        results[model_name] = {
                            "class": pred_class,
                            "probabilities": probs.cpu().numpy().tolist(),
                            "confidence": float(probs[pred_idx].item())
                        }
                        print(f"Prediction completed for {model_name}: {pred_class}")
                        
                        # cleanup
                        del output, logits, probs
                        torch.cuda.empty_cache() if torch.cuda.is_available() else None
                else:
                    results[model_name] = {
                        "error": "Model not available on free tier"
                    }
            except Exception as e:
                print(f"Error predicting with {model_name}: {e}")
                results[model_name] = {
                    "error": str(e)
                }
    
    return results

# ----------------------------------------------------------------------
# API Routes
# ----------------------------------------------------------------------

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        print("Health check called - checking available models...")
        available_models = get_available_models()
        print(f"Available models: {available_models}")
        
        return jsonify({
            "status": "healthy",
            "models_loaded": len(available_models),
            "available_models": available_models,
            "device": DEVICE
        })
    except Exception as e:
        print(f"Health check error: {e}")
        return jsonify({
            "status": "error",
            "error": str(e),
            "models_loaded": 0,
            "available_models": [],
            "device": DEVICE
        }), 500

@app.route('/models', methods=['GET'])
def get_models():
    """Get available models and their info"""
    available_models = get_available_models()
    return jsonify({
        "available_models": available_models,
        "model_info": MODEL_DESCRIPTIONS
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Predict image class using selected models"""
    try:
        print("Predict endpoint called")
        
        # Check if file was uploaded
        if 'file' not in request.files:
            print("No file in request")
            return jsonify({"error": "No file uploaded"}), 400
        
        file = request.files['file']
        if file.filename == '':
            print("Empty filename")
            return jsonify({"error": "No file selected"}), 400
        
        # Get selected models
        models_str = request.form.get('models', '')
        print(f"Models requested: {models_str}")
        if not models_str:
            return jsonify({"error": "No models selected"}), 400
        
        selected_models = [m.strip() for m in models_str.split(',')]
        print(f"Selected models: {selected_models}")
        
        # Validate file type
        if not file.content_type.startswith('image/'):
            print(f"Invalid file type: {file.content_type}")
            return jsonify({"error": "Invalid file type. Please upload an image."}), 400
        
        # Make prediction
        print("Making predictions...")
        predictions = predict_image(file, selected_models)
        print(f"Predictions result: {predictions}")

        gc.collect()
        
        if not predictions:
            return jsonify({"error": "No predictions could be made"}), 500
        
        return jsonify({
            "predictions": predictions,
            "filename": secure_filename(file.filename)
        })
        
    except Exception as e:
        print(f"Error in predict endpoint: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route('/')
def index():
    """Main route - redirect to modern frontend"""
    from flask import redirect
    return redirect('/frontend')

@app.route('/simple', methods=['GET', 'POST'])
def simple_interface():
    """Simple form-based interface (legacy)"""
    if request.method == 'POST':
        model_key = request.form.get('model')
        files = request.files.getlist('image')
        
        if not files or model_key not in MODEL_REGISTRY:
            return render_template('index.html', results=None, choice=None)

        results = []
        upload_folder = Path("static/uploads")
        upload_folder.mkdir(exist_ok=True)

        for file in files:
            if file.filename:
                filename = secure_filename(file.filename)
                save_path = upload_folder / filename
                file.save(save_path)

                # Make prediction
                predictions = predict_image(save_path, [model_key])
                
                if model_key in predictions:
                    pred_data = predictions[model_key]
                    results.append({
                        "filename": filename,
                        "pred_class": pred_data["class"],
                        "confidence": f"{pred_data['confidence'] * 100:.2f}",
                        "image_url": f"/static/uploads/{filename}"
                    })

        return render_template('index.html', results=results, choice=model_key)
    
    return render_template('index.html', results=None, choice=None)

@app.route('/frontend')
def frontend():
    """Serve the new frontend interface"""
    return render_template('frontend.html')

# ----------------------------------------------------------------------
# Error Handlers
# ----------------------------------------------------------------------

@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": "File too large. Maximum size is 10MB."}), 413

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(e):
    return jsonify({"error": "Internal server error"}), 500

# ----------------------------------------------------------------------
if __name__ == "__main__":
    print("Starting Bean Leaf Classifier Server...")
    print(f"Using device: {DEVICE}")
    print(f"Available models: {get_available_models()}")
    
    # Read configuration from environment variables
    debug_mode = os.getenv('FLASK_DEBUG', '0') == '1'
    env = os.getenv('FLASK_ENV', 'development')
    port = int(os.getenv('PORT', 5000))
    
    print(f"Environment: {env}")
    print(f"Debug mode: {debug_mode}")
    print(f"Port: {port}")
    print("Server will be available at:")
    print(f"- Main interface: http://localhost:{port}/")
    print(f"- New frontend: http://localhost:{port}/frontend")
    print(f"- API health check: http://localhost:{port}/health")
    
    app.run(debug=debug_mode, host='0.0.0.0', port=port)