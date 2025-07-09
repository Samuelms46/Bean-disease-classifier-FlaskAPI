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
from transformers import ViTForImageClassification, ViTConfig
import os
import base64
from io import BytesIO

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
MODEL_DIR = Path("models")
CLASS_NAMES = ['angular_leaf_spot', 'bean_rust', 'healthy']
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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
app.config["UPLOAD_FOLDER"].mkdir(exist_ok=True)

# ----------------------------------------------------------------------
# Model loading helpers
# ----------------------------------------------------------------------
@lru_cache(maxsize=1)
def load_vit():
    try:
        config = ViTConfig(num_labels=3)
        model = ViTForImageClassification(config)
        model_path = MODEL_DIR / "vit_model.pt"
        if model_path.exists():
            sd = torch.load(model_path, map_location=DEVICE)
            model.load_state_dict(sd)
            return model.to(DEVICE).eval()
        else:
            print(f"ViT model not found at {model_path}")
            return None
    except Exception as e:
        print(f"Error loading ViT model: {e}")
        return None

@lru_cache(maxsize=1)
def load_resnet():
    try:
        model = models.resnet18()
        model.fc = nn.Linear(model.fc.in_features, 3)
        model_path = MODEL_DIR / "resnet_model.pth"
        if model_path.exists():
            sd = torch.load(model_path, map_location=DEVICE)
            model.load_state_dict(sd)
            return model.to(DEVICE).eval()
        else:
            print(f"ResNet model not found at {model_path}")
            return None
    except Exception as e:
        print(f"Error loading ResNet model: {e}")
        return None

@lru_cache(maxsize=1)
def load_mobilenet():
    try:
        model = models.mobilenet_v2()
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 3)
        model_path = MODEL_DIR / "best_model.pt"
        if model_path.exists():
            sd = torch.load(model_path, map_location=DEVICE)
            model.load_state_dict(sd, strict=False)
            return model.to(DEVICE).eval()
        else:
            print(f"MobileNet model not found at {model_path}")
            return None
    except Exception as e:
        print(f"Error loading MobileNet model: {e}")
        return None

MODEL_REGISTRY = {
    "vit": load_vit,
    "resnet18": load_resnet,
    "mobilenet_v2": load_mobilenet
}

MODEL_DESCRIPTIONS = {
    "vit": "Vision Transformer - Advanced attention-based model for image classification",
    "resnet18": "ResNet-18 - Deep residual network with skip connections",
    "mobilenet_v2": "MobileNet V2 - Lightweight model optimized for mobile devices"
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
    """Predict image class using specified models"""
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
    
    for model_name in model_names:
        if model_name in MODEL_REGISTRY:
            try:
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
    available_models = get_available_models()
    return jsonify({
        "status": "healthy",
        "models_loaded": len(available_models),
        "available_models": available_models,
        "device": DEVICE
    })

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
        # Check if file was uploaded
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        # Get selected models
        models_str = request.form.get('models', '')
        if not models_str:
            return jsonify({"error": "No models selected"}), 400
        
        selected_models = [m.strip() for m in models_str.split(',')]
        
        # Validate file type
        if not file.content_type.startswith('image/'):
            return jsonify({"error": "Invalid file type. Please upload an image."}), 400
        
        # Make prediction
        predictions = predict_image(file, selected_models)
        
        if not predictions:
            return jsonify({"error": "No predictions could be made"}), 500
        
        return jsonify({
            "predictions": predictions,
            "filename": secure_filename(file.filename)
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/', methods=['GET', 'POST'])
def index():
    """Main route - serves the original form-based interface"""
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
    print("Server will be available at:")
    print("- Original interface: http://localhost:5000/")
    print("- New frontend: http://localhost:5000/frontend")
    print("- API health check: http://localhost:5000/health")
    
    app.run(debug=True, host='0.0.0.0', port=5000)