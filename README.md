# 🌱 Bean Disease Classifier Flask API

## Overview

The Bean Disease Classifier is an AI-powered web application that uses deep learning models to classify bean leaf diseases. The application can identify three conditions: **Angular Leaf Spot**, **Bean Rust**, and **Healthy** leaves. It provides both a modern web interface and HTTP API endpoints for image classification using multiple pre-trained models.

## 🚀 Features

- **Multi-Model Classification**: Supports MobileNet V2, ResNet-18, and Vision Transformer (ViT) models
- **Batch Processing**: Upload and classify multiple images simultaneously
- **Modern Web Interface**: Drag-and-drop file upload with real-time results
- **API Endpoints**: Programmatic access for integration with other applications
- **Memory Optimization**: Smart model loading to handle resource constraints
- **Export Results**: Download classification results in JSON, CSV, or text format
- **Cross-Platform**: Dockerized for easy deployment
- **Responsive Design**: Works on desktop and mobile devices

## 🎯 Supported Classifications

| Disease | Description |
|---------|-------------|
| **Angular Leaf Spot** | Bacterial disease causing angular lesions on leaves |
| **Bean Rust** | Fungal disease creating rust-colored pustules |
| **Healthy** | Normal, disease-free bean leaves |

## 📁 Project Structure

```
bean-disease-classifier/
├── app.py                          # Full-featured Flask application
├── app1.py                         # Optimized version for deployment
├── Dockerfile                      # Container configuration
├── requirements.txt                # Production dependencies
├── templates/
│   ├── frontend.html              # Modern web interface
│   └── index.html                 # Legacy interface
├── static/
│   └── uploads/                   # Temporary image storage
├── models/                        # Pre-trained model files
│   ├── mobilenetv2_normalized_weights.pt
│   ├── resnet_model.pth
│   └── vit_model.pt
└── README.md                      # This file
```

## 🔧 Prerequisites

- Python 3.8 or higher
- PyTorch
- PIL (Python Imaging Library)
- Flask and dependencies
- Docker (optional, for containerized deployment)

## 📦 Installation

### Local Development Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/Samuelms46/Bean-disease-classifier-FlaskAPI.git
   cd Bean-disease-classifier-FlaskAPI
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Ensure model files are present**
   ```bash
   # Place your trained models in the models/ directory:
   # - mobilenetv2_normalized_weights.pt
   # - resnet_model.pth  
   # - vit_model.pt
   ```

### Docker Deployment

```bash
# Build the container
docker build -t bean-leaf-disease-classifier .

# Run the container
docker run -p 8080:8080 bean-leaf-disease-classifier
```

## 🚀 Running the Application

### Development Mode
```bash
python app.py
```

### Production Mode (Optimized)
```bash
python app1.py
```

### Using Gunicorn (Production)
```bash
gunicorn --timeout 300 --workers 1 --threads 1 -b 0.0.0.0:8080 app1:app
```

The application will be available at:
- **Modern Interface**: `http://localhost:5000/frontend`
- **Legacy Interface**: `http://localhost:5000/simple`
- **API Health Check**: `http://localhost:5000/health`

## 🌐 Web Interface

### Modern Frontend (`/frontend`)
- **Drag & Drop Upload**: Intuitive file upload interface
- **Multi-Model Selection**: Choose from available AI models
- **Batch Processing**: Upload multiple images at once
- **Real-time Results**: Instant classification with confidence scores
- **Export Options**: Download results in multiple formats
- **Summary Statistics**: Overview of classification results
- **Error Handling**: User-friendly error messages

### Legacy Interface (`/simple`)
- Basic results display

## 🔌 API Endpoints

### Base URL
```
http://localhost:5000
```

### Available Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET    | `/health` | Server health check and model status |
| GET    | `/models` | List available models and descriptions |
| POST   | `/predict` | Classify uploaded images |
| GET    | `/frontend` | Modern web interface |
| GET    | `/simple` | Legacy web interface |

### API Usage Examples

#### Health Check
```bash
curl -X GET http://localhost:5000/health
```

Response:
```json
{
  "status": "healthy",
  "models_loaded": 2,
  "available_models": ["mobilenet_v2", "resnet18"],
  "device": "cpu"
}
```

#### Get Available Models
```bash
curl -X GET http://localhost:5000/models
```

Response:
```json
{
  "available_models": ["mobilenet_v2", "resnet18"],
  "model_info": {
    "mobilenet_v2": "MobileNet V2 - Lightweight model optimized for mobile devices",
    "resnet18": "ResNet-18 - Deep residual network with skip connections"
  }
}
```

#### Image Classification
```bash
curl -X POST http://localhost:5000/predict \
  -F "file=@path/to/healthy_test.jpg" \
  -F "models=mobilenet_v2,resnet18"
```

Response:
```json
{
  "predictions": {
    "mobilenet_v2": {
      "class": "healthy",
      "probabilities": [0.1, 0.2, 0.7],
      "confidence": 0.7
    },
    "resnet18": {
      "class": "healthy", 
      "probabilities": [0.15, 0.25, 0.6],
      "confidence": 0.6
    }
  },
  "filename": "healty_test.jpg"
}
```

## 🤖 AI Models

### MobileNet V2
- **Type**: Convolutional Neural Network
- **Optimization**: Lightweight, mobile-optimized
- **Use Case**: Fast inference, resource-constrained environments
- **File**: `mobilenetv2_normalized_weights.pt`

### ResNet-18
- **Type**: Residual Neural Network
- **Features**: Skip connections, deep architecture
- **Use Case**: Balanced accuracy and performance
- **File**: `resnet_model.pth`

### Vision Transformer (ViT)
- **Type**: Transformer-based model
- **Features**: Attention mechanisms, state-of-the-art accuracy
- **Use Case**: High accuracy requirements
- **File**: `vit_model.pt`
- **Note**: May be disabled on free-tier deployments due to memory constraints

## 🔧 Configuration

### Environment Variables
```bash
export FLASK_ENV=production          # or development
export FLASK_DEBUG=0                 # Set to 1 for debugging
export DEVICE=cpu                    # or cuda if GPU available
```

### Memory Optimization
The application uses smart model loading:
- Only one model loaded at a time
- Automatic memory cleanup
- CPU-first loading with GPU transfer
- Optimized for free-tier deployments

## 🐳 Docker Configuration

The `Dockerfile` is optimized for production deployment:
- Uses Python 3.12 slim base image
- Minimal dependencies for reduced image size
- Gunicorn WSGI server with optimized settings
- Configurable port via environment variable

```bash
# Build and run
docker build -t bean-leaf-disease-classifier .
docker run -p 8080:8080 -e PORT=8080 bean-leaf-disease-classifier
```

## 📊 Image Requirements

- **Formats**: JPG, JPEG, PNG
- **Size Limit**: 10MB per image
- **Recommended**: 224x224 pixels (automatically resized)
- **Content**: Bean leaf images with clear visibility

## 🔍 Error Handling

The API provides consistent error responses:

```json
{
  "error": "Error description",
  "status": "error"
}
```

Common error scenarios:
- Invalid file format
- File size too large
- No models selected
- Model loading failures
- Server connectivity issues

## 🧪 Testing

### Manual Testing
1. Start the application
2. Navigate to `/frontend`
3. Upload test images
4. Verify classifications

## 🚀 Deployment Options

### Local Development
```bash
python app.py  # Full features
python app1.py # Optimized version
```

### Cloud Platforms
- **Render**: Configured with Dockerfile

### Production Considerations
- Use `app1.py` for optimized memory usage
- Configure appropriate worker processes
- Set up proper logging and monitoring
- Implement rate limiting for API endpoints

## 🔒 Security Notes

- File upload validation implemented
- Secure filename handling
- CORS enabled for frontend integration
- Input sanitization for API endpoints
- Memory management to prevent DoS

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/improvement`)
5. Create Pull Request

## 📝 License

[license]

## 👨‍💻 Author

**Samuel Muwanguzi** - [GitHub Profile](https://github.com/Samuelms46)\
**Kashara Alvin Ssali** [GitHub Profile](https://github.com/Kashara-Alvin-Ssali)\
**Edwin** [GitHub Profile]()


## 🙏 Acknowledgments

- PyTorch team for the deep learning framework
- Flask community for the web framework
- Contributors to the pre-trained models
- Makerere AI lab research community for disease classification datasets

---

## 🔧 Troubleshooting

### Common Issues

1. **Model files not found**
   ```bash
   # Ensure model files are in the models/ directory
   ls models/
   ```

2. **Memory errors**
   ```bash
   # Use the optimized version
   python app1.py
   ```

3. **Port conflicts**
   ```bash
   # Kill existing processes
   lsof -ti:5000 | xargs kill -9
   ```

4. **CUDA/GPU issues**
   ```bash
   # Force CPU usage
   export DEVICE=cpu
   ```

### Performance Tips

- Use MobileNet V2 for fastest inference
- Process images in smaller batches
- Resize images to 224x224 before upload
- Use the optimized `app1.py` for production

## 🔄 Model Loading Strategy

The application implements intelligent model management:

1. **On-Demand Loading**: Models are loaded only when requested
2. **Memory Cleanup**: Previous models are cleared before loading new ones
3. **Fallback Handling**: Graceful degradation when models are unavailable
4. **Device Management**: Automatic CPU/GPU detection and allocation

## 🔐 API Security

### Input Validation
- File type checking (JPG, PNG only)
- File size limits (10MB maximum)
- Filename sanitization
- Content-type verification

### Error Handling
- Graceful failure modes
- Informative error messages
- Request timeout handling
- Memory overflow protection

## 🌍 Deployment Environments

### Development
```bash
# Full features, debugging enabled
export FLASK_ENV=development
export FLASK_DEBUG=1
python app.py
```

### Staging
```bash
# Optimized version, 
export FLASK_ENV=production
export FLASK_DEBUG=1
python app1.py
```

### Production
```bash
# Container deployment with Gunicorn
docker run -p 8080:8080 -e PORT=8080 bean-leaf-disease-classifier
```

## 📊 Monitoring and Logging

### Health Monitoring
- `/health` endpoint for service status
- Model availability checking
- Memory usage tracking
- Device status reporting

### Logging
- Request/response logging
- Error tracking
- Performance metrics
- Model loading events

## 🔧 Configuration Options

### Flask Configuration
```python
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10MB
app.config["UPLOAD_FOLDER"] = Path("static/uploads")
```

### Model Configuration
```python
MODEL_DIR = Path("models")
CLASS_NAMES = ['angular_leaf_spot', 'bean_rust', 'healthy']
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
```

### Image Processing
```python
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
```

## 🚀 Scaling Considerations

### Horizontal Scaling
- Stateless design enables multiple instances
- Load balancer compatible
- Shared model storage recommended

### Vertical Scaling
- Memory optimization for larger models
- GPU acceleration support
- Batch processing capabilities

### Performance Optimization
- Model caching strategies
- Image preprocessing optimization
- Response compression
- CDN integration for static assets

## 🔍 Debugging Guide

### Common Debug Steps
1. Check model file existence and permissions
2. Verify Python dependencies
3. Test with sample images
4. Monitor memory usage
5. Check network connectivity

### Debug Commands
```bash
# Check model files
ls -la models/

# Test API endpoints
curl -v http://localhost:5000/health

# Check logs
tail -f app.log
```

## 📚 Additional Resources

### Documentation
- [Flask Documentation](https://flask.palletsprojects.com/)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [Docker Documentation](https://docs.docker.com/)

### Support
- GitHub Issues for bug reports
- Discussions for feature requests
- Community contributions welcome

---

**Last Updated**: [Current Date]
**Version**: 1.0.0
**Compatibility**: Python 3.8+, PyTorch 1.9+, Flask 2.0+
