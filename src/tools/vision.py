"""
Vision Tool — ViT-based Chart Analysis
Uses HuggingFace ViTForImageClassification (google/vit-base-patch16-224)
as used in notebook4624369ca8.ipynb for candlestick pattern classification.
Falls back to custom-trained weights if available at models/vit_base.pth.
"""
import os
import logging
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

logger = logging.getLogger(__name__)

# Device config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Class labels
CLASS_NAMES = ['Bearish', 'Bullish']

# Image preprocessing (matches notebook4624369ca8.ipynb)
_data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- Model Loading ---
# Try HuggingFace ViTForImageClassification first (from notebook4624369ca8.ipynb)
# Fall back to timm-based StockViT for backward compatibility with kaggle_training_script.py

_model = None
_model_type = None  # 'huggingface' or 'timm'

def _load_model():
    """Lazy-load the ViT model. Tries HuggingFace first, then timm."""
    global _model, _model_type
    
    if _model is not None:
        return
    
    weights_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "models", "vit_base.pth"
    )
    
    # Strategy 1: HuggingFace ViT (as in notebook4624369ca8.ipynb)
    try:
        from transformers import ViTForImageClassification
        
        if os.path.exists(weights_path):
            # Load custom-trained HuggingFace weights
            model = ViTForImageClassification.from_pretrained(
                'google/vit-base-patch16-224',
                num_labels=2,
                ignore_mismatched_sizes=True
            )
            model.load_state_dict(torch.load(weights_path, map_location=device), strict=False)
            logger.info(f"Loaded custom ViT weights from {weights_path}")
        else:
            # Use pretrained HuggingFace ViT (will still provide meaningful features)
            model = ViTForImageClassification.from_pretrained(
                'google/vit-base-patch16-224',
                num_labels=2,
                ignore_mismatched_sizes=True
            )
            logger.info("Using HuggingFace pretrained ViT (no custom weights found)")
        
        model = model.to(device)
        model.eval()
        _model = model
        _model_type = 'huggingface'
        logger.info(f"ViT Model (HuggingFace) initialized on {device}")
        return
        
    except ImportError:
        logger.warning("transformers not installed — falling back to timm")
    except Exception as e:
        logger.warning(f"HuggingFace model load failed ({e}) — trying timm fallback")
    
    # Strategy 2: timm-based StockViT (backward compat with kaggle_training_script.py)
    try:
        import timm
        
        class StockViT(nn.Module):
            def __init__(self, num_classes=2):
                super().__init__()
                self.model = timm.create_model('vit_base_patch16_224', pretrained=False)
                n_features = self.model.head.in_features
                self.model.head = nn.Linear(n_features, num_classes)
            def forward(self, x):
                return self.model(x)
        
        model = StockViT(num_classes=2).to(device)
        
        if os.path.exists(weights_path):
            model.load_state_dict(torch.load(weights_path, map_location=device))
            logger.info(f"Loaded timm ViT weights from {weights_path}")
        else:
            logger.info("Using untuned timm ViT architecture (no weights found)")
        
        model.eval()
        _model = model
        _model_type = 'timm'
        logger.info(f"ViT Model (timm) initialized on {device}")
        
    except Exception as e:
        logger.error(f"Failed to initialize any ViT Model: {e}")
        _model = None


def analyze_chart_vision(image_path: str) -> str:
    """
    Analyzes a candlestick chart image using a Vision Transformer.
    Returns a structured result with prediction and confidence score.
    """
    _load_model()
    
    if _model is None:
        return "Vision Agent Error: Model not loaded."
    
    try:
        img = Image.open(image_path).convert('RGB')
        img_tensor = _data_transforms(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            if _model_type == 'huggingface':
                output = _model(pixel_values=img_tensor)
                logits = output.logits
            else:
                logits = _model(img_tensor)
            
            probs = torch.softmax(logits, dim=1)
            confidence, prediction = torch.max(probs, dim=1)
            confidence_val = confidence.item()
            pred_class = CLASS_NAMES[prediction.item()]
        
        return f"Vision Agent Analysis: {pred_class} (confidence: {confidence_val:.2f})"
        
    except FileNotFoundError:
        return f"Vision Agent Error: Image file '{image_path}' not found."
    except Exception as e:
        return f"Vision Agent Error: {str(e)}"


def get_vision_signal_parsed(result_str: str) -> tuple[str, float]:
    """
    Parses the vision result string into (signal, confidence).
    Returns ('Bullish'|'Bearish'|'Unknown', confidence_float).
    """
    signal = 'Unknown'
    confidence = 0.0
    
    if 'Bullish' in result_str:
        signal = 'Bullish'
    elif 'Bearish' in result_str:
        signal = 'Bearish'
    
    try:
        conf_str = result_str.split('confidence:')[1].strip().rstrip(')')
        confidence = float(conf_str)
    except (IndexError, ValueError):
        pass
    
    return signal, confidence
