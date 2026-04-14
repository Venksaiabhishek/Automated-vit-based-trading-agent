import os
import logging
import torch
import torch.nn as nn
import timm
from torchvision import transforms
from PIL import Image
from langchain_core.tools import tool

logger = logging.getLogger(__name__)

# Device config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Simple setup for Vision Transformer (as per the requested pipeline)
class StockViT(nn.Module):
    def __init__(self, num_classes=2):
        super(StockViT, self).__init__()
        self.model = timm.create_model('vit_base_patch16_224', pretrained=False)
        # Note: Pretrained is False here assuming we load custom weights later,
        # but setting up the architecture cleanly now.
        n_features = self.model.head.in_features
        self.model.head = nn.Linear(n_features, num_classes)
        
    def forward(self, x):
        return self.model(x)

# Initialize Model (lazily loaded or mockup for now)
try:
    _model = StockViT(num_classes=2).to(device)
    
    # Check for trained weights from the new training script
    weights_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models", "vit_base.pth")
    if os.path.exists(weights_path):
        _model.load_state_dict(torch.load(weights_path, map_location=device))
        logger.info(f"Loaded trained ViT weights from {weights_path}")
    else:
        logger.info("Using untuned ViT architecture (weights not found).")
        
    _model.eval()
    logger.info(f"ViT Model initialized on {device}.")
except Exception as e:
    logger.error(f"Failed to initialize ViT Model: {e}")
    _model = None

# Classes for output
CLASS_NAMES = ['Bearish', 'Bullish']

_data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def analyze_chart_vision(image_path: str) -> str:
    """
    Analyzes a candlestick chart image using a Vision Transformer.
    Returns 'Bullish' (likely to rise) or 'Bearish' (likely to fall).
    Use this tool whenever visual confirmation of a trend is required to find liquidity gaps or footprints.
    """
    if _model is None:
        return "Vision Agent Error: Model not loaded."
        
    try:
        img = Image.open(image_path).convert('RGB')
        img_tensor = _data_transforms(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = _model(img_tensor)
            prediction = torch.argmax(output, dim=1).item()
            
        result = CLASS_NAMES[prediction]
        return f"Vision Agent Analysis: The chart shows a strong {result} pattern."
        
    except FileNotFoundError:
        return f"Vision Agent Error: Image file '{image_path}' not found."
    except Exception as e:
        return f"Vision Agent Error: {str(e)}"
