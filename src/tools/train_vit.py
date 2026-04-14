import os
import sys
# Append root directory to sys path to resolve 'src'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import logging
import argparse

# Import the model architecture from vision
from src.tools.vision import StockViT

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Same data transformations used in the notebook
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        # Removed HorizontalFlip: Flipping a chart horizontally runs time backwards!
        # Removed ColorJitter: We don't want to accidentally change red/green candle semantics.
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
}

def train_model(data_path: str, epochs: int = 15, batch_size: int = 32, save_path: str = "models/vit_base.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training on device: {device}")
    
    # 1. Prepare Datasets
    try:
        train_dataset = datasets.ImageFolder(os.path.join(data_path, 'Train'), transform=data_transforms['train'])
        test_dataset = datasets.ImageFolder(os.path.join(data_path, 'Test'), transform=data_transforms['test'])
        
        loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        logger.info(f"Dataset ready. Train: {len(train_dataset)} | Test: {len(test_dataset)}")
    except Exception as e:
        logger.error(f"Failed to load dataset from {data_path}. Error: {e}")
        return

    # 2. Init Model
    model = StockViT(num_classes=2).to(device)
    
    # 3. Optimization Setup
    criterion = nn.CrossEntropyLoss()
    # Lowered learning rate for fine-tuning the pretrained ViT
    optimizer = optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.05)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    # 4. Training Loop
    best_acc = 0.0
    
    for epoch in range(epochs):
        model.train()
        train_correct, total = 0, 0
        
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
        # Eval Phase
        model.eval()
        test_correct = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                test_correct += (predicted == labels).sum().item()
                
        scheduler.step()
        
        train_acc = 100 * train_correct / total
        test_acc = 100 * test_correct / len(test_dataset)
        
        logger.info(f"Epoch {epoch+1}/{epochs} | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%")
        
        # Save Best Model
        if test_acc > best_acc:
            best_acc = test_acc
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
            logger.info(f"  --> Saved new best model to {save_path}")
            
    logger.info("Training Complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ViT Transformer for Candlestick Charts")
    parser.add_argument("--data_path", type=str, default="", help="Path to Kaggle dataset root. If empty, uses kagglehub to download.")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="DataLoader batch size")
    parser.add_argument("--save_path", type=str, default="models/vit_base.pth", help="Path to save the trained state dictionary")
    
    args = parser.parse_args()
    
    data_path = args.data_path
    if not data_path:
        logger.info("No data_path provided. Using kagglehub to download 'raimiazeezbabatunde/candle-image-data'...")
        try:
            import kagglehub
            data_path = kagglehub.dataset_download("raimiazeezbabatunde/candle-image-data")
            logger.info(f"Successfully downloaded dataset to: {data_path}")
        except ImportError:
            logger.error("kagglehub is not installed. Please run `pip install kagglehub` or provide a manual --data_path")
            sys.exit(1)
            
    train_model(data_path, args.epochs, args.batch_size, args.save_path)
