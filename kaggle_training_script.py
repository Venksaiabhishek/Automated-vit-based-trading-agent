import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import timm
import kagglehub

# 1. Standalone Model Setup 
class StockViT(nn.Module):
    def __init__(self, num_classes=2):
        super(StockViT, self).__init__()
        # Using pretrained=True here gives you a HUGE edge since you're using Kaggle GPUs.
        # The model will converge much faster on candlestick images!
        self.model = timm.create_model('vit_base_patch16_224', pretrained=True)
        n_features = self.model.head.in_features
        self.model.head = nn.Linear(n_features, num_classes)
        
    def forward(self, x):
        return self.model(x)

# 2. Transformations Pipeline
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

def train_standalone(epochs: int = 15, batch_size: int = 32):
    print("Downloading dataset from KaggleHub...")
    data_path = kagglehub.dataset_download("raimiazeezbabatunde/candle-image-data")
    print(f"Dataset downloaded to {data_path}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    train_dataset = datasets.ImageFolder(os.path.join(data_path, 'Train'), transform=data_transforms['train'])
    test_dataset = datasets.ImageFolder(os.path.join(data_path, 'Test'), transform=data_transforms['test'])
    
    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    model = StockViT(num_classes=2).to(device)
    
    criterion = nn.CrossEntropyLoss()
    # Lowered learning rate for fine-tuning the pretrained ViT
    optimizer = optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.05)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    best_acc = 0.0
    save_path = "vit_base.pth" # Saves to current Kaggle working dir
    
    print("Starting Training Loop...")
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
        
        print(f"Epoch {epoch+1}/{epochs} | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%")
        
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), save_path)
            print(f"  --> Saved new best model to {save_path}")
            
    print("Training Complete! You can now download vit_base.pth from the Kaggle Output section.")

if __name__ == "__main__":
    # Ensure dependencies are installed in Kaggle environment if not already
    os.system("pip install timm kagglehub")
    
    # Run training
    train_standalone(epochs=15, batch_size=32)
