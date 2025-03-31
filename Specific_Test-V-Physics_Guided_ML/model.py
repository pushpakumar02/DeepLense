import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from torchvision import transforms, models
from tqdm import tqdm

# Device configuration
device = torch.device('mps')
print(f'Using device: {device}')

# ========== PHYSICS LAYER ==========
class LensingPhysicsLayer(nn.Module):
    """Implements simplified gravitational lensing equation"""
    def __init__(self):
        super().__init__()
        self.theta_E = nn.Parameter(torch.tensor(1.0))  # Trainable Einstein radius
        
    def forward(self, x):
        B, C, H, W = x.shape
        y_coords = torch.linspace(-1, 1, H, device=x.device)
        x_coords = torch.linspace(-1, 1, W, device=x.device)
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
        r = torch.sqrt(xx**2 + yy**2) + 1e-6
        deflection = self.theta_E**2 / r
        return x * deflection.unsqueeze(0).unsqueeze(0)

# Dataset class
class LensDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.files = list(self.root_dir.rglob('*.npy'))
        self.labels = [self.get_label(f) for f in self.files]
        self.transform = transform or self.default_transform()
    
    def get_label(self, file):
        file_str = str(file).lower()
        if 'vort' in file_str: return 0
        elif 'sphere' in file_str: return 1
        else: return 2
    
    def default_transform(self):
        return transforms.Compose([
            transforms.Lambda(lambda x: x.unsqueeze(0) if x.dim() == 2 else x),
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485], std=[0.229])
        ])
    
    def __len__(self): return len(self.files)
    
    def __getitem__(self, idx):
        img = np.load(self.files[idx], allow_pickle=True)
        img = torch.from_numpy(img).float()
        if img.dim() == 2: img = img.unsqueeze(0)
        if self.transform: img = self.transform(img)
        return img.to(device), torch.tensor(self.labels[idx]).to(device)

# Initialize datasets
project_root = Path('/Users/pushpakumar/Projects/GSoC25_DeepLense-Gravitational Lens Finding /Common_Test-I-MultiClass')
train_data = LensDataset(project_root / 'dataset'/'train')
val_data = LensDataset(project_root / 'dataset'/'val')

# Data loaders
train_loader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=0)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False, num_workers=0)

# Load base model
def load_resnet_from_local(weights_path):
    model = models.resnet18(weights=None)
    state_dict = torch.load(weights_path)
    if all(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    return model

base_model = load_resnet_from_local("/Users/pushpakumar/Downloads/resnet18-f37072fd.pth")
with torch.no_grad():
    base_model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    base_model.conv1.weight = nn.Parameter(base_model.conv1.weight.mean(dim=1, keepdim=True))
base_model.fc = nn.Linear(base_model.fc.in_features, 3)
base_model = base_model.to(device)

# ========== PHYSICS MODEL ==========
physics_model = nn.Sequential(
    LensingPhysicsLayer(),
    base_model
).to(device)

# Loss and optimizer
class_weights = torch.tensor([1.0, 1.5, 1.2]).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.AdamW(physics_model.parameters(), lr=0.0001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.5)

def train_model(model, train_loader, val_loader, epochs=10):
    best_val_acc = 0.0
    train_losses, val_losses, val_accs = [], [], []
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        
        for images, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}'):
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss, correct = 0.0, 0
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for images, labels in val_loader:
                outputs = model(images)
                val_loss += criterion(outputs, labels).item()
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        val_acc = 100 * correct / len(val_loader.dataset)
        scheduler.step(val_acc)
        
        print(f'\nEpoch {epoch+1}:')
        print(f'Train Loss: {epoch_loss/len(train_loader):.4f}, Val Acc: {val_acc:.2f}%')
        print(classification_report(all_labels, all_preds, target_names=['Vort', 'Sphere', 'No Substructure']))
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_physics_model.pth')
            print(f'New best model saved with val acc: {best_val_acc:.2f}%')
        
        if epoch > 7 and val_acc < 90:
            print("Early stopping - model not converging")
            break
    
    return train_losses, val_losses, val_accs

def evaluate_model(model, val_loader):
    model.eval()
    y_true, y_pred, y_probs = [], [], []
    
    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_probs.extend(probs.cpu().numpy())
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Vort', 'Sphere', 'No Substructure'],
                yticklabels=['Vort', 'Sphere', 'No Substructure'])
    plt.savefig('confusion_matrix.png')
    plt.close()

    # ROC Curve
    plt.figure(figsize=(8,6))
    for i in range(3):
        fpr, tpr, _ = roc_curve(np.array(y_true) == i, np.array(y_probs)[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'Class {i} (AUC = {roc_auc:.2f})')
    plt.plot([0,1], [0,1], 'k--')
    plt.legend(loc="lower right")
    plt.savefig('roc_curve.png')
    plt.close()

def generate_training_plots(train_losses, val_losses, val_accs):
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.legend()
    plt.subplot(1,2,2)
    plt.plot(val_accs, label='Val Accuracy')
    plt.legend()
    plt.savefig('training_curves.png')
    plt.close()

def visualize_physics_effect(model, val_loader):
    with torch.no_grad():
        sample = next(iter(val_loader))[0][0:1]
        phys_effect = model[0](sample).cpu()
        plt.figure()
        plt.imshow(phys_effect.squeeze(), cmap='viridis')
        plt.colorbar()
        plt.savefig('physics_effect.png')
        plt.close()

# Main execution
try:
    print("Starting training...")
    train_losses, val_losses, val_accs = train_model(physics_model, train_loader, val_loader, epochs=10)
except KeyboardInterrupt:
    print("\nTraining stopped early")

# Generate all outputs
plt.close('all')
generate_training_plots(train_losses, val_losses, val_accs)
visualize_physics_effect(physics_model, val_loader)

# Load and evaluate best model
physics_model.load_state_dict(torch.load('best_physics_model.pth'))
evaluate_model(physics_model, val_loader)

print("\nAll results saved:")
print("1. training_curves.png\n2. physics_effect.png\n3. confusion_matrix.png\n4. roc_curve.png")