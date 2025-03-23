import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from torchvision import transforms
from tqdm import tqdm
import seaborn as sns


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else 'cpu')
print(f'Using device: {device}')

# Dataset class
class LensFindingDataset(Dataset):
    def __init__(self, lens_dir, nonlens_dir, transform=None):
        self.lens_dir = Path(lens_dir)
        self.nonlens_dir = Path(nonlens_dir)
        self.lens_files = list(self.lens_dir.rglob('*.npy'))
        self.nonlens_files = list(self.nonlens_dir.rglob('*.npy'))
        self.files = self.lens_files + self.nonlens_files
        self.labels = [1] * len(self.lens_files) + [0] * len(self.nonlens_files)  # 1 for lens, 0 for non-lens
        self.transform = transform
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        img = np.load(self.files[idx], allow_pickle=True)
        img = torch.tensor(img, dtype=torch.float32)  # Shape: (3, 64, 64)

        if self.transform:
            img = self.transform(img)
        
        return img.to(device), torch.tensor(self.labels[idx]).to(device)

# Data augmentation and normalization
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize for 3 channels
])

# Dataset paths
# project_root = Path('/Users/pushpakumar/Projects/GSoC25_DeepLense-Gravitational Lens Finding/Specific_Test-II-Binary_Classification/lens-finding-test')
# train_lens_dir = project_root / 'train_lenses'
# train_nonlens_dir = project_root / 'train_nonlenses'
# test_lens_dir = project_root / 'test_lenses'
# test_nonlens_dir = project_root / 'test_nonlenses'

# Dataset paths
project_root = Path("lens-finding-test")  # Relative path
train_lens_dir = project_root / 'train_lenses'
train_nonlens_dir = project_root / 'train_nonlenses'
test_lens_dir = project_root / 'test_lenses'
test_nonlens_dir = project_root / 'test_nonlenses'

train_data = LensFindingDataset(train_lens_dir, train_nonlens_dir, transform=transform)
test_data = LensFindingDataset(test_lens_dir, test_nonlens_dir, transform=transform)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)




# Debug: Print paths to verify
print(f"Train lenses directory: {train_lens_dir}")
print(f"Train non-lenses directory: {train_nonlens_dir}")
print(f"Test lenses directory: {test_lens_dir}")
print(f"Test non-lenses directory: {test_nonlens_dir}")

# Define the model
class LensFinder(nn.Module):
    def __init__(self):
        super(LensFinder, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)  # Input: 3 channels
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 1)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 8 * 8)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))  # Binary classification
        return x

model = LensFinder().to(device)

# Loss function and optimizer
criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Training and validation loop
def train_model(model, train_loader, test_loader, epochs=20):
    train_losses, test_losses = [], []
    best_test_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for images, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}'):
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs.squeeze(), labels.float())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Test evaluation
        model.eval()
        test_loss = 0.0
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                outputs = model(images)
                loss = criterion(outputs.squeeze(), labels.float())
                test_loss += loss.item()
                
                predicted = (outputs.squeeze() > 0.5).float()
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
        
        avg_test_loss = test_loss / len(test_loader)
        test_losses.append(avg_test_loss)
        accuracy = 100 * correct / total

        print(f'Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}, Test Accuracy: {accuracy:.2f}%')

        # Save the best model
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            torch.save(model.state_dict(), 'best_lens_finder_model.pth')

    return train_losses, test_losses

# Train the model
train_losses, test_losses = train_model(model, train_loader, test_loader, epochs=20)

# Plot training and test losses
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Evaluate model (ROC curve and AUC)
def evaluate_model(model, test_loader):
    model.eval()
    y_true, y_scores = [], []

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            y_scores.extend(outputs.squeeze().cpu().numpy())
            y_true.extend(labels.cpu().numpy())
    
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)

    fpr, tpr, _ = roc_curve(y_true, y_scores)
    auc_score = auc(fpr, tpr)

    plt.plot(fpr, tpr, label=f'AUC = {auc_score:.2f}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()

    print(f'AUC Score: {auc_score:.4f}')

evaluate_model(model, test_loader)

# Confusion matrix
def print_confusion_matrix(model, test_loader):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            predicted = (outputs.squeeze() > 0.5).float()
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
        
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

print_confusion_matrix(model, test_loader)