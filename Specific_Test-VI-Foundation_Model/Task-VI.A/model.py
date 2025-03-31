import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve
from tqdm import tqdm
import matplotlib.pyplot as plt
from itertools import cycle

# Device configuration
device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')

class_names = ['no_sub', 'cdm', 'axion']  # Global class names

class LensDataset(Dataset):
    def __init__(self, root_dir, class_names, mode='pretrain', mask_ratio=0.75):
        self.root_dir = Path(root_dir)
        self.mode = mode
        self.mask_ratio = mask_ratio
        self.target_size = (64, 64)
        
        self.file_paths = []
        self.labels = []
        
        for label_idx, class_name in enumerate(class_names):
            class_dir = self.root_dir / class_name
            files = [f for f in class_dir.glob('*.npy') if self._is_valid_file(f)]
            print(f"Found {len(files)} valid files in {class_name}")
            self.file_paths.extend(files)
            self.labels.extend([label_idx] * len(files))
            
        if mode == 'pretrain':
            self.file_paths = [f for f in self.file_paths if 'no_sub' in str(f).lower()]
            print(f"Using {len(self.file_paths)} no_sub files for pretraining")

    def _is_valid_file(self, file_path):
        """Updated validation to handle Axion's special format"""
        try:
            data = np.load(file_path, allow_pickle=True)
            if isinstance(data, np.ndarray):
                if data.dtype == object and data.shape == (2,):  # Axion format
                    return isinstance(data[0], np.ndarray) and data[0].shape == (64, 64)
                return data.dtype.kind in {'f', 'i', 'u', 'b'} and data.size > 0
            return False
        except:
            return False

    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        try:
            loaded = np.load(self.file_paths[idx], allow_pickle=True)
            
            # Handle Axion's special format
            if loaded.dtype == object and loaded.shape == (2,):
                img = loaded[0]  # Extract the image array
            else:
                img = loaded
                
            img = torch.tensor(img, dtype=torch.float32)
            
            # Convert to 3 channels if needed
            if img.ndim == 2:
                img = img.unsqueeze(0).repeat(3, 1, 1)
            elif img.shape[0] == 1:
                img = img.repeat(3, 1, 1)
            
            # Resize if needed
            if img.shape[1:] != self.target_size:
                img = F.interpolate(img.unsqueeze(0), size=self.target_size, mode='bilinear').squeeze(0)
            
            if self.mode == 'pretrain':
                _, H, W = img.shape
                num_pixels = H * W
                mask = torch.zeros(num_pixels)
                mask[:int(num_pixels * self.mask_ratio)] = 1
                mask = mask[torch.randperm(num_pixels)].reshape(1, H, W)
                masked_img = img * (1 - mask)
                return masked_img, img, mask
            else:
                return img, torch.tensor(self.labels[idx])
            
        except Exception as e:
            print(f"Error loading {self.file_paths[idx]}: {e}")
            return self[np.random.randint(0, len(self))]

class MAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 3, 3, padding=1),
            nn.Tanh()
        )
    
    def forward(self, x):
        return self.decoder(self.encoder(x))

class Classifier(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 3)
        )
    
    def forward(self, x):
        return self.classifier(self.encoder(x))

def train_mae(model, train_loader, epochs=5):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    for epoch in range(epochs):
        total_loss = 0
        for masked_imgs, target_imgs, _ in tqdm(train_loader, desc=f'Pretrain Epoch {epoch+1}/{epochs}'):
            masked_imgs = masked_imgs.to(device)
            target_imgs = target_imgs.to(device)
            
            optimizer.zero_grad()
            outputs = model(masked_imgs)
            loss = criterion(outputs, target_imgs)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1} Loss: {total_loss/len(train_loader):.4f}")

def train_classifier(model, train_loader, test_loader, epochs=10):
    model.train()
    # Class weights: Higher weight for cdm (class 1)
    class_weights = torch.tensor([1.0, 2.0, 1.0]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    best_accuracy = 0.0
    train_losses = []
    train_accuracies = []
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for images, labels in tqdm(train_loader, desc=f'Finetune Epoch {epoch+1}/{epochs}'):
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_loss = total_loss / len(train_loader)
        epoch_accuracy = correct / total
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_accuracy)
        
        # Validation
        val_accuracy = evaluate_model(model, test_loader, verbose=False)
        scheduler.step(val_accuracy)
        
        print(f"Epoch {epoch+1}: Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_accuracy:.4f}, Val Acc: {val_accuracy:.4f}")
        
        # Save best model
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), 'best_classifier.pth')
            print(f"New best model saved with accuracy {best_accuracy:.4f}")
    
    # Plot training curves
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
    plt.close()

def evaluate_model(model, test_loader, verbose=True):
    model.eval()
    all_labels = []
    all_probs = []
    all_preds = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            
            all_labels.append(labels.cpu())
            all_probs.append(probs.cpu())
            all_preds.append(outputs.argmax(dim=1).cpu())
    
    all_labels = torch.cat(all_labels).numpy()
    all_probs = torch.cat(all_probs).numpy()
    all_preds = torch.cat(all_preds).numpy()
    
    # ROC AUC and Curve
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    n_classes = all_probs.shape[1]
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve((all_labels == i).astype(int), all_probs[:, i])
        roc_auc[i] = roc_auc_score((all_labels == i).astype(int), all_probs[:, i])
    
    # Plot ROC curves
    plt.figure(figsize=(10, 8))
    colors = cycle(['blue', 'red', 'green'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'ROC curve of {class_names[i]} (AUC = {roc_auc[i]:0.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multi-class ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig('roc_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Confusion Matrix with values
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    
    tick_marks = np.arange(n_classes)
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Calculate metrics
    accuracy = np.mean(all_preds == all_labels)
    avg_auc = np.mean(list(roc_auc.values()))
    
    if verbose:
        print("\nEvaluation Metrics:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Average AUC: {avg_auc:.4f}")
        for i in range(n_classes):
            print(f"{class_names[i]} AUC: {roc_auc[i]:.4f}")
        print("\nConfusion Matrix:")
        print(cm)
    
    return accuracy

if __name__ == '__main__':
    print(f"Using device: {device}")
    
    # Dataset setup
    data_dir = Path("Dataset")
    
    # Pretrain MAE
    print("\nPretraining MAE...")
    pretrain_dataset = LensDataset(data_dir, class_names, 'pretrain')
    pretrain_loader = DataLoader(pretrain_dataset, batch_size=32, shuffle=True, num_workers=0)
    
    mae = MAE().to(device)
    train_mae(mae, pretrain_loader, epochs=5)
    
    # Save the pretrained weights for Task VI.B
    torch.save(mae.state_dict(), 'mae_pretrained.pth')
    print("\nSaved pretrained MAE weights to mae_pretrained.pth")
    
    # Finetune Classifier
    print("\nFinetuning Classifier...")
    full_dataset = LensDataset(data_dir, class_names, 'finetune')
    train_size = int(0.8 * len(full_dataset))
    train_dataset, test_dataset = random_split(full_dataset, [train_size, len(full_dataset)-train_size])
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=32, num_workers=0)
    
    classifier = Classifier(mae.encoder).to(device)
    train_classifier(classifier, train_loader, test_loader, epochs=10)
    
    # Final Evaluation
    print("\nFinal Evaluation:")
    evaluate_model(classifier, test_loader)
    
    print("\nTraining complete! Saved the following files:")
    print("- mae_pretrained.pth (pretrained MAE weights for Task VI.B)")
    print("- best_classifier.pth (best classifier weights)")
    print("- roc_curve.png (ROC curve visualization)")
    print("- confusion_matrix.png (confusion matrix with values)")
    print("- training_curves.png (training loss and accuracy curves)")


