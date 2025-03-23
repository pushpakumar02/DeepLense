# phase3
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from torchvision import transforms, models
from tqdm import tqdm

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else 'cpu')
print(f'Using device: {device}')

# Dataset class
class LensDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.files = list(self.root_dir.rglob('*.npy'))
        self.labels = [self.get_label(f) for f in self.files]
        self.transform = transform
    
    def get_label(self, file):
        if 'vort' in str(file):
            return 0
        elif 'sphere' in str(file):
            return 1
        else:
            return 2

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        img = np.load(self.files[idx], allow_pickle=True)
        img = torch.tensor(img, dtype=torch.float32)

        if self.transform:
            img = self.transform(img)
        
        return img.to(device), torch.tensor(self.labels[idx]).to(device)

# Enhanced data augmentation and normalization
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.RandomResizedCrop((150, 150), scale=(0.7, 1.0)),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Dataset paths
project_root = Path('/Users/pushpakumar/Projects/GSoC25_DeepLense-Gravitational Lens Finding /Common_Test-I-MultiClass')
train_data = LensDataset(project_root / 'dataset'/'train', transform=transform)
val_data = LensDataset(project_root / 'dataset'/'val', transform=transform)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
val_loader = DataLoader(val_data, batch_size=64, shuffle=False)

# Use ResNet18 with modified first layer for single-channel input
model = models.resnet18(weights=None)  # Do not download weights

# Modify the first convolutional layer for 1-channel input
model.conv1 = nn.Conv2d(1, 128, kernel_size=7, stride=2, padding=3, bias=False)  # Increased filters

# Modify the corresponding batch normalization layer
model.bn1 = nn.BatchNorm2d(128)

# Modify the first residual block (layer1) to accept 128 input channels
model.layer1[0].conv1 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=False)
model.layer1[0].bn1 = nn.BatchNorm2d(64)

# Update the downsample layer in the first residual block to handle 128 input channels
model.layer1[0].downsample = nn.Sequential(
    nn.Conv2d(128, 64, kernel_size=1, stride=1, bias=False),
    nn.BatchNorm2d(64)
)

# Modify the final fully connected layer for 3-class output
model.fc = nn.Linear(model.fc.in_features, 3)

# Load weights (excluding incompatible layers)
weights_path = "/Users/pushpakumar/Downloads/resnet18-f37072fd.pth"
pretrained_dict = torch.load(weights_path)
model_dict = model.state_dict()

# Filter out incompatible keys
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.size() == model_dict[k].size()}

# Update the model's state dict
model_dict.update(pretrained_dict)
model.load_state_dict(model_dict)

model = model.to(device)

# Loss function with adjusted class weights
class_weights = torch.tensor([1.0, 2.0, 1.0]).to(device)  # Adjusted for class imbalance
criterion = nn.CrossEntropyLoss(weight=class_weights)

# Optimizer and learning rate scheduler
optimizer = optim.Adam(model.parameters(), lr=0.0001)  # Increased learning rate
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# Early stopping with increased patience
early_stopping_patience = 5  # Increased from 3
best_val_loss = float('inf')
patience_counter = 0

# Training and validation loop
def train_model(model, train_loader, val_loader, epochs=20):
    train_losses, val_losses = [], []

    # Declare global variables
    global best_val_loss, patience_counter

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for images, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}'):
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        accuracy = 100 * correct / total

        print(f'Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Accuracy: {accuracy:.2f}%')

        # Save checkpoint after each epoch
        torch.save(model.state_dict(), f'checkpoint_epoch_{epoch+1}.pth')

        # Save the best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print("Early stopping triggered!")
                break

        # Learning rate scheduler step
        scheduler.step()

    return train_losses, val_losses

# Train the model
train_losses, val_losses = train_model(model, train_loader, val_loader, epochs=20)

# Plot training and validation losses
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Evaluate model (ROC curve and AUC)
def evaluate_model(model, val_loader):
    model.eval()
    y_true, y_scores = [], [] 

    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            y_scores.extend(probs)
            y_true.extend(labels.cpu().numpy())
    
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)

    for i in range(3):
        fpr, tpr, _ = roc_curve(y_true == i, y_scores[:, i])
        auc_score = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'Class {i} (AUC = {auc_score:.2f})')
    
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()

evaluate_model(model, val_loader)

# Confusion matrix
def print_confusion_matrix(model, val_loader):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
        
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

print_confusion_matrix(model, val_loader)

# Sample image visualization
def show_sample_images(dataset, num_samples=5):
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 5))
    for i in range(num_samples):
        img, label = dataset[i]
        img = img.squeeze().cpu().numpy()
        img = (img + 1) / 2  # Change range from [-1, 1] to [0, 1]
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f'Label: {label}')
        axes[i].axis('off')
    plt.show()

show_sample_images(train_data)







# phase2
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.metrics import roc_curve, auc, confusion_matrix
# from torch.utils.data import DataLoader, Dataset
# from pathlib import Path
# from torchvision import transforms, models
# from tqdm import tqdm

# # Device configuration
# device = torch.device('cuda' if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else 'cpu')
# print(f'Using device: {device}')

# # Dataset class
# class LensDataset(Dataset):
#     def __init__(self, root_dir, transform=None):
#         self.root_dir = Path(root_dir)
#         self.files = list(self.root_dir.rglob('*.npy'))
#         self.labels = [self.get_label(f) for f in self.files]
#         self.transform = transform
    
#     def get_label(self, file):
#         if 'vort' in str(file):
#             return 0
#         elif 'sphere' in str(file):
#             return 1
#         else:
#             return 2

#     def __len__(self):
#         return len(self.files)
    
#     def __getitem__(self, idx):
#         img = np.load(self.files[idx], allow_pickle=True)
#         img = torch.tensor(img, dtype=torch.float32) # Add channel dimension

#         if self.transform:
#             img = self.transform(img)
        
#         return img.to(device), torch.tensor(self.labels[idx]).to(device)

# # Data augmentation and normalization
# transform = transforms.Compose([
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomRotation(10),
#     transforms.RandomResizedCrop((150, 150), scale=(0.8, 1.0)),
#     transforms.ColorJitter(brightness=0.2, contrast=0.2),
#     transforms.Normalize(mean=[0.5], std=[0.5])
# ])

# # Dataset paths
# project_root = Path("/Users/pushpakumar/Projects/GSOC2025-Gravitational Lens Finding")
# train_data = LensDataset(project_root / 'dataset'/'train', transform=transform)
# val_data = LensDataset(project_root / 'dataset'/'val', transform=transform)

# train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
# val_loader = DataLoader(val_data, batch_size=64, shuffle=False)

# # Use ResNet18 with modified first layer for single-channel input
# model = models.resnet18(weights=None)  # Do not download weights

# # Modify the first convolutional layer for 1-channel input
# model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

# # Modify the final fully connected layer for 3-class output
# model.fc = nn.Linear(model.fc.in_features, 3)

# # Load weights (excluding incompatible layers)
# weights_path = "/Users/pushpakumar/Downloads/resnet18-f37072fd.pth"
# pretrained_dict = torch.load(weights_path)
# model_dict = model.state_dict()

# # Filter out incompatible keys
# pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.size() == model_dict[k].size()}

# # Update the model's state dict
# model_dict.update(pretrained_dict)
# model.load_state_dict(model_dict)

# model = model.to(device)

# # Loss function with class weights
# class_weights = torch.tensor([1.0, 1.0, 1.0]).to(device)  # Adjust based on class distribution
# criterion = nn.CrossEntropyLoss(weight=class_weights)

# # Optimizer and learning rate scheduler
# optimizer = optim.Adam(model.parameters(), lr=0.00001)
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# # Early stopping
# early_stopping_patience = 3
# best_val_loss = float('inf')
# patience_counter = 0

# # Training and validation loop
# def train_model(model, train_loader, val_loader, epochs=20):
#     train_losses, val_losses = [], []

#     # Declare global variables
#     global best_val_loss, patience_counter

#     for epoch in range(epochs):
#         model.train()
#         running_loss = 0.0
        
#         for images, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}'):
#             optimizer.zero_grad()
#             outputs = model(images)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()
#             running_loss += loss.item()

#         avg_train_loss = running_loss / len(train_loader)
#         train_losses.append(avg_train_loss)

#         # Validation
#         model.eval()
#         val_loss = 0.0
#         correct, total = 0, 0
#         with torch.no_grad():
#             for images, labels in val_loader:
#                 outputs = model(images)
#                 loss = criterion(outputs, labels)
#                 val_loss += loss.item()
                
#                 _, predicted = torch.max(outputs, 1)
#                 correct += (predicted == labels).sum().item()
#                 total += labels.size(0)
        
#         avg_val_loss = val_loss / len(val_loader)
#         val_losses.append(avg_val_loss)
#         accuracy = 100 * correct / total

#         print(f'Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Accuracy: {accuracy:.2f}%')

#         # Save checkpoint after each epoch
#         torch.save(model.state_dict(), f'checkpoint_epoch_{epoch+1}.pth')

#         # Save the best model
#         if avg_val_loss < best_val_loss:
#             best_val_loss = avg_val_loss
#             patience_counter = 0
#             torch.save(model.state_dict(), 'best_model.pth')
#         else:
#             patience_counter += 1
#             if patience_counter >= early_stopping_patience:
#                 print("Early stopping triggered!")
#                 break

#         # Learning rate scheduler step
#         scheduler.step()

#     return train_losses, val_losses

# model.load_state_dict(torch.load('best_model.pth'))

# # Train the model
# train_losses, val_losses = train_model(model, train_loader, val_loader, epochs=20)

# # Plot training and validation losses
# plt.plot(train_losses, label='Train Loss')
# plt.plot(val_losses, label='Validation Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()

# # Evaluate model (ROC curve and AUC)
# def evaluate_model(model, val_loader):
#     model.eval()
#     y_true, y_scores = [], [] 

#     with torch.no_grad():
#         for images, labels in val_loader:
#             outputs = model(images)
#             probs = torch.softmax(outputs, dim=1).cpu().numpy()
#             y_scores.extend(probs)
#             y_true.extend(labels.cpu().numpy())
    
#     y_true = np.array(y_true)
#     y_scores = np.array(y_scores)

#     for i in range(3):
#         fpr, tpr, _ = roc_curve(y_true == i, y_scores[:, i])
#         auc_score = auc(fpr, tpr)
#         plt.plot(fpr, tpr, label=f'Class {i} (AUC = {auc_score:.2f})')
    
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title('ROC Curve')
#     plt.legend()
#     plt.show()

# evaluate_model(model, val_loader)

# # Confusion matrix
# def print_confusion_matrix(model, val_loader):
#     model.eval()
#     y_true, y_pred = [], []
#     with torch.no_grad():
#         for images, labels in val_loader:
#             outputs = model(images)
#             _, predicted = torch.max(outputs, 1)
#             y_true.extend(labels.cpu().numpy())
#             y_pred.extend(predicted.cpu().numpy())
        
#     cm = confusion_matrix(y_true, y_pred)
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
#     plt.xlabel('Predicted')
#     plt.ylabel('True')
#     plt.show()

# print_confusion_matrix(model, val_loader)

# # Sample image visualization
# def show_sample_images(dataset, num_samples=5):
#     fig, axes = plt.subplots(1, num_samples, figsize=(15, 5))
#     for i in range(num_samples):
#         img, label = dataset[i]
#         img = img.squeeze().cpu().numpy()
#         img = (img + 1) / 2  # Change range from [-1, 1] to [0, 1]
#         axes[i].imshow(img, cmap='gray')
#         axes[i].set_title(f'Label: {label}')
#         axes[i].axis('off')
#     plt.show()

# show_sample_images(train_data)











# phase1
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.metrics import roc_curve, auc
# from torch.utils.data import DataLoader, Dataset
# from pathlib import Path
# from torchvision import transforms
# from tqdm import tqdm

# #device
# device = torch.device('cuda' if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else 'cpu')
# print(f'Using device: {device}')

# #dataset
# class LensDataset(Dataset):
#     def __init__(self, root_dir, transform=None):
#         self.root_dir = Path(root_dir)
#         self.files = list(self.root_dir.rglob('*.npy'))
#         self.labels = [self.get_label(f) for f in self.files]
#         self.transform = transform
    
#     def get_label(self, file):
#         if 'vort' in str(file):
#             return 0
#         elif 'sphere' in str(file):
#             return 1
#         else:
#             return 2

#     def __len__(self):
#         return len(self.files)
    
#     def __getitem__(self, idx):
#         img = np.load(self.files[idx], allow_pickle=True)
#         img = torch.tensor(img, dtype=torch.float32)

#         if self.transform:
#             img = self.transform(img)
        
#         return img.to(device), torch.tensor(self.labels[idx]).to(device)

# transform = transforms.Compose([  
#     transforms.Normalize(mean=[0.5], std=[0.5])
# ])

# #dataset path
# project_root = Path("/Users/pushpakumar/Projects/GSOC2025-Gravitational Lens Finding")
# train_data = LensDataset(project_root / 'dataset'/'train', transform=transform)
# val_data = LensDataset(project_root / 'dataset'/'val', transform=transform)

# train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
# val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

# #CNN model
# class CNN(nn.Module):
#     def __init__(self):
#         super(CNN, self).__init__()
#         self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
#         self.pool = nn.MaxPool2d(2, 2)

#         self.flatten_size = self._get_conv_output((1, 150, 150))

#         self.fc1 = nn.Linear(self.flatten_size, 128)
#         self.fc2 = nn.Linear(128, 3)

#     def _get_conv_output(self, shape):
#         """ Helper function to determine the size after convolution layers """
#         x = torch.zeros(1, *shape)  # Dummy input tensor
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         return x.numel()

#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = x.view(x.size(0), -1)  # Dynamically flatten
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x

# model = CNN().to(device)

# criterian = nn.CrossEntropyLoss()   #loss function
# optimizer = optim.Adam(model.parameters(), lr=0.0001)   #optimizer

# #training and validation loop
# def train_model(model, train_loader, val_loader, epochs=10):
#     train_losses, val_losses = [], []
#     best_val_loss = float('inf')

#     for epoch in range(epochs):
#         model.train()
#         running_loss = 0.0
        
#         for images, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}'):
#             optimizer.zero_grad()
#             outputs = model(images)
#             loss = criterian(outputs, labels)
#             loss.backward()
#             optimizer.step()
#             running_loss += loss.item()

#         avg_train_loss = running_loss / len(train_loader)
#         train_losses.append(avg_train_loss)

#         #validation
#         model.eval()
#         val_loss = 0.0
#         correct, total = 0, 0
#         with torch.no_grad():
#             for images, labels in val_loader:
#                 outputs = model(images)
#                 loss = criterian(outputs, labels)
#                 val_loss += loss.item()
                
#                 _, predicted = torch.max(outputs, 1)
#                 correct += (predicted == labels).sum().item()
#                 total += labels.size(0)
        
#         avg_val_loss = val_loss / len(val_loader)
#         val_losses.append(avg_val_loss)
#         accuracy = 100*correct/total

#         print(f'Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Accuracy: {accuracy:.2f}%')

#         #save best model
#         if avg_val_loss < best_val_loss:
#             best_val_loss = avg_val_loss
#             torch.save(model.state_dict(), 'best_model.pth')

#     return train_losses, val_losses

# #train model
# train_losses, val_losses = train_model(model, train_loader, val_loader, epochs=10)

# #evaluate model (ROC curve and AUC)
# def evaluate_model(model, val_loader):
#     model.eval()
#     y_true, y_scores = [], [] 

#     with torch.no_grad():
#         for images, labels in val_loader:
#             outputs = model(images)
#             probs = torch.softmax(outputs, dim=1).cpu().numpy()
#             y_scores.extend(probs)
#             y_true.extend(labels.cpu().numpy())
    
#     y_true = np.array(y_true)
#     y_scores = np.array(y_scores)

#     for i in range(3):
#         fpr, tpr, _ = roc_curve(y_true == i, y_scores[:, i])
#         auc_score = auc(fpr, tpr)
#         plt.plot(fpr, tpr, label=f'Class {i} (AUC = {auc_score:.2f})')
    
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title('ROC Curve')
#     plt.legend()
#     plt.show()

# evaluate_model(model, val_loader)

# #sample image visualization
# def show_sample_images(dataset, num_samples=5):
#     fig, axes = plt.subplots(1, num_samples, figsize=(15, 5))
#     for i in range(num_samples):
#         img, label = dataset[i]
#         img = img.squeeze().cpu().numpy()
#         img = (img + 1) / 2  # Change range from [-1, 1] to [0, 1]
#         axes[i].imshow(img, cmap='gray')
#         axes[i].set_title(f'Label: {label}')
#         axes[i].axis('off')
#     plt.show()

# show_sample_images(train_data)


