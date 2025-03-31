import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import os
import sys

# Device configuration
device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')

class SRDataset(Dataset):
    def __init__(self, lr_dir, hr_dir):
        self.lr_dir = Path(lr_dir)
        self.hr_dir = Path(hr_dir)
        self.file_names = sorted([f.name for f in self.lr_dir.glob('*.npy')])
        
    def __len__(self):
        return len(self.file_names)
    
    def __getitem__(self, idx):
        lr = np.load(self.lr_dir/self.file_names[idx]).squeeze().astype(np.float32)
        hr = np.load(self.hr_dir/self.file_names[idx]).squeeze().astype(np.float32)
        return (
            torch.from_numpy(lr).unsqueeze(0).float(),
            torch.from_numpy(hr).unsqueeze(0).float()
        )

class SuperResolutionNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder (matches MAE exactly)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, stride=1, padding=1)
        )
        
        # Decoder with proper upsampling
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(size=(150, 150), mode='bilinear', align_corners=True),
            nn.Conv2d(64, 1, 3, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        features = self.encoder(x)
        return self.decoder(features)
    
    def load_pretrained_weights(self, path):
        """Load MAE encoder weights with proper channel conversion"""
        try:
            mae_weights = torch.load(path, map_location=device)
            model_dict = self.state_dict()
            
            # First conv layer (3-channel -> 1-channel)
            if 'encoder.0.weight' in mae_weights:
                model_dict['encoder.0.weight'] = mae_weights['encoder.0.weight'].mean(dim=1, keepdim=True)
                model_dict['encoder.0.bias'] = mae_weights['encoder.0.bias']
                loaded = 2
            else:
                raise RuntimeError("Missing first conv layer")
            
            # Transfer remaining compatible weights
            weight_map = {
                'encoder.1.weight': 'encoder.1.weight',
                'encoder.1.bias': 'encoder.1.bias',
                'encoder.4.weight': 'encoder.4.weight',
                'encoder.4.bias': 'encoder.4.bias',
                'encoder.5.weight': 'encoder.5.weight',
                'encoder.5.bias': 'encoder.5.bias'
            }
            
            for mae_name, sr_name in weight_map.items():
                if mae_name in mae_weights and sr_name in model_dict:
                    model_dict[sr_name] = mae_weights[mae_name]
                    loaded += 1
            
            if loaded < 6:
                raise RuntimeError(f"Only {loaded}/8 weights loaded")
                
            self.load_state_dict(model_dict, strict=False)
            print(f"✅ Successfully loaded {loaded} pretrained layers")
            return True
            
        except Exception as e:
            print(f"Weight loading failed: {str(e)}")
            sys.exit(1)

def train():
    # Create output directory (clears previous best images)
    os.makedirs('results/best_comparisons', exist_ok=True)
    for f in os.listdir('results/best_comparisons'):
        os.remove(os.path.join('results/best_comparisons', f))
    
    # Dataset setup
    dataset = SRDataset('Dataset/LR', 'Dataset/HR')
    train_size = int(0.9 * len(dataset))
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, len(dataset)-train_size])
    
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=16)
    
    # Model initialization
    model = SuperResolutionNet().to(device)
    model.load_pretrained_weights('../Task-VI.A/mae_pretrained.pth')
    
    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.L1Loss()
    
    best_psnr = 0
    no_improve = 0
    patience = 5
    
    for epoch in range(30):
        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
            
        # Training
        model.train()
        epoch_loss = 0
        for lr, hr in tqdm(train_loader, desc=f'Epoch {epoch+1}/30'):
            lr, hr = lr.to(device), hr.to(device)
            
            optimizer.zero_grad()
            sr = model(lr)
            loss = criterion(sr, hr)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        # Validation
        val_metrics = evaluate(model, val_loader)
        
        print(f"\nEpoch {epoch+1}:")
        print(f"Train Loss: {epoch_loss/len(train_loader):.4f}")
        print(f"MSE: {val_metrics['mse']:.6f} | PSNR: {val_metrics['psnr']:.2f} dB | SSIM: {val_metrics['ssim']:.4f}")
        
        if val_metrics['psnr'] > best_psnr:
            best_psnr = val_metrics['psnr']
            no_improve = 0
            torch.save(model.state_dict(), 'best_sr_model.pth')
            print(f"New best model (PSNR: {best_psnr:.2f} dB)")
            
            # Generate new best comparison (replaces previous)
            generate_best_comparison(model, val_loader, epoch+1)
        else:
            no_improve += 1
            print(f"No improvement ({no_improve}/{patience})")
    
    print("\nTraining complete! Best model saved as 'best_sr_model.pth'")
    print(f"Best comparison images saved in 'results/best_comparisons/' (PSNR: {best_psnr:.2f} dB)")

def evaluate(model, loader):
    model.eval()
    metrics = {'mse': 0, 'psnr': 0, 'ssim': 0}
    with torch.no_grad():
        for lr, hr in loader:
            lr, hr = lr.to(device), hr.to(device)
            sr = model(lr)
            
            # Ensure output matches target size
            if sr.shape[-2:] != hr.shape[-2:]:
                sr = F.interpolate(sr, size=hr.shape[-2:], mode='bilinear', align_corners=True)
            
            # Calculate metrics
            mse = F.mse_loss(sr, hr)
            metrics['mse'] += mse.item() * lr.size(0)
            psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
            metrics['psnr'] += psnr.item() * lr.size(0)
            
            for i in range(sr.shape[0]):
                metrics['ssim'] += ssim(
                    sr[i].cpu().numpy().squeeze(),
                    hr[i].cpu().numpy().squeeze(),
                    data_range=1.0
                )
    return {k: v/len(loader.dataset) for k,v in metrics.items()}

def generate_best_comparison(model, loader, epoch, num_samples=3):
    """Generates and saves ONLY the current best model comparisons"""
    model.eval()
    os.makedirs('results/best_comparisons', exist_ok=True)
    
    # Clear previous best images
    for f in os.listdir('results/best_comparisons'):
        os.remove(os.path.join('results/best_comparisons', f))
    
    with torch.no_grad():
        for i, (lr, hr) in enumerate(loader):
            if i >= num_samples: break
            lr, hr = lr.to(device), hr.to(device)
            sr = model(lr)
            
            if sr.shape[-2:] != hr.shape[-2:]:
                sr = F.interpolate(sr, size=hr.shape[-2:], mode='bilinear', align_corners=True)
            
            # Calculate metrics for this sample
            mse = F.mse_loss(sr, hr).item()
            psnr = 20 * np.log10(1.0 / np.sqrt(mse))
            ssim_val = ssim(
                sr[0].cpu().numpy().squeeze(),
                hr[0].cpu().numpy().squeeze(),
                data_range=1.0
            )
            
            # Create comparison figure
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            
            # Low Res
            axes[0].imshow(lr[0].cpu().numpy().squeeze(), cmap='gray')
            axes[0].set_title(f'Low Resolution\n{lr.shape[-2]}x{lr.shape[-1]}')
            axes[0].axis('off')
            
            # Super-Resolved
            axes[1].imshow(sr[0].cpu().numpy().squeeze(), cmap='gray')
            axes[1].set_title(f'Super-Resolved\nPSNR: {psnr:.2f} dB | SSIM: {ssim_val:.4f}')
            axes[1].axis('off')
            
            # High Res
            axes[2].imshow(hr[0].cpu().numpy().squeeze(), cmap='gray')
            axes[2].set_title(f'High Resolution\n{hr.shape[-2]}x{hr.shape[-1]} (Ground Truth)')
            axes[2].axis('off')
            
            plt.tight_layout()
            plt.savefig(f'results/best_comparisons/best_epoch_{epoch}_sample_{i}.png', bbox_inches='tight', dpi=300)
            plt.close()

if __name__ == '__main__':
    train()



