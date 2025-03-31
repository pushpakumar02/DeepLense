import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def inspect_dataset_samples():
    base_dir = Path("Dataset")
    lr_dir = base_dir/"LR"
    hr_dir = base_dir/"HR"
    
    lr_files = sorted(lr_dir.glob("*.npy"))[:3]
    hr_files = sorted(hr_dir.glob("*.npy"))[:3]
    
    print(f"Found {len(lr_files)} LR and {len(hr_files)} HR samples")
    
    for lr_path, hr_path in zip(lr_files, hr_files):
        print("\n" + "="*80)
        print(f"File Pair: {lr_path.name} (LR) | {hr_path.name} (HR)")
        
        lr_img = np.load(lr_path).squeeze()  # Remove single-dimensional entries
        hr_img = np.load(hr_path).squeeze()
        
        print(f"LR Shape: {lr_img.shape} | Dtype: {lr_img.dtype} | Range: [{lr_img.min():.2f}, {lr_img.max():.2f}]")
        print(f"HR Shape: {hr_img.shape} | Dtype: {hr_img.dtype} | Range: [{hr_img.min():.2f}, {hr_img.max():.2f}]")
        
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.imshow(lr_img, cmap='gray', vmin=0, vmax=1)
        plt.title(f"LR: {lr_img.shape}")
        plt.colorbar()
        
        plt.subplot(1, 2, 2)
        plt.imshow(hr_img, cmap='gray', vmin=0, vmax=1)
        plt.title(f"HR: {lr_img.shape} → {hr_img.shape}")
        plt.colorbar()
        
        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    inspect_dataset_samples()