from pathlib import Path
import numpy as np

def debug_axion_files():
    axion_dir = Path("Dataset/axion")
    sample_files = list(axion_dir.glob("*.npy"))[:3]  # Just check 3 files now
    
    for file in sample_files:
        print("\n" + "="*80)
        print(f"DEBUGGING FILE: {file.name}")
        data = np.load(file, allow_pickle=True)
        
        print("\nElement 0 Contents:")
        print(f"Type: {type(data[0])}")
        print(f"Shape: {data[0].shape if isinstance(data[0], np.ndarray) else 'N/A'}")
        print(f"Dtype: {data[0].dtype if isinstance(data[0], np.ndarray) else 'N/A'}")
        print("First 5 elements:" if isinstance(data[0], np.ndarray) else "Content:")
        print(data[0][:5] if isinstance(data[0], np.ndarray) else data[0])
        
        print("\nElement 1 Contents:")
        print(data[1])

debug_axion_files()