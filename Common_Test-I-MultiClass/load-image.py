# Description: This script loads an image from a file and prints its shape.
# It also visualizes the image using matplotlib.
import numpy as np

file_path = "/Users/pushpakumar/Projects/GSoC25_DeepLense-Gravitational Lens Finding /Common_Test-I-MultiClass/dataset/train/no/9896.npy"

try:
    data = np.load(file_path, allow_pickle=True)
    print("File loaded successfully! Shape:", data.shape)
except Exception as e:
    print("Error loading file:", e)
