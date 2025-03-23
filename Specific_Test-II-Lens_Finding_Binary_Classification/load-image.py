# to visualise the data/images from the .npy files
import numpy as np
import matplotlib.pyplot as plt
import os
import random

# Path to the train_lenses folder
train_lenses_folder = "lens-finding-test/train_lenses"

# List all .npy files in the folder
npy_files = [f for f in os.listdir(train_lenses_folder) if f.endswith('.npy')]

# Pick a random file
random_file = random.choice(npy_files)
file_path = os.path.join(train_lenses_folder, random_file)

# Load the .npy file
image = np.load(file_path)  # Shape: (3, 64, 64)

# Transpose the image to (64, 64, 3) for visualization
image = np.transpose(image, (1, 2, 0))

# Normalize the image to [0, 1] if necessary
if image.max() > 1:
    image = image / 255.0

# Display the image
plt.imshow(image)
plt.axis('off')  # Hide axes
plt.title(f"Visualizing: {random_file}")
plt.show()