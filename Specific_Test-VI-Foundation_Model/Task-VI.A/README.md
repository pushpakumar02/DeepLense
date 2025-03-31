# Specific Test VI: Foundation Model (Task VI.A)

## Overview
This project implements a self-supervised Masked Autoencoder (MAE) and a classifier built upon its encoder. The approach involves pretraining the MAE on masked galaxy images followed by finetuning a classifier to distinguish between three classes: `no_sub`, `cdm`, and `axion`.

## Dataset
- **Dataset Directory**: `Dataset`
  - Each class (i.e., `no_sub`, `cdm`, `axion`) should have its own subdirectory containing `.npy` files.
- **Pretraining Mode**: Uses only the `no_sub` class files.
- **Finetuning Mode**: Uses files from all classes.

## Model Architecture
- **Masked Autoencoder (MAE)**:
  - **Encoder**: Three convolutional layers with Batch Normalization, ReLU activations, and Max Pooling.
  - **Decoder**: Two transposed convolutional layers with ReLU, followed by a final convolutional layer with Tanh activation.
- **Classifier**:
  - Built upon the MAE encoder.
  - Uses Adaptive Average Pooling, a linear layer with ReLU, dropout, and a final linear layer for 3-class classification.
  
## Training
1. **Pretraining MAE**:
   - The MAE is pretrained to reconstruct images from their masked versions using Mean Squared Error loss.
   - Pretrained weights are saved as `mae_pretrained.pth` for later use for Task b.
2. **Finetuning Classifier**:
   - The classifier is trained using Cross-Entropy Loss with class weights (giving higher weight to the `cdm` class).
   - The best performing classifier weights (based on validation accuracy) are saved as `best_classifier.pth`.
   - Training curves, ROC curve, and confusion matrix are generated and saved as images.

## Results
- **Pretrained MAE Weights**: `mae_pretrained.pth`
- **Best Classifier Weights**: `best_classifier.pth`
- **Visualizations**:
  - Best Test Accuracy: 
    ![Best Accuracy](results/terminal_accuracy.png)
  - ROC Curve:  
    ![ROC Curve](results/roc_curve.png)
  - Confusion Matrix:  
    ![Confusion Matrix](results/confusion_matrix.png)
  - Training Loss & Accuracy Curves:  
    ![Training Curves](results/training_curves.png)

## How to Run
1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`.
3. Run the model:
   - **Option 1**: Execute the Python script: `python model.py`.
   - **Option 2**: Run the Jupyter Notebook: `model.ipynb`.
