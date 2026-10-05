# Week 5 - Deep Learning Application in Data Science

## Project
Handwritten Digit Classification Using a PyTorch Neural Network

## Objective
Build and evaluate a neural network for multiclass classification of handwritten digits.

## Dataset
Scikit-learn Digits dataset (1797 samples, 64 pixel features, 10 classes).

## Model
- Input: 64 normalized pixel features
- Dense layer: 128 neurons
- ReLU activation
- Dropout: 20%
- Dense layer: 128 neurons
- ReLU activation
- Dropout: 20%
- Output: 10 classes
- Loss: Cross-Entropy Loss
- Optimizer: Adam
- Learning rate: 0.001
- Weight decay: 0.0001
- Batch size: 32
- Early stopping patience: 10

## Results
- Test Accuracy: 98.1481%
- Weighted Precision: 98.2206%
- Weighted Recall: 98.1481%
- Weighted F1-score: 98.1590%
- Best Validation Accuracy: 98.1481%
- Epochs Completed: 70

## Run
```bash
pip install -r requirements.txt
python deep_learning_digit_classifier.py
```

The report is provided as a DOCX file in the project package.
