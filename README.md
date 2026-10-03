# YuVaintern Week 4 – Supervised Learning Model Implementation

## Project
**Breast Cancer Diagnostic Classification using Logistic Regression**

## Objective
Build and evaluate a supervised learning classification model using Python and scikit-learn.

## Dataset
Scikit-learn's public **Breast Cancer Wisconsin (Diagnostic)** dataset.

- Samples: 569
- Original features: 30
- Engineered features: 20
- Target: binary classification
- Train/test split: 80/20 with stratification

## Workflow
1. Load dataset
2. Inspect dimensions and class distribution
3. Split into training and testing sets
4. Remove highly correlated features (>0.90), using training data only
5. Standardize features
6. Train Logistic Regression
7. Evaluate using accuracy, precision, recall, F1-score and ROC-AUC
8. Validate with 5-fold stratified cross-validation

## Test Results
- Accuracy: 0.9825
- Precision: 0.9861
- Recall: 0.9861
- F1-score: 0.9861
- ROC-AUC: 0.9940
- Mean 5-fold CV accuracy: 0.9754
- CV standard deviation: 0.0129

## Important limitation
This is an educational machine-learning project. It is not a medical diagnostic system and should not be used for clinical decisions.

## Run
```bash
pip install -r requirements.txt
python supervised_learning.py
```
