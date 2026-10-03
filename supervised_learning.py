"""
YuVaintern Week 4 - Supervised Learning Model Implementation
Problem: Binary classification of breast-cancer diagnostic records.
Dataset: scikit-learn Breast Cancer Wisconsin (Diagnostic) dataset.

The script:
1. Loads the public dataset
2. Checks the data structure and class distribution
3. Splits data using stratification
4. Performs feature engineering by removing highly correlated features
5. Scales features and trains Logistic Regression
6. Evaluates the model with test metrics
7. Performs 5-fold stratified cross-validation
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report
)

# 1. Load public dataset
data = load_breast_cancer(as_frame=True)
X = data.data.copy()
y = data.target.copy()

print("Dataset shape:", X.shape)
print("Class distribution:")
print(y.value_counts())

# 2. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 3. Feature engineering
# Calculate correlations only on the training set.
# This avoids using information from the test set.
corr = X_train.corr().abs()
upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
drop_cols = [c for c in upper.columns if any(upper[c] > 0.90)]

X_train_fe = X_train.drop(columns=drop_cols)
X_test_fe = X_test.drop(columns=drop_cols)
X_fe = X.drop(columns=drop_cols)

print("\nRemoved highly correlated features:")
for col in drop_cols:
    print("-", col)

print("\nFeatures before:", X.shape[1])
print("Features after :", X_train_fe.shape[1])

# 4. Pipeline: scaling + Logistic Regression
model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=2000, random_state=42))
])

# 5. Train
model.fit(X_train_fe, y_train)

# 6. Test evaluation
y_pred = model.predict(X_test_fe)
y_prob = model.predict_proba(X_test_fe)[:, 1]

print("\nTest metrics")
print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall   :", recall_score(y_test, y_pred))
print("F1-score :", f1_score(y_test, y_pred))
print("ROC-AUC  :", roc_auc_score(y_test, y_prob))

print("\nConfusion matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification report:")
print(classification_report(y_test, y_pred, target_names=data.target_names))

# 7. Cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X_fe, y, cv=cv, scoring="accuracy")

print("\n5-fold CV accuracy:", cv_scores)
print("Mean CV accuracy  :", cv_scores.mean())
print("CV standard dev.  :", cv_scores.std())
