# ST61-55890-2025-Decision-Tree-Algorithms-and-Ensemble-Methods

## 1. Student Details

### 1.1 Personal Information

* Name: Peter Mulupi
* Reg No: ST61/55890/2025

### 1.2 Course Information

* Course: CSA 821 Machine Learning

## 2. Project Overview

This project applies Decision Tree and Ensemble Methods (Random Forest, AdaBoost, and Stacking) to predict student performance using the Student Exam Performance dataset.

## 3. Problem Statement

The objective is to classify students into:

* Pass (1)
* Fail (0)

## 4. Methods Used

### 4.1 Decision Tree

#### Description

A simple and interpretable model used as a baseline for classification.

### 4.2 Random Forest (Bagging)

#### Description

An ensemble method that builds multiple decision trees and combines their outputs to improve accuracy and reduce overfitting.

### 4.3 AdaBoost (Boosting)

#### Description

A boosting technique that focuses on misclassified instances to improve model performance.

### 4.4 Stacking Classifier

#### Description

A combination of multiple models (Decision Tree, KNN) with Logistic Regression as a meta-model to enhance prediction performance.

## 5. Evaluation Metrics

### 5.1 Metrics Used

* Accuracy
* Precision
* Recall
* F1-score

### 5.2 Additional Evaluation

* Confusion Matrix

## 6. Key Findings

* Ensemble methods outperformed the Decision Tree model
* Random Forest reduced overfitting and improved accuracy
* AdaBoost improved classification by focusing on difficult cases
* Stacking achieved the best performance

## 7. Conclusion

Decision Trees are simple and interpretable but prone to overfitting. Ensemble methods provide better generalization and higher predictive performance. Stacking is the most effective approach for this task.

## 8. Files in Repository

### 8.1 Project Files

* model_training.ipynb → Main implementation
* model_training.py → Script version

### 8.2 Documentation

* README.md → Project documentation

