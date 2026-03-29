#!/usr/bin/env python
# coding: utf-8

# In[1]:


# =========================
# STEP 1: IMPORT LIBRARIES (SAFE)
# =========================

import pandas as pd  # for data manipulation using DataFrames
import numpy as np  # for numerical operations and arrays
import matplotlib.pyplot as plt  # for plotting graphs

# Safe seaborn import
try:
    import seaborn as sns  # for advanced data visualization
    USE_SEABORN = True  # flag to indicate seaborn is available
except:
    print("Seaborn not installed → using matplotlib fallback")  # notify if seaborn missing
    USE_SEABORN = False  # fallback to matplotlib

from sklearn.model_selection import train_test_split  # to split dataset into train and test
from sklearn.preprocessing import StandardScaler, LabelEncoder  # for scaling and encoding

from sklearn.tree import DecisionTreeClassifier  # decision tree model
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, StackingClassifier  # ensemble models

from sklearn.linear_model import LogisticRegression  # logistic regression model
from sklearn.neighbors import KNeighborsClassifier  # KNN model

from sklearn.metrics import (
    confusion_matrix, classification_report,  # evaluation metrics
    accuracy_score, precision_score, recall_score, f1_score  # performance scores
)


# =========================
# STEP 2: LOAD DATA
# =========================


df = pd.read_csv("student_exam_performance_dataset.csv") # load dataset from file path

print("Dataset Loaded:", df.shape)  # print number of rows and columns


# =========================
# STEP 3: PREPROCESSING
# =========================

if 'student_id' in df.columns:
    df = df.drop(columns=['student_id'])  # remove unnecessary ID column

le = LabelEncoder()  # create label encoder
df['pass_fail'] = le.fit_transform(df['pass_fail'])  # convert target labels to numeric

df = pd.get_dummies(df, drop_first=True)  # convert categorical variables to numeric
df = df.dropna()  # remove missing values


# =========================
# STEP 4: SPLIT DATA
# =========================

X = df.drop('pass_fail', axis=1)  # features (independent variables)
y = df['pass_fail']  # target variable

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y  # split data into 80% train, 20% test
)


# =========================
# STEP 5: SCALE DATA
# =========================

scaler = StandardScaler()  # initialize scaler
X_train = scaler.fit_transform(X_train)  # fit and scale training data
X_test = scaler.transform(X_test)  # scale test data using same scaler


# =========================
# STEP 6: TRAIN MODELS
# =========================

dt = DecisionTreeClassifier(random_state=42)  # initialize decision tree
rf = RandomForestClassifier(n_estimators=100, random_state=42)  # initialize random forest with 100 trees
ada = AdaBoostClassifier(n_estimators=100, random_state=42)  # initialize AdaBoost with 100 estimators

estimators = [
    ('dt', DecisionTreeClassifier()),  # base model 1 for stacking
    ('knn', KNeighborsClassifier())  # base model 2 for stacking
]

stack = StackingClassifier(
    estimators=estimators,  # base models
    final_estimator=LogisticRegression(max_iter=1000)  # meta-model to combine predictions
)

# Fit models
dt.fit(X_train, y_train)  # train decision tree
rf.fit(X_train, y_train)  # train random forest
ada.fit(X_train, y_train)  # train AdaBoost
stack.fit(X_train, y_train)  # train stacking model

# Predictions
y_pred_dt = dt.predict(X_test)  # predict using decision tree
y_pred_rf = rf.predict(X_test)  # predict using random forest
y_pred_ada = ada.predict(X_test)  # predict using AdaBoost
y_pred_stack = stack.predict(X_test)  # predict using stacking


# =========================
# STEP 7: EVALUATION
# =========================

models = {
    "Decision Tree": y_pred_dt,  # store DT predictions
    "Random Forest": y_pred_rf,  # store RF predictions
    "AdaBoost": y_pred_ada,  # store AdaBoost predictions
    "Stacking": y_pred_stack  # store stacking predictions
}

for name, preds in models.items():
    print(f"\n--- {name} ---")  # print model name
    print(confusion_matrix(y_test, preds))  # print confusion matrix
    print(classification_report(y_test, preds))  # print detailed metrics


# =========================
# STEP 8: MODEL COMPARISON
# =========================

results = []  # list to store results

for name, preds in models.items():
    results.append([
        name,  # model name
        accuracy_score(y_test, preds),  # accuracy
        precision_score(y_test, preds, zero_division=0),  # precision
        recall_score(y_test, preds),  # recall
        f1_score(y_test, preds)  # F1 score
    ])

results_df = pd.DataFrame(results, columns=[
    "Model", "Accuracy", "Precision", "Recall", "F1-score"  # column names
])

print("\n=== MODEL COMPARISON ===")
print(results_df)  # display comparison table


# =========================
# STEP 9: SAFE BEST MODEL (NO ERROR)
# =========================

if not results_df.empty:
    best_model = results_df.loc[results_df["F1-score"].idxmax()]  # select model with highest F1-score
else:
    best_model = None  # handle empty case


# =========================
# STEP 10: AUTOMATIC CONCLUSION
# =========================

if best_model is not None:
    print("\n" + "="*60)
    print("AUTOMATIC CONCLUSION")  # section title
    print("="*60)

    print(f"Best Model: {best_model['Model']}")  # print best model name
    print(f"Accuracy: {best_model['Accuracy']:.2f}")  # print accuracy
    print(f"Precision: {best_model['Precision']:.2f}")  # print precision
    print(f"Recall: {best_model['Recall']:.2f}")  # print recall
    print(f"F1-score: {best_model['F1-score']:.2f}")  # print F1 score

    print("\nInterpretation:")
    print("The selected model provides the best balance between precision and recall.")  # explanation
    print("Ensemble methods generally perform better due to combining multiple learners.")  # explanation
    print("="*60)


# =========================
# STEP 11: EXPORT RESULTS
# =========================

results_df.to_excel("model_results.xlsx", index=False)  # save results to Excel

# Safe report export
if best_model is not None:
    report_text = f"""
MODEL EVALUATION REPORT

Best Model: {best_model['Model']}

Accuracy: {best_model['Accuracy']:.2f}
Precision: {best_model['Precision']:.2f}
Recall: {best_model['Recall']:.2f}
F1-score: {best_model['F1-score']:.2f}

Conclusion:
The {best_model['Model']} model performed best overall.
"""

    with open("model_report.txt", "w") as f:
        f.write(report_text)  # write report to text file


# =========================
# STEP 12: VISUALIZATION (SAFE)
# =========================

fig, axes = plt.subplots(2, 2, figsize=(12,10))  # create subplot grid
axes = axes.flatten()  # flatten axes for easy iteration

for i, (name, preds) in enumerate(models.items()):
    cm = confusion_matrix(y_test, preds)  # compute confusion matrix

    if USE_SEABORN:
        sns.heatmap(cm, annot=True, fmt='d', ax=axes[i])  # plot heatmap with seaborn
    else:
        axes[i].imshow(cm)  # fallback plot using matplotlib

    axes[i].set_title(name)  # set plot title

plt.tight_layout()  # adjust layout
plt.show()  # display plots

