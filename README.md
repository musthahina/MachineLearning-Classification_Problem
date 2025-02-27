# Breast Cancer Classification Models
This project demonstrates the implementation and comparison of several classification algorithms on the Breast Cancer dataset from the sklearn library. The objective is to evaluate the performance of different classification algorithms using metrics such as accuracy and classification reports.

## Table of Contents
#### Introduction
#### Dataset
#### Preprocessing
#### Classification Algorithms
#### Model Comparison

## **Introduction**
The goal of this project is to implement and evaluate the performance of five popular classification algorithms:

1.Logistic Regression \
2.Decision Tree Classifier\
3.Random Forest Classifier\
4.Support Vector Machine (SVM)\
5.k-Nearest Neighbors (k-NN)\
Each algorithm is applied to the Breast Cancer dataset, and the results are compared based on accuracy and classification reports.

## **Dataset**
This project uses the Breast Cancer dataset from sklearn. It contains features computed from breast cancer cell nuclei, which are used to predict whether a tumor is malignant or benign.

**Target:** The target variable (y) indicates whether a tumor is malignant (1) or benign (0).\
**Features:** The dataset contains 30 features that describe characteristics of the cell nuclei, such as radius, texture, smoothness, etc.
## **Preprocessing**

- **Loading the Data:** The dataset is loaded using the load_breast_cancer() function from sklearn.datasets.
- **Handling Missing Values:** While no missing values exist in this dataset, a check is performed for real-world readiness.
- **Duplicate Rows:** Identify and remove duplicates (if any) to avoid bias in the model.
- **Removing Outliers:** Outliers are detected using the IQR method and removed to improve model performance.
- **Skewness:** Highly skewed features are transformed (e.g., using yeo-johnson).
- **Feature Selection:** Highly correlated features are removed to prevent multicollinearity.
- **Feature Scaling:** StandardScaler was used to scale the features. This step is necessary because many machine learning models (like SVM and k-NN) are sensitive to the scale of input features.
- **Train-Test Split:** The data is split into training and testing sets (80% training, 20% testing).
## **Classification Algorithms**
The following classification algorithms were implemented:

**1. Logistic Regression**\
Logistic Regression is a linear model used for binary classification. It predicts the probability of a binary outcome using the logistic (sigmoid) function.

**2. Decision Tree Classifier**\
A Decision Tree is a non-linear model that splits the dataset into subsets based on feature values. It builds a tree-like structure for decision-making.

**3. Random Forest Classifier**\
Random Forest is an ensemble method that builds multiple decision trees and combines their predictions to improve accuracy and reduce overfitting.

**4. Support Vector Machine (SVM)**\
SVM is a powerful classification model that finds the optimal hyperplane to separate data points of different classes. It is effective for both linear and non-linear decision boundaries.

**5. k-Nearest Neighbors (k-NN)**\
k-NN is a simple algorithm that classifies a data point based on the majority class of its k nearest neighbors in the feature space.

## **Model Comparison**
After training and evaluating the models, their performance was compared using accuracy scores and classification reports. The **LogisticRegression** performed the best with an accuracy of **98%**, while the **Decision Tree Classifier** performed the worst with an accuracy of **93%**.

