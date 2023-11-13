# Disease-Prediction-with-Machine-Learning
## Overview
This repository contains the code for a machine learning model that predicts diseases based on various symptoms. The model is built using the Random Forest classifier and is trained on a dataset containing information about symptoms and corresponding diseases.

## Table of Contents
* Overview
* Dataset
* Dependencies
* Model Training
* Hyperparameter Optimization
* Model Evaluation
* SHAP Analysis

### Dataset
The dataset used for training and testing the model is available from Kaggle:
https://www.kaggle.com/datasets/kaushil268/disease-prediction-using-machine-learning/data

### Dependencies
* Python 3.x
* pandas
* scikit-learn
* SHAP
* Matplotlib
* Jupyter Notebook (optional, for exploring the code interactively)

### Model Training
The Random Forest classifier is employed for disease prediction. The model is trained on the training dataset, and hyperparameter optimization is performed using Grid Search.

### Hyperparameter Optimization
Grid Search is utilized to find the best hyperparameters for the Random Forest model. The optimized model is then trained on the scaled dataset.

### Model Evaluation
The model is evaluated using accuracy metrics and a confusion matrix on the testing dataset. The results indicate high accuracy on the provided data.

### SHAP Analysis
SHAP values are calculated and visualized to interpret the model's predictions. The SHAP summary plot provides insights into feature importance.
