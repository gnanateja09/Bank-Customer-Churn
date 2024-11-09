# Bank Customer Churn Prediction Model

This project aims to build a machine learning model to predict customer churn for a bank using historical customer data. The model uses a **Support Vector Machine (SVM)** classifier to predict whether a customer will churn or not based on various features such as gender, age, balance, and more.

## Project Structure
/Bank-Customer-Churn/ │ ├── bankchurn.ipynb # Jupyter Notebook containing the machine learning code ├── bank_customer_churn.csv # Dataset containing customer information └── README.md # Project description and instructions

## Requirements

To run this project, you'll need the following Python libraries:

- `pandas`: For data manipulation and analysis.
- `scikit-learn`: For machine learning models, data preprocessing, and evaluation.
- `imbalanced-learn`: For handling imbalanced data using SMOTE (Synthetic Minority Over-sampling Technique).
- `matplotlib`: For visualizations (optional, if you want to visualize results).

### Installation
You can install the required libraries using pip:
pip install pandas scikit-learn imbalanced-learn matplotlib
Dataset
The dataset (bank_customer_churn.csv) contains historical data of bank customers, with features like:

Age: Age of the customer.
Balance: Account balance.
Gender: Gender of the customer (Male/Female).
Geography: Customer's country (e.g., France, Germany, Spain).
CreditScore: Customer's credit score.
Churn: Target variable indicating whether the customer churned (1) or not (0).
Model Overview
Steps in the Code:
Data Preprocessing:

Loading Data: The dataset is loaded into a Pandas DataFrame.
Data Encoding: Categorical variables (such as Gender and Geography) are encoded using LabelEncoder to convert them into numerical values, as machine learning models require numeric input.
Handling Imbalanced Data:

The target variable (Churn) has imbalanced classes, meaning that there are more customers who didn't churn than those who did. To address this, we use SMOTE (Synthetic Minority Over-sampling Technique) to generate synthetic samples for the minority class (customers who churn).
Feature Scaling:

Standard Scaling is applied to ensure that all features have the same scale, which is important for distance-based algorithms like SVM.
Model Training:

An SVM classifier is initialized and trained on the preprocessed data (X_train and y_train).
Hyperparameter Tuning: Grid search is used to find the optimal hyperparameters for the SVM model (e.g., C, gamma, and kernel).
Model Evaluation:

The model is evaluated on the test data (X_test and y_test) using metrics like precision, recall, and F1-score.
A confusion matrix is used to assess the true positives, true negatives, false positives, and false negatives.
How to Run the Code
Clone the Repository: First, clone the repository to your local machine or Google Colab:
git clone https://github.com/gnanateja09/Bank-Customer-Churn.git
cd Bank-Customer-Churn
Load the Dataset: Place the bank_customer_churn.csv file in the same directory or provide the correct file path in the code.

Run the Code: Open the bankchurn.ipynb file in Jupyter Notebook or Google Colab and run all the cells to train and evaluate the model.

Evaluate the Model: After training, the output will include:

The best hyperparameters found using GridSearchCV.
The classification report with metrics like precision, recall, F1-score.
The confusion matrix.
Sample Output:
Best Parameters: {'C': 1, 'gamma': 'scale', 'kernel': 'rbf'}
Classification Report:
              precision    recall  f1-score   support

           0       0.89      0.97      0.93      1607
           1       0.54      0.25      0.34       393

    accuracy                           0.87      2000
   macro avg       0.72      0.61      0.64      2000
weighted avg       0.84      0.87      0.85      2000

Confusion Matrix:
[[1566   41]
 [ 294   99]]
Conclusion
This machine learning project demonstrates how to use an SVM classifier to predict customer churn in a bank. It involves key techniques like data preprocessing, handling imbalanced data, hyperparameter tuning, and model evaluation. The model's effectiveness can be assessed using various metrics, and improvements can be made by exploring different machine learning algorithms or tuning the model further.
