# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from collections import Counter

# Load and sample dataset
df = pd.read_csv('Bank Churn Modelling.csv')  # replace with your dataset path
df_sample = df.sample(frac=0.3, random_state=42)  # Use a 30% sample for faster execution

# Separate features and target variable
X = df_sample.drop('Churn', axis=1)
y = df_sample['Churn']

# Data Encoding
for column in X.columns:
    if X[column].dtype == 'object':
        label_encoder = LabelEncoder()
        X[column] = label_encoder.fit_transform(X[column])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Handle Imbalanced Data using SMOTE
print(f'Original dataset shape {Counter(y_train)}')
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)
print(f'Resampled dataset shape {Counter(y_train)}')

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define the SVM model
svm_model = SVC()

# Set up a reduced hyperparameter grid for GridSearchCV
param_grid = {
    'C': [0.1, 1, 10],
    'gamma': ['scale', 0.1],
    'kernel': ['rbf']
}

# Grid Search for Hyperparameter Tuning with reduced grid and fewer folds
grid_search = GridSearchCV(estimator=svm_model, param_grid=param_grid, scoring='f1', cv=3, verbose=2, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best model from grid search
best_svm_model = grid_search.best_estimator_
print("Best Parameters:", grid_search.best_params_)

# Predict on the test data
y_pred = best_svm_model.predict(X_test)

# Evaluate the model
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
