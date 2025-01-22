# Bank Customer Churn Prediction

This project aims to predict customer churn for a bank using a **Support Vector Machine (SVM)** classifier. By analyzing historical customer data, the model identifies patterns and features that indicate whether a customer is likely to churn.

## Project Structure

```
/Bank-Customer-Churn/
├── bankchurn.ipynb          # Jupyter Notebook containing the machine learning code
├── bank_customer_churn.csv  # Dataset with customer information
└── README.md                # Project description and instructions
```

## Requirements

To run this project, you will need the following Python libraries:

- **pandas**: For data manipulation and analysis.
- **scikit-learn**: For machine learning models, preprocessing, and evaluation.
- **imbalanced-learn**: To handle imbalanced data using SMOTE (Synthetic Minority Over-sampling Technique).
- **matplotlib**: For visualizations (optional).

### Installation

Install the required libraries using pip:

```bash
pip install pandas scikit-learn imbalanced-learn matplotlib
```

## Dataset Overview

The dataset (`bank_customer_churn.csv`) contains customer details and includes the following features:

- **Age**: Age of the customer.  
- **Balance**: Account balance.  
- **Gender**: Gender of the customer (Male/Female).  
- **Geography**: Customer's country (e.g., France, Germany, Spain).  
- **CreditScore**: Customer's credit score.  
- **Churn**: Target variable (1 if the customer churned, 0 otherwise).  

## Model Workflow

### 1. Data Preprocessing

- **Loading Data**: Load the dataset into a Pandas DataFrame.
- **Encoding**: Use `LabelEncoder` to convert categorical variables (e.g., Gender, Geography) into numeric values for compatibility with machine learning models.

### 2. Handling Imbalanced Data

- The dataset is imbalanced, with more customers who did not churn compared to those who did.
- To address this, **SMOTE** is used to generate synthetic samples for the minority class (churned customers).

### 3. Feature Scaling

- Apply **StandardScaler** to normalize features, ensuring they are on the same scale. This is crucial for algorithms like SVM.

### 4. Model Training

- **SVM Classifier**: Train an SVM model on the preprocessed data.
- **Hyperparameter Tuning**: Use **GridSearchCV** to optimize parameters such as `C`, `gamma`, and `kernel`.

### 5. Model Evaluation

- Evaluate the model using the test data and the following metrics:
  - **Precision**, **Recall**, **F1-score**
  - **Confusion Matrix**: To assess true positives, true negatives, false positives, and false negatives.

## How to Run

1. **Clone the Repository**:  
   Clone the repository to your local machine or Google Colab:

   ```bash
   git clone https://github.com/gnanateja09/Bank-Customer-Churn.git
   cd Bank-Customer-Churn
   ```

2. **Load the Dataset**:  
   Place the `bank_customer_churn.csv` file in the same directory or provide the correct path in the code.

3. **Run the Notebook**:  
   Open the `bankchurn.ipynb` file in Jupyter Notebook or Google Colab and execute all the cells.

4. **View the Results**:  
   After training, the output includes:
   - Best hyperparameters from **GridSearchCV**.
   - Classification report (precision, recall, F1-score).
   - Confusion matrix.

## Sample Output

**Best Parameters**:  
`{'C': 1, 'gamma': 'scale', 'kernel': 'rbf'}`

**Classification Report**:

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0 (No Churn) | 0.89 | 0.97 | 0.93 | 1607 |
| 1 (Churn) | 0.54 | 0.25 | 0.34 | 393 |

**Accuracy**: 87%  
**Confusion Matrix**:  
```
[[1566   41]
 [ 294   99]]
```

## Conclusion

This project demonstrates how to build a machine learning model to predict bank customer churn. Key techniques include:

- Handling imbalanced data with **SMOTE**.
- Optimizing model performance using **GridSearchCV**.
- Evaluating the model with various metrics.

The current model performs well for the majority class (No Churn) but has room for improvement in predicting the minority class (Churn). Future enhancements could include:

- Trying other algorithms like Random Forest or Gradient Boosting.
- Engineering new features or refining existing ones.
- Using advanced techniques like ensemble methods or neural networks.

Feel free to contribute or provide feedback on this project!
