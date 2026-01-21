# Fraud Detection Model

This project builds a **Logistic Regression–based fraud detection model** using Python and scikit-learn. The workflow covers **data preprocessing, model training, prediction, and performance evaluation**.

---

## 📂 Steps Performed

### 1. Data Loading
- Loaded the dataset using `pandas.read_csv()`.
- Stored it in a DataFrame for processing.

### 2. Feature Selection
- Selected key variables such as:
  - Transaction amount
  - Account balance
  - Risk score
  - Transaction type
  - Failed transactions in last 7 days
  - Previous fraudulent activity
- Defined `Fraud_Label` as the target variable.

### 🔧 3. Data Preprocessing
**Categorical Encoding**  
- Used **one-hot encoding** to convert `Transaction_Type` into numerical binary columns.
- `drop_first=True` avoids multicollinearity.

**Feature Scaling**  
- Applied `StandardScaler` to normalize numeric features:
  - Mean = 0  
  - Standard deviation = 1  
- Helps the model converge faster and perform better.

### 🧪 4. Train–Test Split
- Split the data into:
  - 80% training (used to learn patterns)  
  - 20% testing (used to evaluate performance)  
- Used **stratification** to maintain the fraud ratio across splits.
- Set `random_state=42` for reproducibility.

### 🤖 5. Model Training
- Trained a **Logistic Regression** classifier with:
  - `class_weight='balanced'` to handle class imbalance (fraud cases are rare)  
  - `max_iter=1000` to ensure convergence
- The model learns weights (coefficients) for each feature to determine fraud probability.

### 🔮 6. Prediction
- Predicted fraud on the test dataset using `predict()`:
  - `0` → legitimate transaction  
  - `1` → fraudulent transaction
- Compared predictions with actual labels to measure performance.

### 📊 7. Model Evaluation
**Confusion Matrix**  
- True Positives (TP): correctly predicted fraud  
- True Negatives (TN): correctly predicted non-fraud  
- False Positives (FP): predicted fraud but actually legit  
- False Negatives (FN): missed fraud  
- Helps assess model strengths and weaknesses.

**Classification Report**  
- Precision: How many predicted fraud cases were actually fraud  
- Recall: How many actual fraud cases the model caught  
- F1-score: Balance of precision and recall  
- Support: Number of samples per class

**Model Coefficients**  
- Printed the learned weights for each feature.  
- Helps interpret which features increase or decrease fraud probability.

---

## 🎯 Purpose
This workflow prepares, trains, and evaluates a machine learning model that predicts the likelihood of a transaction being fraudulent based on financial behavior and historical activity. It provides a clear, **end-to-end approach for fraud classification tasks**.
