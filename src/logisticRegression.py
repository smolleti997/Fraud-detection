import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

data = pd.read_csv("../data/dataset.csv")

log_features = ['Transaction_Amount','Account_Balance','Daily_Transaction_Count',
                'Avg_Transaction_Amount_7d','Failed_Transaction_Count_7d','Card_Age',
                'Transaction_Distance','IP_Address_Flag','Previous_Fraudulent_Activity',
                'Transaction_Type','Device_Type','Merchant_Category','Location','Card_Type']

X = data[log_features]
y = data['Fraud_Label']

numeric_features = X.select_dtypes(include=['int64','float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

preprocessor_log = ColumnTransformer([
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])

log_model = Pipeline([
    ('prep', preprocessor_log),
    ('clf', LogisticRegression(max_iter=3000, solver='saga', class_weight='balanced'))
])

# ===== Train the model =====
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
log_model.fit(X_train, y_train)  # <--- Must fit before predicting

# ===== Predict probabilities =====
y_probs = log_model.predict_proba(X_test)[:,1]

# ===== Apply lower threshold for fraud detection =====
y_pred_log_optimized = (y_probs > 0.3).astype(int)

# ===== Evaluation =====
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_log_optimized))
print("\nClassification Report:\n", classification_report(y_test, y_pred_log_optimized))
