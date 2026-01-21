import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score, mean_squared_error, confusion_matrix, classification_report
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline

# ==========================
# Load Dataset
# ==========================
data = pd.read_csv("../data/dataset.csv") # update path if needed

# ==========================
# Simple Regression
# ==========================
# Predict Transaction_Amount using Avg_Transaction_Amount_7d
X_simple = data[['Avg_Transaction_Amount_7d']]
y_simple = data['Transaction_Amount']

X_train, X_test, y_train, y_test = train_test_split(X_simple, y_simple, test_size=0.2, random_state=42)

simple_model = LinearRegression()
simple_model.fit(X_train, y_train)
y_pred_simple = simple_model.predict(X_test)

print("\n--- Simple Regression ---")
print(f"Model: Transaction_Amount = {simple_model.intercept_:.4f} + ({simple_model.coef_[0]:.4f}) * Avg_Transaction_Amount_7d")
print("Coefficient:", simple_model.coef_)
print("Intercept:", simple_model.intercept_)
print("R2 Score:", r2_score(y_test, y_pred_simple))
print("MSE:", mean_squared_error(y_test, y_pred_simple))

# ==========================
# Multiple Regression
# ==========================
# Predict Risk_Score using numeric + categorical features
features = ['Transaction_Amount','Account_Balance','Daily_Transaction_Count',
            'Avg_Transaction_Amount_7d','Failed_Transaction_Count_7d','Card_Age',
            'Transaction_Distance','IP_Address_Flag','Previous_Fraudulent_Activity',
            'Transaction_Type','Device_Type','Merchant_Category','Location','Card_Type']

target = 'Risk_Score'

X = data[features]
y = data[target]

numeric_features = X.select_dtypes(include=['int64','float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

preprocessor = ColumnTransformer([
    ('num', 'passthrough', numeric_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])

multiple_model = Pipeline([
    ('prep', preprocessor),
    ('reg', LinearRegression())
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

multiple_model.fit(X_train, y_train)
y_pred_multi = multiple_model.predict(X_test)

print("\n--- Multiple Regression ---")
num_coefs = multiple_model.named_steps['reg'].coef_[:len(numeric_features)]
print("Multiple Regression Formula (numeric features only):")
for feature, coef in zip(numeric_features, num_coefs):
    print(f" {feature}: {coef:.4f}")
print("R2 Score:", r2_score(y_test, y_pred_multi))
print("MSE:", mean_squared_error(y_test, y_pred_multi))

# ==========================
# Logistic Regression
# ==========================
# Predict Fraud_Label
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
    ('clf', LogisticRegression(max_iter=2000))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

log_model.fit(X_train, y_train)
y_pred_log = log_model.predict(X_test)

print("\n--- Logistic Regression ---")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_log))
print("\nClassification Report:\n", classification_report(y_test, y_pred_log))
