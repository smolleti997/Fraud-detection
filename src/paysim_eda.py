# src/paysim_eda.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler


# 1. Load Dataset
df = pd.read_csv("../data/dataset.csv")

print("✅ Data loaded successfully!")
print(df.head())
print(df.info())
print(df.describe())

# Drop columns that are completely empty
df.dropna(axis=1, how='all', inplace=True)

#scale numeric value makes numeric features comparable
scaler = StandardScaler()
numeric_cols = ['Transaction_Amount', 'Account_Balance', 'Daily_Transaction_Count', 'Avg_Transaction_Amount_7d', 'Failed_Transaction_Count_7d', 'Card_Age', 'Transaction_Distance', 'Risk_Score', 'Previous_Fraudulent_Activity']
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

#drop duplicates
df.drop_duplicates(inplace=True)

#### UNIVARIATE ANALYSIS
#fraud vs non fraud count
sns.countplot(x='Fraud_Label', data=df)
plt.show()

# distribution of transaction amounts
sns.histplot(df['Transaction_Amount'], bins=50, kde=True)
plt.show()

#distribution of card ages
sns.histplot(df['Card_Age'], bins=30, kde=True)
plt.show()

#failed transaction count
sns.histplot(df['Failed_Transaction_Count_7d'], bins=20, kde=True)
plt.show()

# BIVARIATE ANALYSIS
# Transaction amount vs Fraud
sns.boxplot(x='Fraud_Label', y='Transaction_Amount', data=df)
plt.show()

#Account balance vs Fraud
sns.boxplot(x='Fraud_Label', y='Account_Balance', data=df)
plt.show()

#Risk Score vs Fraud
sns.boxplot(x='Fraud_Label', y='Risk_Score', data=df)
plt.show()

#Scatter: Transaction distance vs Amount
sns.scatterplot(x='Transaction_Distance', y='Transaction_Amount', hue='Fraud_Label', data=df)
plt.show()

#Scatter: Transaction distance vs Amount
sns.scatterplot(x='Transaction_Amount', y='Transaction_Type', hue='Fraud_Label', data=df)
plt.show()

##Categorical Variable Analysis
#Transaction Type vs Fraud
sns.countplot(x='Transaction_Type', hue='Fraud_Label', data=df)
plt.show()

#Device Type vs Fraud
sns.countplot(x='Device_Type', hue='Fraud_Label', data=df)
plt.show()

#Merchant Category vs Fraud
sns.countplot(x='Merchant_Category', hue='Fraud_Label', data=df)
plt.show()

#Is Weekend vs Fraud
sns.countplot(x='Is_Weekend', hue='Fraud_Label', data=df)
plt.show()
