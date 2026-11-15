import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt

# ==========================================
# 1. Load Dataset & Basic Exploration
# ==========================================
# Load the dataset (adjust path if necessary)
data = pd.read_csv("./datasets/Telcom_Customer_Churn.csv")

print("--- Dataset Shape ---")
print(data.shape)

print("\n--- First 5 Rows ---")
print(data.head())

print("\n--- Column Names ---")
print(data.columns)

print("\n--- Missing Values Check ---")
print(data.isnull().sum())


# ==========================================
# 2. Data Cleaning & Preprocessing
# ==========================================
# Fix TotalCharges data type (Handles whitespace strings as NaN)
data['TotalCharges'] = data['TotalCharges'].replace(' ', np.nan)
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'])

# Impute missing values for TotalCharges with the median
data['TotalCharges'] = data['TotalCharges'].fillna(data['TotalCharges'].median())

# Check number of rows before removing duplicates
print("\nNumber of rows before removing duplicates:", len(data))

# Remove duplicate records
data_cleaned = data.drop_duplicates()
print("Number of rows after removing duplicates:", len(data_cleaned))


# ==========================================
# 3. Exploratory Data Analysis (EDA)
# ==========================================
print("\n--- Descriptive Statistics ---")
print(data_cleaned.describe())

# Frequency distribution checks for numerical features
unique_tenure, counts_tenure = np.unique(data_cleaned['tenure'], return_counts=True)
unique_mc, counts_mc = np.unique(data_cleaned['MonthlyCharges'], return_counts=True)

# Boxplots for numerical columns
plt.figure(figsize=(6, 4))
plt.boxplot(data_cleaned['tenure'])
plt.title("Boxplot of Tenure")
plt.show()

plt.figure(figsize=(6, 4))
plt.boxplot(data_cleaned['MonthlyCharges']) 
plt.title("Boxplot of Monthly Charges")
plt.show()


# ==========================================
# 4. Feature Selection & Train-Test Split
# ==========================================
# Define Features (X) and Target (y)
X = data_cleaned.drop("Churn", axis=1) 
y = data_cleaned["Churn"]

# Split the dataset into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\n--- Dataset Split Shapes ---")
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)


# ==========================================
# 5. Export Cleaned Dataset
# ==========================================
data_cleaned.to_csv("./datasets/Cleaned_Telecom_Customer_Churn.csv", index=False)
print("\nCleaned dataset exported successfully.")
