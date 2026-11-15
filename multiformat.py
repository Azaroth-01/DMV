import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import json

# 1. Load data from multiple file formats
csv = pd.read_csv("./datasets/sales_data_sample.csv", encoding="cp1252")
ed = pd.read_excel("./datasets/Sample-Sales-Data.xlsx")

with open("./datasets/customers.json", "r") as json_file:
    json_data = json.load(json_file)

# 2. Explore the structure and content
print("--- CSV Tail ---")
print(csv.tail())

print("\n--- CSV Info ---")
csv.info()

print("\n--- CSV Describe ---")
print(csv.describe())

print("\n--- Excel Head ---")
print(ed.head())

print("\n--- Excel Tail ---")
print(ed.tail())

ed.info()
print(ed.describe())

# 3. Data Cleaning and Preprocessing
# Drop missing values and remove duplicates
cleaned_csv = csv.dropna().drop_duplicates()

# 4. Data Unification (Concatenating datasets)
unified_data = pd.concat([csv, ed], ignore_index=True)

# 5. Data Analysis & Aggregation
total_sales = unified_data['SALES'].sum()
print("\nTotal Sales:", total_sales)

category_sales = unified_data.groupby('ORDERNUMBER')['SALES'].mean()

# 6. Data Visualization
category_counts = unified_data['SALES'].value_counts()
category_counts.plot(kind='bar')
plt.title('Product Category Distribution')
plt.xlabel('Category')
plt.ylabel('Count')
plt.show()
