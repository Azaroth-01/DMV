import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



# Configure visual appearance

sns.set_theme(style="whitegrid")



# Step 1 & 2: Load and Explore Dataset

# Assuming 'Retail_Sales_Data.csv' is available locally or via URL

try:

    df = pd.read_csv('Retail_Sales_Data.csv')

except FileNotFoundError:

    # Creating mock data if the file is not present for demonstration purposes

    np.random.seed(42)

    regions = ['North', 'South', 'East', 'West']

    categories = ['Beauty', 'Clothing', 'Electronics']

    data = {

        'transaction_date': pd.date_range(start='2026-01-01', periods=500, freq='D'),

        'region': np.random.choice(regions, 500),

        'product_category': np.random.choice(categories, 500),

        'quantity_sold': np.random.randint(1, 10, 500),

        'sales_amount': np.random.uniform(50, 500, 500)

    }

    df = pd.DataFrame(data)

    df.to_csv('Retail_Sales_Data.csv', index=False)



print("Dataset Shape:", df.shape)

print(df.head())

print(df.info())



# Step 3 & 4: Group sales data by region and calculate total sales amount

region_sales = df.groupby('region')['sales_amount'].sum().reset_index()

region_sales = region_sales.sort_values(by='sales_amount', ascending=False)

print("\nTotal Sales by Region:")

print(region_sales)



# Step 5: Visualizing Regional Sales Distribution (Bar Plot)

plt.figure(figsize=(6, 4))

sns.barplot(data=region_sales, x='region', y='sales_amount', hue='region', palette='viridis', legend=False)

plt.title('Total Sales by Region')

plt.xlabel('Region')

plt.ylabel('Total Sales ($)')

plt.xticks(rotation=45)

plt.tight_layout()

plt.show()



# Step 6: Identify the Top-Performing Region

top_region = region_sales.iloc[0]

print(f"\nThe top-performing region is {top_region['region']} with total sales of ${top_region['sales_amount']:.2f}")



# Step 7: Group sales data by region and product category

category_sales = df.groupby(['region', 'product_category'])['sales_amount'].sum().reset_index()

print("\nSales Summary by Region and Product Category:")

print(category_sales)



# Step 8: Create Stacked Bar Plots to Compare Sales Across Regions and Product Categories

pivot_sales = category_sales.pivot(index='region', columns='product_category', values='sales_amount')



pivot_sales.plot(kind='bar', stacked=True, figsize=(7, 5), colormap='tab20')

plt.title('Sales Distribution by Region and Product Category')

plt.xlabel('Region')

plt.ylabel('Sales Amount ($)')

plt.legend(title='Product Category', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.xticks(rotation=45)

plt.tight_layout()

plt.show()
