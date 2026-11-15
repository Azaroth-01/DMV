import pandas as pd

import numpy as np

from matplotlib import pyplot as plt

import warnings



# Suppressing update warnings

warnings.filterwarnings('ignore')



# 1. Load Dataset

df1 = pd.read_csv("./datasets/Bengaluru_House_Data.csv")



# 2. Initial Inspection

print(df1.head())

print(df1.shape)

print(df1.columns)

print(df1['area_type'].unique())

print(df1['area_type'].value_counts())



# 3. Drop Unnecessary Columns

df2 = df1.drop(['area_type', 'society', 'balcony', 'availability'], axis='columns')

print(df2.shape)

print(df2.isnull().sum())



# 4. Handle Missing Values

df3 = df2.dropna()

print(df3.isnull().sum())

print(df3.shape)



# 5. Feature Engineering: Extract BHK from size

print(df3['size'].unique())

df3['bhk'] = df3['size'].apply(lambda x: int(x.split(' ')[0]))

print(df3.head())

print(df3.bhk.unique())

print(df3[df3.bhk > 20])



# 6. Clean total_sqft Column

def is_float(x):

    try:

        float(x)

        return True

    except (ValueError, TypeError):

        return False



print(df3[~df3['total_sqft'].apply(is_float)].head(10))



def convert_sqft_to_num(x):

    tokens = x.split('-')

    if len(tokens) == 2:

        try:

            return (float(tokens[0]) + float(tokens[1])) / 2

        except ValueError:

            return None

    try:

        return float(x)

    except ValueError:

        return None



df4 = df3.copy()

df4.total_sqft = df4.total_sqft.apply(convert_sqft_to_num)

df4 = df4[df4.total_sqft.notnull()]

print(df4.loc[30])



# 7. Feature Engineering: Price per Square Foot

df5 = df4.copy()

df5['price_per_sqft'] = df5['price'] * 100000 / df5['total_sqft']

print(df5.head())

print(df5['price_per_sqft'].describe())



# Save checkpoint

df5.to_csv("./datasets/bhp.csv", index=False)



# 8. Dimensionality Reduction on Location

df5.location = df5.location.apply(lambda x: x.strip())

location_stats = df5['location'].value_counts(ascending=False)

print(location_stats)

print("Locations > 10:", len(location_stats[location_stats > 10]))

print("Total locations:", len(location_stats))

print("Locations <= 10:", len(location_stats[location_stats <= 10]))



location_stats_less_than_10 = location_stats[location_stats <= 10]

df5.location = df5.location.apply(lambda x: 'other' if x in location_stats_less_than_10 else x)

print("Unique locations after grouping:", len(df5.location.unique()))

print(df5.head(10))



# 9. Outlier Removal (Business Logic: min 300 sqft per bhk)

print(df5[df5.total_sqft / df5.bhk < 300].head())

print(df5.shape)

df6 = df5[~(df5.total_sqft / df5.bhk < 300)]

print(df6.shape)

print(df6.columns)



# 10. Outlier Removal (Statistical IQR Method & Boxplots)

plt.boxplot(df6['total_sqft'])

plt.show()



# Removing total_sqft outliers using IQR

Q1 = np.percentile(df6['total_sqft'], 25.)

Q3 = np.percentile(df6['total_sqft'], 75.)

IQR = Q3 - Q1

ll = Q1 - (1.5 * IQR)

ul = Q3 + (1.5 * IQR)

upper_outliers = df6[df6['total_sqft'] > ul].index.tolist()

lower_outliers = df6[df6['total_sqft'] < ll].index.tolist()

bad_indices = list(set(upper_outliers + lower_outliers))

df6.drop(bad_indices, inplace=True, errors='ignore')



plt.boxplot(df6['bath'])

plt.show()



# Removing bath outliers using IQR

Q1 = np.percentile(df6['bath'], 25.)

Q3 = np.percentile(df6['bath'], 75.)

IQR = Q3 - Q1

ll = Q1 - (1.5 * IQR)

ul = Q3 + (1.5 * IQR)

upper_outliers = df6[df6['bath'] > ul].index.tolist()

lower_outliers = df6[df6['bath'] < ll].index.tolist()

bad_indices = list(set(upper_outliers + lower_outliers))

df6.drop(bad_indices, inplace=True, errors='ignore')



plt.boxplot(df6['price'])

plt.show()



# Removing price outliers using IQR

Q1 = np.percentile(df6['price'], 25.)

Q3 = np.percentile(df6['price'], 75.)

IQR = Q3 - Q1

ll = Q1 - (1.5 * IQR)

ul = Q3 + (1.5 * IQR)

upper_outliers = df6[df6['price'] > ul].index.tolist()

lower_outliers = df6[df6['price'] < ll].index.tolist()

bad_indices = list(set(upper_outliers + lower_outliers))

df6.drop(bad_indices, inplace=True, errors='ignore')



plt.boxplot(df6['bhk'])

plt.show()



# Removing price_per_sqft outliers using IQR

Q1 = np.percentile(df6['price_per_sqft'], 25.)

Q3 = np.percentile(df6['price_per_sqft'], 75.)

IQR = Q3 - Q1

ll = Q1 - (1.5 * IQR)

ul = Q3 + (1.5 * IQR)

upper_outliers = df6[df6['price_per_sqft'] > ul].index.tolist()

lower_outliers = df6[df6['price_per_sqft'] < ll].index.tolist()

bad_indices = list(set(upper_outliers + lower_outliers))

df6.drop(bad_indices, inplace=True, errors='ignore')



plt.boxplot(df6['price_per_sqft'])

plt.show()



print("Cleaned DataFrame shape:", df6.shape)



# 11. Prepare Features (X) and Target (y) for Model Training

X = df6.drop(['price'], axis='columns')

print(X.head(3))

print(X.shape)



y = df6.price

print(y.head(3))

print(len(y))



# 12. Train-Test Split

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)



print("X_train shape:", X_train.shape)

print("y_train shape:", y_train.shape)

print("X_test shape:", X_test.shape)

print("y_test shape:", y_test.shape)
