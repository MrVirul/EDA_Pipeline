# 🚗 Car Dataset EDA Pipeline

This repository contains an **Exploratory Data Analysis (EDA) pipeline** for a car dataset with **10,000+ rows** and **10+ features** including `Engine Fuel Type`, `Engine HP`, `Transmission Type`, `Highway MPG`, and `City MPG`.  

The goal is to preprocess, clean, and visualize the dataset to make it suitable for **Machine Learning modeling**.  

---

## 📌 Features of the EDA Pipeline
- ✅ Data Cleaning (duplicates, null values, irrelevant columns)  
- ✅ Outlier Detection & Removal (IQR method)  
- ✅ Renaming & reshaping columns for readability  
- ✅ Statistical & Visual Insights  
- ✅ Ready for ML modeling  

---

## 🛠️ Libraries Used
- [**Pandas**](https://pandas.pydata.org/) → Data manipulation & cleaning  
- [**NumPy**](https://numpy.org/) → Numerical operations  
- [**Matplotlib**](https://matplotlib.org/) → Visualization (plots, charts)  
- [**Seaborn**](https://seaborn.pydata.org/) → Statistical data visualization  

```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

📂 Steps in the Pipeline
1️⃣ Load the Dataset
df = pd.read_csv("car_data.csv")

2️⃣ Display Sample Data
df.head()    # first 5 rows
df.tail()    # last 5 rows

3️⃣ Data Types Check

Ensures all columns are in correct format for analysis (e.g., MSRP should be numeric).

df.dtypes

4️⃣ Remove Irrelevant Columns

Dropped columns not useful for ML modeling:

Engine Fuel Type

Market Category

Vehicle Style

Popularity

Number of Doors

Vehicle Size

df.drop(["Engine Fuel Type","Market Category","Vehicle Style",
         "Popularity","Number of Doors","Vehicle Size"], axis=1, inplace=True)

5️⃣ Rename Columns for Readability
df.rename(columns={"Engine HP":"Horsepower",
                   "MSRP":"Price"}, inplace=True)

6️⃣ Handle Duplicates
print("Before:", df.shape)
df = df.drop_duplicates()
print("After:", df.shape)

7️⃣ Handle Missing Values

Since missing values are few → dropped.

df = df.dropna()

8️⃣ Outlier Detection & Removal

Using Interquartile Range (IQR) Method

Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1

df = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]


Visual check with Boxplots:

sns.boxplot(df["Price"])
sns.boxplot(df["Horsepower"])

9️⃣ Data Visualization
🔹 Scatter Plot
sns.scatterplot(x="Horsepower", y="Price", data=df)

🔹 Histogram
df["Highway MPG"].hist(bins=30)

🔹 Correlation Heatmap
numeric_df = df.select_dtypes(include=["number"])
sns.heatmap(numeric_df.corr(), annot=True, cmap="BrBG")
