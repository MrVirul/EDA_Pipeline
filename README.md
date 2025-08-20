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
