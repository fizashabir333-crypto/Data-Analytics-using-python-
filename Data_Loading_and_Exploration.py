# Import required libraries
import pandas as pd
import matplotlib.pyplot as plt

# 1. Load datasets

# CSV file
csv_data = pd.read_csv("data.csv")
print(csv_data.to_string()) 


# Excel file
excel_data = pd.read_excel("data.xlsx")
print(excel_data.to_string())


# Using one dataset for analysis
df = csv_data

# -----------------------------
# 2. Display first few rows
# -----------------------------
print("First 5 rows:")
print(df.head())

# -----------------------------
# 3. Dataset information
# -----------------------------
print("\nDataset Info:")
print(df.info())

# -----------------------------
# 4. Summary statistics
# -----------------------------
print("\nSummary Statistics:")
print(df.describe())

# -----------------------------
# 5. Data types inspection
# -----------------------------
print("\nData Types:")
print(df.dtypes)

# -----------------------------
# 6. Missing value detection
# -----------------------------
print("\nMissing Values:")
print(df.isnull().sum())


# 7. Duplicate values check

print("\nDuplicate Rows:")
print(df.duplicated().sum())


# 8. Basic Visualization


# Histogram for numerical columns
df.hist(figsize=(10,8))
plt.suptitle("Histogram of Dataset Features")
plt.show()

# Boxplot for outlier detection
df.plot(kind='box', figsize=(10,6))
plt.title("Boxplot")
plt.show()

# Correlation heatmap
correlation = df.select_dtypes(include=['number']).corr()

plt.figure(figsize=(8,6))
plt.imshow(correlation)
plt.colorbar()
plt.xticks(range(len(correlation.columns)), correlation.columns, rotation=45)
plt.yticks(range(len(correlation.columns)), correlation.columns)
plt.title("Correlation Heatmap")
plt.show()