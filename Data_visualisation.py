# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_excel("data.xlsx", engine="openpyxl")

# Select numerical columns
numeric_df = df.select_dtypes(include=['number'])

# -----------------------------
# 1. Histogram
# -----------------------------
plt.figure(figsize=(8,6))

for column in numeric_df.columns:
    plt.hist(
        numeric_df[column],
        bins=10,
        alpha=0.5,
        label=column
    )

plt.title("Histogram")
plt.xlabel("Values")
plt.ylabel("Frequency")
plt.legend()
plt.show()

# -----------------------------
# 2. Boxplot
# -----------------------------
plt.figure(figsize=(10,6))

sns.boxplot(data=numeric_df)

plt.title("Boxplot")
plt.xticks(rotation=45)
plt.show()

# -----------------------------
# 3. Scatter Plot
# -----------------------------
plt.figure(figsize=(8,6))

plt.scatter(
    numeric_df.iloc[:,0],
    numeric_df.iloc[:,1]
)

plt.xlabel(numeric_df.columns[0])
plt.ylabel(numeric_df.columns[1])

plt.title("Scatter Plot")
plt.show()

# -----------------------------
# 4. Correlation Matrix
# -----------------------------
correlation = numeric_df.corr()

print("Correlation Matrix:")
print(correlation)

# -----------------------------
# 5. Heatmap
# -----------------------------
plt.figure(figsize=(8,6))

sns.heatmap(
    correlation,
    annot=True,
    cmap='coolwarm'
)

plt.title("Heatmap")
plt.show()

# -----------------------------
# 6. Pair Plot
# -----------------------------
sns.pairplot(numeric_df)

plt.show()