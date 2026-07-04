# Import libraries
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_excel("data.xlsx", engine="openpyxl")

print("Original Dataset:")
print(df.head())

# -----------------------------
# 1. Handle Missing Values
# -----------------------------

# Fill numerical missing values with mean
numeric_columns = df.select_dtypes(include=['number']).columns

for col in numeric_columns:
    df[col] = df[col].fillna(df[col].mean())

# Fill categorical missing values with mode
categorical_columns = df.select_dtypes(include=['object']).columns

for col in categorical_columns:
    df[col] = df[col].fillna(df[col].mode()[0])

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

# -----------------------------
# 2. Remove Duplicate Rows
# -----------------------------
df = df.drop_duplicates()

print("\nDuplicate rows removed")

# -----------------------------
# 3. Feature Scaling
# -----------------------------

# Normalization (0–1 range)
minmax_scaler = MinMaxScaler()

df[numeric_columns] = minmax_scaler.fit_transform(
    df[numeric_columns]
)

print("\nNormalized Data:")
print(df.head())

# Standardization
standard_scaler = StandardScaler()

scaled_data = standard_scaler.fit_transform(
    df[numeric_columns]
)

scaled_df = pd.DataFrame(
    scaled_data,
    columns=numeric_columns
)

print("\nStandardized Data:")
print(scaled_df.head())

# -----------------------------
# 4. Encode Categorical Variables
# -----------------------------

label_encoder = LabelEncoder()

for col in categorical_columns:
    df[col] = label_encoder.fit_transform(df[col])

print("\nEncoded Dataset:")
print(df.head())

# -----------------------------
# 5. Outlier Detection (IQR Method)
# -----------------------------

for col in numeric_columns:

    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - (1.5 * IQR)
    upper = Q3 + (1.5 * IQR)

    outliers = df[
        (df[col] < lower) |
        (df[col] > upper)
    ]

    print(f"\nOutliers in {col}:")
    print(len(outliers))