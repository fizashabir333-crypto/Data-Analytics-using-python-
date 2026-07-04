# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load data
df = pd.read_excel(
    "data.xlsx",
    engine="openpyxl"
)

# Keep only numeric columns
numeric_df = df.select_dtypes(
    include=['number']
)

# Remove missing values
numeric_df = numeric_df.dropna()

# Features = all columns except last
X = numeric_df.iloc[:, :-1]

# Target = last column
y = numeric_df.iloc[:, -1]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create model
multi_model = LinearRegression()

# Train model
multi_model.fit(
    X_train,
    y_train
)

print("Model trained successfully")


from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error
)

import numpy as np

# Predict values
y_pred = multi_model.predict(X_test)

# Calculate metrics
r2 = r2_score(y_test, y_pred)

mse = mean_squared_error(
    y_test,
    y_pred
)

rmse = np.sqrt(mse)

mae = mean_absolute_error(
    y_test,
    y_pred
)

# Print results
print("\nModel Evaluation")
print("R² Score:", r2)
print("MSE:", mse)
print("RMSE:", rmse)
print("MAE:", mae)