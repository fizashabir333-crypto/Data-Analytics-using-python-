# Import libraries
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc
)

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_excel(
    "data.xlsx",
    engine="openpyxl"
)

# Features
X = df.iloc[:, :-1]

# Target column (last column)
y = df.iloc[:, -1]

print(type(y))
print(y.head())

print("Data type:")
print(y.dtype)

print("Unique values:")
print(y.unique())

print(df.columns)


# -----------------------------
# Data preprocessing
# -----------------------------

# Remove missing values
df = df.dropna()

# Convert categorical values into numbers
encoder = LabelEncoder()

for col in df.select_dtypes(include=['object']).columns:
    df[col] = encoder.fit_transform(
        df[col]
    )

# -----------------------------
# Features and Target
# -----------------------------

# Features = all columns except last
X = df.iloc[:, :-1]

# Target = last column
y = df.iloc[:, -1]

# Convert continuous target into classes
y = pd.cut(
    y,
    bins=3,
    labels=[0,1,2]
)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# -----------------------------
# Create models
# -----------------------------

models = {
    "Logistic Regression":
        LogisticRegression(),

    "kNN":
        KNeighborsClassifier(n_neighbors=5),

    "Decision Tree":
        DecisionTreeClassifier(),

    "SVM":
        SVC(probability=True)
}

# -----------------------------
# Train and evaluate
# -----------------------------

for name, model in models.items():

    print("\n", "="*40)
    print(name)

    # Train
    model.fit(
        X_train,
        y_train
    )

    # Prediction
    y_pred = model.predict(
        X_test
    )

    # Evaluation metrics
    cm = confusion_matrix(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        average='weighted'
    )

    recall = recall_score(
        y_test,
        y_pred,
        average='weighted'
    )

    f1 = f1_score(
        y_test,
        y_pred,
        average='weighted'
    )

    print("\nConfusion Matrix:")
    print(cm)

    print("\nPrecision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)

    # -----------------------------
    # ROC Curve
    # -----------------------------

    if len(y.unique()) == 2:

        y_probability = model.predict_proba(
            X_test
        )[:,1]

        fpr, tpr, threshold = roc_curve(
            y_test,
            y_probability
        )

        roc_auc = auc(
            fpr,
            tpr
        )

        plt.figure()

        plt.plot(
            fpr,
            tpr,
            label="AUC="+str(roc_auc)
        )

        plt.plot(
            [0,1],
            [0,1]
        )

        plt.xlabel(
            "False Positive Rate"
        )

        plt.ylabel(
            "True Positive Rate"
        )

        plt.title(
            "ROC Curve - " + name
        )

        plt.legend()

        plt.show()