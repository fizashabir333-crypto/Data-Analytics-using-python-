# Import libraries
import pandas as pd
from scipy import stats
from scipy.stats import chi2_contingency
from scipy.stats import f_oneway

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_excel("data.xlsx", engine="openpyxl")

# Select numerical columns
numeric_df = df.select_dtypes(include=['number'])

# -----------------------------
# 1. Descriptive Statistics
# -----------------------------
print("Descriptive Statistics")
print(df.describe())

# Mean
print("\nMean:")
print(numeric_df.mean())

# Variance
print("\nVariance:")
print(numeric_df.var())

# Standard Deviation
print("\nStandard Deviation:")
print(numeric_df.std())

# -----------------------------
# 2. Correlation Analysis
# -----------------------------
print("\nCorrelation Matrix:")
correlation = numeric_df.corr()

print(correlation)

# -----------------------------
# 3. Hypothesis Testing
# T-Test
# -----------------------------

# Use first two numeric columns
group1 = numeric_df.iloc[:,0]
group2 = numeric_df.iloc[:,1]

t_statistic, p_value = stats.ttest_ind(
    group1,
    group2
)

print("\nT-Test Results")
print("T-statistic:", t_statistic)
print("P-value:", p_value)

if p_value < 0.05:
    print("Reject Null Hypothesis")
else:
    print("Fail to Reject Null Hypothesis")

# -----------------------------
# 4. Chi-Square Test
# -----------------------------

# Use first two categorical columns
categorical_df = df.select_dtypes(include=['object'])

if len(categorical_df.columns) >= 2:

    contingency_table = pd.crosstab(
        categorical_df.iloc[:,0],
        categorical_df.iloc[:,1]
    )

    chi2, p, dof, expected = chi2_contingency(
        contingency_table
    )

    print("\nChi-Square Test")
    print("Chi-square value:", chi2)
    print("P-value:", p)

# -----------------------------
# 5. ANOVA Test
# -----------------------------

if len(numeric_df.columns) >= 3:

    anova_result = f_oneway(
        numeric_df.iloc[:,0],
        numeric_df.iloc[:,1],
        numeric_df.iloc[:,2]
    )

    print("\nANOVA Test")
    print("F-statistic:",
          anova_result.statistic)

    print("P-value:",
          anova_result.pvalue)

    if anova_result.pvalue < 0.05:
        print("Significant Difference Exists")
    else:
        print("No Significant Difference")



# describe() → descriptive statistics summary
# mean() → average value
# var() → variance calculation
# std() → standard deviation
# corr() → correlation between variables
# ttest_ind() → compares means of two groups
# chi2_contingency() → tests relationship between categorical variables
# f_oneway() → performs ANOVA across multiple groups