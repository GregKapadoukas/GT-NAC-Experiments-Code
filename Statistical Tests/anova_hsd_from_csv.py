import argparse
import sys

import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Set up argparse to handle command-line arguments
parser = argparse.ArgumentParser(description="Perform ANOVA on model evaluation data.")
parser.add_argument(
    "filename", type=str, help="Path to the CSV file containing the data"
)

# Parse the arguments
args = parser.parse_args()

try:
    df = pd.read_csv(args.filename)
    print("File loaded successfully.")
except FileNotFoundError:
    print(
        f"Error: The file '{args.filename}' was not found. Please check the filename and try again."
    )
    sys.exit(1)

# Ensure necessary columns are present
required_columns = {"Model", "Metric", "Fold", "Score"}
if not required_columns.issubset(df.columns):
    print(f"Error: The CSV file must contain the following columns: {required_columns}")
    sys.exit(1)

df_accuracy = df[df["Metric"] == "Accuracy"]
df_precision = df[df["Metric"] == "Precision"]
df_recall = df[df["Metric"] == "Recall"]

#  Perform ANOVA for Accuracy
print("Performing statistical tests for Accuracy")
anova_model = ols("Score ~ C(Model)", data=df_accuracy).fit()
anova_table = sm.stats.anova_lm(anova_model, typ=2)
print("\nANOVA Table:")
print(anova_table)

#  Perform Tukey’s HSD test for pairwise comparisons between models for Accuracy
print("\nTukey’s HSD Test Results for Model across all Metrics:")
tukey_model = pairwise_tukeyhsd(
    endog=df_accuracy["Score"], groups=df_accuracy["Model"], alpha=0.05
)
print(tukey_model)

#  Perform ANOVA for Precision
print("Performing statistical tests for Precision")
anova_model = ols("Score ~ C(Model)", data=df_precision).fit()
anova_table = sm.stats.anova_lm(anova_model, typ=2)
print("\nANOVA Table:")
print(anova_table)

#  Perform Tukey’s HSD test for pairwise comparisons between models for Precision
print("\nTukey’s HSD Test Results for Model across all Metrics:")
tukey_model = pairwise_tukeyhsd(
    endog=df_precision["Score"], groups=df_precision["Model"], alpha=0.05
)
print(tukey_model)

#  Perform ANOVA for Recall
print("Performing statistical tests for Recall")
anova_model = ols("Score ~ C(Model)", data=df_recall).fit()
anova_table = sm.stats.anova_lm(anova_model, typ=2)
print("\nANOVA Table:")
print(anova_table)

#  Perform Tukey’s HSD test for pairwise comparisons between models for Precision
print("\nTukey’s HSD Test Results for Model across all Metrics:")
tukey_model = pairwise_tukeyhsd(
    endog=df_recall["Score"], groups=df_recall["Model"], alpha=0.05
)
print(tukey_model)
