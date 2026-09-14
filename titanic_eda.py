import os
import pandas as pd
import matplotlib.pyplot as plt

DATA_PATH = "data/train.csv"
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load data

df = pd.read_csv(DATA_PATH)

# Initial inspection
print("Original shape:", df.shape)
print("\nMissing values before cleaning:\n", df.isnull().sum())
print("\nDuplicate rows:", df.duplicated().sum())

# Data cleaning
# Age: fill missing values with median
# Embarked: fill missing values with mode
# Cabin: remove because most entries are missing

df["Age"] = df["Age"].fillna(df["Age"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
df = df.drop(columns=["Cabin"])
df = df.drop_duplicates()

# Save cleaned dataset
cleaned_path = os.path.join(OUTPUT_DIR, "cleaned_titanic.csv")
df.to_csv(cleaned_path, index=False)

print("\nShape after cleaning:", df.shape)
print("\nMissing values after cleaning:\n", df.isnull().sum())

# Style
plt.style.use("default")

# 1. Survival distribution
plt.figure(figsize=(7, 5))
df["Survived"].value_counts().sort_index().plot(kind="bar")
plt.title("Titanic Survival Distribution")
plt.xlabel("Survived (0 = No, 1 = Yes)")
plt.ylabel("Number of Passengers")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "01_survival_distribution.png"), dpi=200)
plt.close()

# 2. Survival by gender
plt.figure(figsize=(7, 5))
pd.crosstab(df["Sex"], df["Survived"]).plot(kind="bar")
plt.title("Survival by Gender")
plt.xlabel("Gender")
plt.ylabel("Number of Passengers")
plt.legend(title="Survived", labels=["No", "Yes"])
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "02_survival_by_gender.png"), dpi=200)
plt.close()

# 3. Survival by passenger class
plt.figure(figsize=(7, 5))
pd.crosstab(df["Pclass"], df["Survived"]).plot(kind="bar")
plt.title("Survival by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Number of Passengers")
plt.legend(title="Survived", labels=["No", "Yes"])
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "03_survival_by_class.png"), dpi=200)
plt.close()

# 4. Age distribution
plt.figure(figsize=(8, 5))
df["Age"].plot(kind="hist", bins=30, edgecolor="black")
plt.title("Age Distribution of Passengers")
plt.xlabel("Age")
plt.ylabel("Number of Passengers")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "04_age_distribution.png"), dpi=200)
plt.close()

# 5. Fare distribution
plt.figure(figsize=(8, 5))
df["Fare"].plot(kind="hist", bins=30, edgecolor="black")
plt.title("Fare Distribution")
plt.xlabel("Fare")
plt.ylabel("Number of Passengers")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "05_fare_distribution.png"), dpi=200)
plt.close()

# 6. Correlation heatmap for numeric variables
plt.figure(figsize=(9, 7))
numeric_df = df.select_dtypes(include="number")
plt.imshow(numeric_df.corr(), cmap="coolwarm", aspect="auto")
plt.colorbar()
plt.xticks(range(len(numeric_df.columns)), numeric_df.columns, rotation=45, ha="right")
plt.yticks(range(len(numeric_df.columns)), numeric_df.columns)
for i in range(len(numeric_df.columns)):
    for j in range(len(numeric_df.columns)):
        plt.text(j, i, f"{numeric_df.corr().iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
plt.title("Correlation Heatmap of Numerical Variables")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "06_correlation_heatmap.png"), dpi=200)
plt.close()

# Analysis summary
survival_rate = df["Survived"].mean() * 100
gender_rates = df.groupby("Sex")["Survived"].mean().mul(100).round(2)
class_rates = df.groupby("Pclass")["Survived"].mean().mul(100).round(2)

with open(os.path.join(OUTPUT_DIR, "analysis_summary.txt"), "w", encoding="utf-8") as f:
    f.write("Titanic EDA Summary\n")
    f.write("===================\n\n")
    f.write(f"Passengers analysed: {len(df)}\n")
    f.write(f"Overall survival rate: {survival_rate:.2f}%\n\n")
    f.write("Survival rate by gender:\n")
    for key, value in gender_rates.items():
        f.write(f"- {key}: {value:.2f}%\n")
    f.write("\nSurvival rate by passenger class:\n")
    for key, value in class_rates.items():
        f.write(f"- Class {key}: {value:.2f}%\n")
    f.write("\nKey observations:\n")
    f.write("- Female passengers had a substantially higher survival rate than male passengers.\n")
    f.write("- First-class passengers had a higher survival rate than second- and third-class passengers.\n")
    f.write("- Age and fare distributions show substantial variation across passengers.\n")

print("\nEDA completed. Outputs saved in:", OUTPUT_DIR)
