# Prodigy InfoTech – Task 2: Titanic Dataset EDA

## 📌 Task Objective

Perform **data cleaning and exploratory data analysis (EDA)** on a dataset such as the Titanic dataset. Explore relationships between variables and identify patterns and trends in the data.

## 📊 Dataset

This project uses the **Titanic `train.csv` dataset** supplied in the Prodigy InfoTech Task 2 reference repository.

- **Reference:** [Prodigy InfoTech Data Science Datasets – Task 2](https://github.com/Prodigy-InfoTech/data-science-datasets/tree/main/Task%202)
- **Original source:** Kaggle Titanic dataset
- **Rows:** 891
- **Columns:** 12 before cleaning
- **Target variable:** `Survived`

### Main variables

`PassengerId`, `Survived`, `Pclass`, `Name`, `Sex`, `Age`, `SibSp`, `Parch`, `Ticket`, `Fare`, `Cabin`, `Embarked`

## 🛠️ Technologies Used

- Python
- Pandas
- Matplotlib
- Seaborn

## 🧹 Data Cleaning

The following cleaning steps were performed:

1. Loaded the Titanic dataset using Pandas.
2. Checked the dataset shape, data types, missing values, and duplicate rows.
3. Filled missing `Age` values using the median age.
4. Filled missing `Embarked` values using the most frequent value (mode).
5. Removed the `Cabin` column because it contains a large proportion of missing values.
6. Checked and removed duplicate rows.
7. Saved the cleaned dataset as `output/cleaned_titanic.csv`.

## 🔎 Exploratory Data Analysis

The analysis explores:

- Overall survival distribution
- Relationship between gender and survival
- Relationship between passenger class and survival
- Passenger age distribution
- Fare distribution
- Correlations among numerical variables

## 📈 Visualizations

The script generates six visualizations in the `output/` folder:

1. `01_survival_distribution.png`
2. `02_survival_by_gender.png`
3. `03_survival_by_class.png`
4. `04_age_distribution.png`
5. `05_fare_distribution.png`
6. `06_correlation_heatmap.png`

## 💡 Key Findings

- The overall survival rate was approximately **38.38%**.
- Female passengers had a substantially higher survival rate than male passengers.
- First-class passengers had a higher survival rate than second- and third-class passengers.
- The passenger ages were concentrated around young and middle-adult ages, with children also represented.
- Fare values were strongly right-skewed, with a smaller number of passengers paying much higher fares.
- Passenger class and fare show a strong relationship, while survival is related to several passenger characteristics.

Exact calculated values are also saved in `output/analysis_summary.txt`.

## 📁 Project Structure

```text
Prodigy-InfoTech-Task-2-Titanic-EDA/
│
├── README.md
├── titanic_eda.py
├── requirements.txt
├── .gitignore
│
├── data/
│   └── train.csv
│
└── output/
    ├── 01_survival_distribution.png
    ├── 02_survival_by_gender.png
    ├── 03_survival_by_class.png
    ├── 04_age_distribution.png
    ├── 05_fare_distribution.png
    ├── 06_correlation_heatmap.png
    ├── cleaned_titanic.csv
    └── analysis_summary.txt
```

## ▶️ How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the analysis

```bash
python titanic_eda.py
```

The cleaned dataset and visualizations will be generated inside the `output/` folder.

## 🎓 Internship

**Organization:** Prodigy InfoTech  
**Program:** Data Science Internship  
**Task:** Task 2 – Data Cleaning and Exploratory Data Analysis

## 👨‍💻 Author

**Amit Mishra**  
B.Sc. Data Science and Data Analytics

## 📄 License

This repository is created for educational and internship purposes.
