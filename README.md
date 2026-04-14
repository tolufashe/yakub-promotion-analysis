# 🏢 Yakub Trading Group — Algorithmic Staff Promotion Audit

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Jupyter Notebook](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-red.svg)](https://scikit-learn.org/)

---

## 📖 Summary

When Abdullah Baba Yakub took over the Yakub Trading Group, one of the first things his staff told him was that the promotion system felt unfair. People believed who you knew, where you were from, or your gender mattered more than how hard you worked. Rather than dismiss or accept these claims, Abdullah commissioned a rigorous, data-driven investigation.

This project analyzes **38,312 employee records** spanning 19 variables — from performance scores and training results to gender, marital status, and state of origin. Four machine learning models were trained and compared, a full demographic bias audit was conducted, and the key drivers of promotion were ranked by statistical importance.

**The verdict: the promotion system is predominantly merit-based.** Meeting annual targets and performing well in training account for over 50% of what predicts a promotion. No statistically significant bias was found across gender, marital status, or foreign education. Two factors — state of origin and age — showed minor variation and are flagged for ongoing monitoring, but do not constitute evidence of active discrimination.

The best model (Gradient Boosting) achieved a **ROC-AUC of 0.908**, meaning it can correctly distinguish promoted from non-promoted employees with 90.8% reliability — strong enough to serve as a decision-support tool for future promotion cycles.

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Dataset](#-dataset)
- [Installation](#-installation)
- [Usage](#-usage)
- [Key Findings](#-key-findings)
- [Project Structure](#-project-structure)
- [Methodology](#-methodology)
- [Results](#-results)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🎯 Project Overview

This project addresses three critical business questions:

1. **Is the promotion system merit-based or biased?**
2. **What are the most important features driving promotion decisions?**
3. **Can we build a predictive model for promotion eligibility?**

### Business Context

Abdullah Baba Yakub, the newly appointed heir to the Yakub business dynasty, commissioned this analysis after staff raised concerns about promotion bias during his first open house. With 16 years of international business experience (including a Senior VP role at a US conglomerate), Abdullah sought a **scientific, data-driven approach** to either validate or refute these claims.

---

## 📊 Dataset

The dataset contains employee records with 19 variables:

| Feature | Description | Type |
|---|---|---|
| `EmployeeNo` | Unique employee identifier | String |
| `Division` | Department where employee works | Categorical |
| `Qualification` | Highest educational qualification | Categorical |
| `Gender` | Employee gender (Male/Female) | Binary |
| `Channel_of_Recruitment` | How employee was hired | Categorical |
| `Trainings_Attended` | Number of trainings attended | Numeric |
| `Year_of_birth` | Birth year | Numeric |
| `Last_performance_score` | Previous year's performance rating (0-14) | Numeric |
| `Year_of_recruitment` | Year joined company | Numeric |
| `Targets_met` | Whether annual targets were met (1/0) | Binary |
| `Previous_Award` | Previous award indicator (1/0) | Binary |
| `Training_score_average` | Average training evaluation score | Numeric |
| `State_Of_Origin` | Employee's state of origin | Categorical |
| `Foreign_schooled` | Whether education was abroad (Yes/No) | Binary |
| `Marital_Status` | Marital status | Categorical |
| `Past_Disciplinary_Action` | Disciplinary history (Yes/No) | Binary |
| `Previous_IntraDepartmental_Movement` | Department transfers (Yes/No) | Binary |
| `No_of_previous_employers` | Number of previous companies | Numeric |
| `Promoted_or_Not` | **Target: Whether employee was promoted (1/0)** | Binary |

**Dataset Statistics:**
- Total Records: 38,312 employees
- Promotion Rate: 8.5%
- Missing Data: 4.4% (Qualification column only)

---

## 🚀 Installation

### Prerequisites

- Python 3.9 or higher
- Git

### Setup Instructions

```bash
# 1. Clone the repository
git clone https://github.com/tolufashe/yakub-promotion-analysis.git
cd yakub-promotion-analysis

# 2. Create virtual environment
python -m venv .venv

# 3. Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Launch Jupyter
jupyter notebook
```

---

## 💻 Usage

### Run the Complete Analysis

```bash
# Navigate to the final notebook
jupyter notebook notebooks/final/Yakub_Promotion_Analysis_Enhanced.ipynb
```

### Use as Python Module

```python
from src.data_processing import load_and_clean_data
from src.feature_engineering import create_features
from src.models import train_gradient_boosting, evaluate_model

# Load and preprocess data
df = load_and_clean_data('data/raw/Promotion Dataset.csv')
df = create_features(df)

# Train model
X_train, X_test, y_train, y_test = prepare_data(df)
model = train_gradient_boosting(X_train, y_train)

# Evaluate
metrics = evaluate_model(model, X_test, y_test)
print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
```

### Run the Full Pipeline

```bash
python main.py
```

### Run Tests

```bash
pytest tests/
```

---

## 🔑 Key Findings

### ✅ Verdict: The System is Predominantly Merit-Based

Our comprehensive analysis found **no evidence of systematic bias** based on gender, marital status, or foreign education. The promotion system rewards performance.

### Top Promotion Drivers

| Rank | Feature | Importance | Interpretation |
|---|---|---|---|
| 1 | `Targets_met` | 0.260 | Meeting annual targets is the strongest predictor |
| 2 | `Training_score_average` | 0.254 | Training performance is critical |
| 3 | `Last_performance_score` | 0.091 | Annual ratings matter significantly |
| 4 | `Previous_Award` | 0.057 | Past recognition indicates future success |
| 5 | `Age` | 0.053 | Career stage has moderate influence |

### Bias Audit Results

| Factor | Correlation | Finding |
|---|---|---|
| Gender | -0.010 | ✅ No significant bias |
| Marital Status | -0.004 | ✅ No significant bias |
| Foreign Schooled | 0.003 | ✅ No significant bias |
| State of Origin | 0.032 | ⚠️ Minor variation — monitor |
| Age | -0.018 | ⚠️ Slight youth advantage — monitor |

### Best Model Performance

| Model | ROC-AUC | F1 Score | Accuracy |
|---|---|---|---|
| **Gradient Boosting** | **0.908** | **0.502** | **0.942** |
| Random Forest | 0.882 | 0.393 | 0.820 |
| Logistic Regression | 0.874 | 0.370 | 0.760 |
| Decision Tree | 0.851 | 0.319 | 0.682 |

---

## 📁 Project Structure

```
yakub-promotion-analysis/
├── 📁 data/                        # Dataset files
│   ├── raw/                        # Original, immutable data
│   ├── processed/                  # Cleaned, transformed data
│   └── README.md                   # Data dictionary
│
├── 📁 notebooks/                   # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_model_training.ipynb
│   └── final/
│       └── Yakub_Promotion_Analysis_Enhanced.ipynb
│
├── 📁 src/                         # Reusable Python modules
│   ├── __init__.py
│   ├── data_processing.py          # Data cleaning functions
│   ├── feature_engineering.py      # Feature creation functions
│   ├── models.py                   # Model training functions
│   └── evaluation.py               # Evaluation metrics
│
├── 📁 models/                      # Saved model files (.pkl generated on run)
│
├── 📁 reports/                     # Generated reports
│   └── figures/                    # Plots and visualizations (generated on run)
│
├── 📁 tests/                       # Unit tests
│   └── test_data_processing.py
│
├── .gitignore
├── LICENSE
├── README.md
├── main.py                         # Runs the full pipeline end-to-end
├── requirements.txt
└── setup.py
```

> **Note:** The `models/` and `reports/figures/` folders contain only a `.gitkeep` placeholder in the repo. The actual `.pkl` model files and chart images are generated locally when you run the analysis.

---

## 🔬 Methodology

### Data Preprocessing

1. **Missing Value Treatment** — Missing qualifications encoded as `'Unknown'` rather than dropped, preserving all 38,312 records
2. **Feature Engineering** — Created `Age`, `Years_at_Company`, and `State_Score` from existing columns
3. **Encoding** — Binary, ordinal, and one-hot encoding applied to categorical variables
4. **Train-Test Split** — 80/20 stratified split to preserve the 8.5% promotion class distribution

### Modeling Approach

1. **Algorithms Tested** — Logistic Regression, Decision Tree, Random Forest, Gradient Boosting
2. **Class Imbalance Handling** — Used `class_weight='balanced'` to prevent the model from simply predicting "not promoted" for every employee
3. **Feature Scaling** — StandardScaler applied for distance-based algorithms
4. **Evaluation Metrics** — ROC-AUC (primary), F1 Score, Precision, Recall

### Bias Detection

1. **Correlation Analysis** — Examined Pearson correlation between each demographic variable and promotion outcome
2. **Feature Importance** — Ranked drivers using Gradient Boosting's built-in importance scores
3. **Statistical Testing** — Chi-square test for independence between categorical demographics and promotion

---

## 📈 Results

### Model Comparison

> ROC curve chart will appear here after running the analysis and committing `reports/figures/roc_curves.png`

### Feature Importance

> Feature importance chart will appear here after running the analysis and committing `reports/figures/feature_importance.png`

### Departmental Analysis

| Division | Promotion Rate | Employee Count |
|---|---|---|
| Information Technology | 10.7% | 4,952 |
| Sourcing and Purchasing | 9.7% | 5,052 |
| Information and Strategy | 9.4% | 3,721 |
| Customer Support | 8.8% | 7,973 |
| Commercial Sales | 7.2% | 11,695 |

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. **Fork** the repository
2. Create a **feature branch** (`git checkout -b feature/your-feature`)
3. **Commit** your changes (`git commit -m 'Add your feature'`)
4. **Push** to the branch (`git push origin feature/your-feature`)
5. Open a **Pull Request**

Please read [CONTRIBUTING.md](https://github.com/tolufashe/yakub-promotion-analysis/blob/main/CONTRIBUTING.md) for detailed guidelines.

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](https://github.com/tolufashe/yakub-promotion-analysis/blob/main/LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Yakub Trading Group** for providing the dataset
- **Abdullah Baba Yakub** for commissioning this analysis
- **Data Science Community** for tools and best practices

---

## 📧 Contact

For questions or feedback, please open an issue or contact:

- Email: [tolufashejohn@gmail.com](mailto:tolufashejohn@gmail.com)
- GitHub: [@tolufashe](https://github.com/tolufashe)

---

*Built with Python, Pandas, Scikit-Learn, and Jupyter*
