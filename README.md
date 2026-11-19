# Credit Card Fraud Detection

This project explores the anonymized credit-card transaction dataset and compares
classification models for identifying fraudulent transactions. The exploratory
analysis and model experiments are in
[`Project_Introduction_to_DataScience.ipynb`](Project_Introduction_to_DataScience.ipynb).

## Dataset

The dataset is not committed to this repository. Download the
[Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
and place `creditcard.csv` at:

```text
content/creditcard.csv
```

Run notebook commands from the repository root. The notebook uses this
platform-independent relative path.

## Environment

Use Python 3.11 or newer:

```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# Windows PowerShell: .venv\\Scripts\\Activate.ps1
python -m pip install -r requirements.txt
jupyter notebook Project_Introduction_to_DataScience.ipynb
```

## Evaluation methodology

The preprocessing helpers in [`preprocessing.py`](preprocessing.py) enforce
the order required for a trustworthy holdout evaluation:

1. Split the original data into stratified training and test partitions.
2. Fit `StandardScaler` on training features only, then transform both partitions.
3. Apply SMOTE only to the scaled training partition.
4. Evaluate every model against the untouched, naturally imbalanced test partition.

This prevents information from the holdout set leaking into scaling statistics
or synthetic minority examples. The test set therefore reflects the original
class distribution rather than an artificially balanced one.

## Tests

The preprocessing regression tests use synthetic data and do not require the
Kaggle dataset:

```bash
python -m pip install -r requirements-test.txt
python -m unittest discover -s tests -v
```

GitHub Actions runs this suite for pull requests and pushes to `main`.
