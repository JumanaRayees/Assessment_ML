# Freight Rate Prediction Challenge

Machine learning solution for predicting freight rates from shipment, route, geographic, temporal, and market-related features.

## Overview

The goal of this project is to build a supervised machine learning regression model that predicts the freight rate (`posted_rate`) for individual loads.

The workflow includes:

1. Data cleaning and quality checks
2. Exploratory Data Analysis (EDA)
3. Feature engineering
4. Time-based train/validation splitting
5. Walk-forward validation
6. Extra Trees regression
7. Final model training on all development data
8. Prediction of 12,000 validation loads
9. December 2025 scenario predictions
10. Submission validation and scoring

---

## Dataset

The development dataset contains **48,000 labeled shipment records** covering:

**January 1, 2025 → October 31, 2025**

The target variable is:

```text
posted_rate
```

Main input variables include:

* Pickup and delivery locations
* Pickup and delivery coordinates
* Distance
* Equipment type
* Weight
* Date
* Market index
* Quote signal

The external validation dataset contains **12,000 loads** for final prediction.

---

## Project Structure

```text
project assessment/
│
├── Data/
│   ├── train-test.csv
│   ├── train-test_clean.csv
│   ├── features_ready.csv
│   └── december-chart-inputs.csv
│
├── src/
│   ├── data_cleaning.py
│   ├── eda.py
│   ├── feature_engineering.py
│   ├── train.py
│   ├── walk_forward_validation.py
│   ├── tune_extra_trees.py
│   ├── predict.py
│   ├── fix_december.py
│   ├── prepare_december_submission.py
│   └── score.py
│
├── outputs/
│   ├── figures/
│   ├── models/
│   └── validation/
│
├── scorer_results/
│   └── candidate_december.png
│
├── validation_predictions.csv
└── README.md
```

---

# 1. Data Cleaning

The raw `train-test.csv` dataset was inspected for:

* Duplicate rows
* Duplicate load IDs
* Missing values
* Incorrect data types
* Numerical outliers
* Invalid infinite values

### Data quality results

* Rows: **48,000**
* Duplicate rows: **0**
* Duplicate load IDs: **0**
* Missing `weight`: **592**
* Missing `market_index`: **374**

The date column was converted to a proper datetime format.

Missing numerical values were handled through **median imputation inside the model preprocessing pipeline**. This prevents information from the validation set from being used during preprocessing.

Extreme target values were investigated rather than automatically removed. High freight rates were associated with legitimate long-distance routes, so these observations were retained.

---

# 2. Exploratory Data Analysis

EDA was performed to understand the relationships between shipment characteristics and freight rates.

The strongest numerical relationship was between:

```text
distance → posted_rate
```

with a correlation of approximately:

```text
0.91
```

Equipment type also showed meaningful differences in average rates.

The EDA showed that freight rates are strongly influenced by route distance, while equipment type and geographic characteristics provide additional information.

EDA visualizations are saved in:

```text
outputs/figures/
```

---

# 3. Feature Engineering

A reusable feature engineering pipeline was implemented in:

```text
src/feature_engineering.py
```

The engineered features include:

### Date features

* Year
* Month
* Day
* Day of week
* Week of year
* Day of year
* Weekend indicator
* Month start indicator
* Month end indicator

### Route features

* Combined pickup → delivery route
* Route-specific categorical representation

### Geographic features

* Latitude difference
* Longitude difference
* Absolute latitude difference
* Absolute longitude difference
* Haversine geographic distance

### Distance features

* Log-transformed distance
* Distance bucket

### Weight features

* Log-transformed weight
* Weight bucket
* Weight per distance

### Market and quote features

* Market index deviation
* Quote signal deviation
* Squared quote signal

### Interaction features

* Distance × weight
* Distance × market index
* Distance × quote signal

Infinite values generated during transformations were converted to missing values before preprocessing.

---

# 4. Leakage Prevention

The target variable:

```text
posted_rate
```

was excluded from the model features.

The EDA-only variable:

```text
rate_per_mile
```

was also excluded from model training because it is derived from the target and would introduce target leakage.

The external `validation.csv` dataset was not used for model selection or hyperparameter tuning.

---

# 5. Validation Strategy

Because freight rates are time-dependent, a chronological validation strategy was used instead of a random train/test split.

The main development split was:

```text
Training:
January 1, 2025 → September 30, 2025
43,147 rows

Validation:
October 1, 2025 → October 31, 2025
4,853 rows
```

This setup simulates the real-world situation of training on historical loads and predicting future loads.

---

# 6. Walk-Forward Validation

To test whether model performance was stable across different time periods, expanding-window walk-forward validation was performed.

| Fold | Training Period | Validation Period |
| ---- | --------------- | ----------------- |
| 1    | Jan → Jun       | July              |
| 2    | Jan → Jul       | August            |
| 3    | Jan → Aug       | September         |
| 4    | Jan → Sep       | October           |

### Results

| Fold                |        MAE |       RMSE |         R² |
| ------------------- | ---------: | ---------: | ---------: |
| June → July         |     145.87 |     672.33 |     0.8004 |
| July → August       |     178.36 |     685.41 |     0.7840 |
| August → September  |     140.00 |     654.70 |     0.8153 |
| September → October |     152.71 |     703.20 |     0.7884 |
| **Mean**            | **154.24** | **678.91** | **0.7970** |

The results were reasonably consistent across the four monthly validation periods, with no severe performance collapse in any individual fold.

---

# 7. Machine Learning Model

The selected model is a:

**Supervised Learning → Regression → Extra Trees Regressor**

Extra Trees (Extremely Randomized Trees) was selected because it can model:

* Non-linear relationships
* Feature interactions
* Mixed numerical and categorical information
* Complex route-rate relationships

The final configuration is:

```text
n_estimators = 300
max_features = 0.8
min_samples_leaf = 2
max_depth = None
random_state = 42
n_jobs = -1
```

Categorical features were encoded using One-Hot Encoding.

Numerical features were median-imputed.

The preprocessing and model were combined into a reproducible pipeline.

---

# 8. Final Model Training

After validation, the final model was retrained using **all 48,000 labeled development records**.

The saved model is:

```text
outputs/models/final_extra_trees.joblib
```

The final model was then used to predict the external validation dataset containing 12,000 loads.

---

# 9. Final Validation Predictions

The final prediction file is:

```text
validation_predictions.csv
```

It contains exactly:

```text
load_id
predicted_rate
```

Validation checks confirmed:

* **12,000 rows**
* No missing predictions
* No duplicate IDs
* Correct load ID set
* Positive numeric predictions

Observed prediction range:

```text
Minimum: 154.13
Maximum: 10582.66
Mean: 2387.54
```

---

# 10. December 2025 Scenario

The December scenario uses the fixed shipment:

```text
Pickup: Lexington
Delivery: Fort Wayne
Distance: 360.0
Equipment: Dry Van
Weight: 32,000
```

Predictions were generated for all **31 days of December 2025**.

The final December file contains exactly these columns:

```text
pickup
delivery
distance
equipment
weight
date
predicted_rate
```

The scorer confirmed:

```text
Validated 31 fixed December predictions.
```

The generated chart is:

```text
scorer_results/candidate_december.png
```

---

# 11. Submission Validation

The final scoring script was executed successfully:

```bash
python src/score.py \
  --predictions validation_predictions.csv \
  --december-predictions Data/december-chart-inputs.csv
```

Result:

```text
Validated 12,000 final predictions.
Validated 31 fixed December predictions.
Created chart: scorer_results\candidate_december.png
Final validation metrics are calculated by Spotter after submission.
```

The final hidden validation metrics are therefore expected to be calculated by Spotter after submission.

---

# 12. Reproducibility

The main workflow can be reproduced using the scripts in `src/`.

### Clean the data

```bash
python src/data_cleaning.py
```

### Run EDA

```bash
python src/eda.py
```

### Generate engineered features

```bash
python src/feature_engineering.py
```

### Train the baseline model

```bash
python src/train.py
```

### Run walk-forward validation

```bash
python src/walk_forward_validation.py
```

### Generate final predictions

```bash
python src/predict.py
```

### Prepare December submission

```bash
python src/prepare_december_submission.py
```

### Validate the final submission

```bash
python src/score.py \
  --predictions validation_predictions.csv \
  --december-predictions Data/december-chart-inputs.csv
```

---

# 13. Key Takeaways

The final pipeline follows a chronological machine learning workflow:

```text
Raw Data
   ↓
Data Cleaning
   ↓
EDA
   ↓
Feature Engineering
   ↓
Chronological Validation
   ↓
Walk-Forward Validation
   ↓
Extra Trees Regressor
   ↓
Train on All Development Data
   ↓
12,000 Final Predictions
   ↓
December Predictions
   ↓
Submission Validation
```

The model achieved a mean walk-forward:

```text
MAE = 154.24
RMSE = 678.91
R² = 0.7970
```

on the development walk-forward evaluation.

The final submission files passed all required structural and validity checks.

---

## Author

**Jumana Al Rayes**

Data Science & Artificial Intelligence

Freight Rate Prediction Challenge
