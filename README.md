"This project builds an end-to-end machine learning system for detecting fraudulent transactions using temporal and behavioral features.

## Dataset
IEEE-CIS Fraud Detection (Kaggle)

## Pipeline

### 1. Exploratory Data Analysis
- Fraud distribution (~3%)
- Missing value analysis
- Temporal fraud patterns

### 2. Preprocessing
- Missing value handling
- Categorical encoding
- Data validation

### 3. Feature Engineering
- Time-based features (hour, day)
- Lag features (time since last transaction)
- Velocity features (transaction frequency)
- Behavioral statistics (amount patterns)

### 4. Model
- LightGBM classifier
- Handles class imbalance
- Probability-based predictions

### 5. Evaluation
- Precision, Recall, F1-score
- ROC-AUC evaluation

## Results
- Fraud Precision: 0.96
- Fraud Recall: 0.61
- F1-score: 0.74

## Next Steps
- Improve recall using threshold tuning
- Handle class imbalance (scale_pos_weight)
- Add SHAP explainability