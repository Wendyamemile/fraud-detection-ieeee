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

### Step 6 - Model Improvement

- Handled class imbalance using scale_pos_weight
- Improved recall from 0.61 → 0.69
- Optimized decision threshold
- Evaluated using ROC-AUC and PR-AUC
- Added SHAP explainability for feature insights

## Results
- Precision (Fraud): 0.87  
- Recall (Fraud): 0.69  
- F1-score: 0.77  
- PR-AUC: ~0.81

## Key Improvements

- Better fraud detection recall
- Improved fraud recall (+8%)
- Better balance between precision and recall (tradeoff)
- Handled class imbalance
- Optimized decision threshold
- Added SHAP explainability for feature insights