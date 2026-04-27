# Temporal Fraud Detection

This project focuses on detecting fraudulent transactions using temporal feature engineering and gradient boosting models.

## Dataset
IEEE-CIS Fraud Detection (Kaggle)

## Structure
- src/ → pipeline code
- notebooks/ → EDA & experiments
- outputs/ → generated reports

## Step 1
EDA with time-based analysis (fraud by hour, missing values, imbalance)

## Step 2 - Preprocessing Pipeline

- Data validation checks
- Missing value handling
- Categorical encoding
- Basic feature engineering (time-based)

Pipeline implemented in src/preprocessing.py

## Step 3 - Feature Engineering

- Time-based features (hour, day)
- Lag features (time since last transaction)
- Velocity features (transaction counts)
- Behavioral features (amount statistics)

This step transforms raw transactions into behavioral fraud signals.