from baseline_model import convert_to_binary, get_baseline_metrics
from baseline_modelV2 import extract_additional_features
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score
import pandas as pd
import joblib
import matplotlib.pyplot as plt

def run_advanced_model():
    # Load the dataset
    data = pd.read_csv('Preprocessed_URL_Dataset.csv')
    
    # Normalize column names
    data.columns = (
        data.columns.str.strip()
        .str.lower()
        .str.replace(' ', '_')
        .str.replace('[', '_')
        .str.replace(']', '_')
        .str.replace('<', '_')
        .str.replace('>', '_')
    )
    
    # Extract additional features from URL
    print("Extracting additional features...")
    url_features = data['url'].apply(extract_additional_features)
    data = pd.concat([data, url_features], axis=1)
    
    # Verify required columns
    required_columns = ['length', 'num_subdomains', 'has_ip_address', 'has_special_char', 'is_malicious']
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        raise ValueError(f"The following required columns are missing from the dataset: {missing_columns}")
    
    # Convert is_malicious to numeric
    data['is_malicious'] = data['is_malicious'].apply(convert_to_binary)
    
    # Handle missing values
    data = data.fillna(0)
    
    # Define features
    features = [
        'length', 'num_subdomains', 'has_ip_address', 'has_special_char',
        'special_chars_ratio', 'digits_ratio',
        'path_length', 'query_length', 'fragment_length',
        'dots_count', 'hyphens_count', 'has_suspicious_words',
        'has_hexadecimal', 'has_data_uri'
    ]
    
    # Drop columns with invalid data types for XGBoost
    invalid_columns = ['split', 'url', 'domain', 'path']
    data = data.drop(columns=invalid_columns, errors='ignore')
    
    # Drop TLD columns
    tld_columns = [col for col in data.columns if col.startswith('tld_')]
    data = data.drop(columns=tld_columns, errors='ignore')
    
    # Initialize the XGBoost classifier
    model = XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.7,
        random_state=42,
    )
    
    # Prepare the dataset for cross-validation
    X = data[features]
    y = data['is_malicious']
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Lists to store metrics for each fold
    precision_scores = []
    recall_scores = []
    f1_scores = []
    
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        # Train the model with early stopping
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            # early_stopping_rounds=10,
            verbose=False
        )
        
        # Predict on the test set
        y_pred = model.predict(X_test)
        
        # Evaluate the model
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # Store the metrics
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)
        
        print(f"Fold results - Precision: {precision}, Recall: {recall}, F1 Score: {f1}\n")
    
    # Calculate average metrics
    avg_precision = sum(precision_scores) / len(precision_scores)
    avg_recall = sum(recall_scores) / len(recall_scores)
    avg_f1 = sum(f1_scores) / len(f1_scores)
    
    print(f"Average Precision: {avg_precision}")
    print(f"Average Recall: {avg_recall}")
    print(f"Average F1 Score: {avg_f1}\n")
    
    # Get baseline metrics
    baseline_precision, baseline_recall, baseline_f1 = get_baseline_metrics()
    
    # Compare these metrics to your baseline
    print(f"Baseline Precision: {baseline_precision}")
    print(f"Baseline Recall: {baseline_recall}")
    print(f"Baseline F1 Score: {baseline_f1}\n")
    
    print(f"XGBoost Precision Improvement: {avg_precision - baseline_precision}")
    print(f"XGBoost Recall Improvement: {avg_recall - baseline_recall}")
    print(f"XGBoost F1 Score Improvement: {avg_f1 - baseline_f1}")
    
    # Now retrain the final model on ALL data (optional but recommended)
    model.fit(X, y)
    
    # Save the final model
    joblib.dump(model, "advanced_xgb_model.pkl")
    print("Final model retrained on the entire dataset and saved as 'advanced_xgb_model.pkl'.")
    
    # Plot feature importance
    xgb.plot_importance(model)
    plt.show()

def main():
    run_advanced_model()

if __name__ == "__main__":
    main()
