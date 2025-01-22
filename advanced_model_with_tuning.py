from baseline_model import convert_to_binary, get_baseline_metrics
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import precision_score, recall_score, f1_score
import pandas as pd

def run_model_with_tuning():
    # Print library versions for debugging
    print(f"xgboost version: {xgb.__version__}")
    print(f"scikit-learn version: {GridSearchCV.__module__.split('.')[0]} {GridSearchCV.__module__.split('.')[1]}")
    
    # Load the dataset
    data = pd.read_csv('Preprocessed_URL_Dataset.csv')
    
    # Normalize column names
    data.columns = data.columns.str.strip().str.lower().str.replace(' ', '_')
    
    # Drop columns with invalid data types for XGBoost
    invalid_columns = ['split', 'url', 'domain', 'path']
    data = data.drop(columns=invalid_columns, errors='ignore')
    
    # Verify required columns
    required_columns = ['length', 'num_subdomains', 'has_ip_address', 'is_malicious']
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        raise ValueError(f"The following required columns are missing from the dataset: {missing_columns}")
    
    # Convert is_malicious to numeric
    data['is_malicious'] = data['is_malicious'].apply(convert_to_binary)
    
    # Prepare the dataset for cross-validation
    X = data.drop('is_malicious', axis=1)
    y = data['is_malicious']
    
    # Define the parameter grid
    param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }
    
    # Initialize the XGBoost classifier
    model = XGBClassifier(random_state=42)
    
    # print(X.dtypes)  # Ensure all columns are numeric or categorical
    
    # Initialize GridSearchCV
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    print(f"CV type: {type(cv)}")
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=cv, scoring='f1', verbose=1, n_jobs=-1)
    
    # Fit GridSearchCV
    grid_search.fit(X, y)
    
    # Get the best model
    best_model = grid_search.best_estimator_
    
    # Lists to store metrics for each fold
    precision_scores = []
    recall_scores = []
    f1_scores = []
    
    # Perform cross-validation with the best model
    for train_index, test_index in StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        # Train the best model with early stopping
        best_model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            eval_metric="logloss",
            # early_stopping_rounds=10,
            verbose=False
        )
        
        # Predict on the test set
        y_pred = best_model.predict(X_test)
        
        # Evaluate the model
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # Store the metrics
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)
        
        print(f"Fold results - Precision: {precision}, Recall: {recall}, F1 Score: {f1}")
    
    # Calculate average metrics
    avg_precision = sum(precision_scores) / len(precision_scores)
    avg_recall = sum(recall_scores) / len(recall_scores)
    avg_f1 = sum(f1_scores) / len(f1_scores)
    
    print(f"Average Precision: {avg_precision}, Average Recall: {avg_recall}, Average F1 Score: {avg_f1}")
    
    # Get baseline metrics
    baseline_precision, baseline_recall, baseline_f1 = get_baseline_metrics()
    
    # Compare these metrics to your baseline to see if XGBoost outperforms it in raw form
    print(f"Baseline Precision: {baseline_precision}, Baseline Recall: {baseline_recall}, Baseline F1 Score: {baseline_f1}")
    print(f"XGBoost Precision Improvement: {avg_precision - baseline_precision}")
    print(f"XGBoost Recall Improvement: {avg_recall - baseline_recall}")
    print(f"XGBoost F1 Score Improvement: {avg_f1 - baseline_f1}")

def main():
    run_model_with_tuning()

if __name__ == "__main__":
    main()