import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV
from urllib.parse import urlparse
import re

def extract_additional_features(url):
    try:
        # Convert URL to string if it isn't already
        url = str(url)
        
        # Basic URL characteristics
        length = len(url)
        
        # Count special characters
        special_chars = sum(c in '!@#$%^&*()_+-=[]{}|;:,.<>?' for c in url)
        special_chars_ratio = special_chars / length if length > 0 else 0
        
        # Number of digits
        digits = sum(c.isdigit() for c in url)
        digits_ratio = digits / length if length > 0 else 0
        
        # Parse URL
        try:
            parsed = urlparse(url)
            path_length = len(parsed.path)
            query_length = len(parsed.query)
            fragment_length = len(parsed.fragment)
        except:
            path_length = query_length = fragment_length = 0
        
        # Count dots and hyphens
        dots = url.count('.')
        hyphens = url.count('-')
        
        # Suspicious patterns
        has_suspicious_words = int(bool(re.search(r'(login|account|secure|verify|update|confirm)', url.lower())))
        has_hexadecimal = int(bool(re.search(r'0x[0-9a-fA-F]+', url)))
        has_data_uri = int(bool(re.search(r'data:', url)))
        
        return pd.Series({
            'url_length': length,
            'special_chars_ratio': special_chars_ratio,
            'digits_ratio': digits_ratio,
            'path_length': path_length,
            'query_length': query_length,
            'fragment_length': fragment_length,
            'dots_count': dots,
            'hyphens_count': hyphens,
            'has_suspicious_words': has_suspicious_words,
            'has_hexadecimal': has_hexadecimal,
            'has_data_uri': has_data_uri
        })
    except:
        return pd.Series({
            'url_length': 0,
            'special_chars_ratio': 0,
            'digits_ratio': 0,
            'path_length': 0,
            'query_length': 0,
            'fragment_length': 0,
            'dots_count': 0,
            'hyphens_count': 0,
            'has_suspicious_words': 0,
            'has_hexadecimal': 0,
            'has_data_uri': 0
        })

# Load and preprocess data
data = pd.read_csv('Preprocessed_URL_Dataset.csv')
data.columns = data.columns.str.strip().str.lower().str.replace(' ', '_')

# Extract additional features from URL
print("Extracting additional features...")
url_features = data['url'].apply(extract_additional_features)
data = pd.concat([data, url_features], axis=1)

# Convert boolean values
def convert_to_binary(value):
    if isinstance(value, bool):
        return 1 if value else 0
    elif isinstance(value, str):
        return 1 if value.lower() == 'true' else 0
    elif isinstance(value, (int, float)):
        return 1 if value == 1 or value == 1.0 else 0
    return None

data['is_malicious'] = data['is_malicious'].apply(convert_to_binary)

# Handle missing values
data = data.fillna(0)

# Define features
features = [
    'length', 'num_subdomains', 'has_ip_address',
    'url_length', 'special_chars_ratio', 'digits_ratio',
    'path_length', 'query_length', 'fragment_length',
    'dots_count', 'hyphens_count', 'has_suspicious_words',
    'has_hexadecimal', 'has_data_uri'
]

# Split data
train_data = data[data['split'].str.capitalize() == 'Training']
val_data = data[data['split'].str.capitalize() == 'Validation']
test_data = data[data['split'].str.capitalize() == 'Testing']

# Prepare features and target
X_train = train_data[features]
y_train = train_data['is_malicious']
X_val = val_data[features]
y_val = val_data['is_malicious']
X_test = test_data[features]
y_test = test_data['is_malicious']

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Define Random Forest parameters for search
rf_params = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2'],
    'class_weight': ['balanced', 'balanced_subsample']
}

# Initialize Random Forest
rf = RandomForestClassifier(random_state=42)

# Perform RandomizedSearchCV
print("Performing hyperparameter tuning...")
rf_random = RandomizedSearchCV(
    estimator=rf,
    param_distributions=rf_params,
    n_iter=20,
    cv=3,
    random_state=42,
    n_jobs=-1,
    scoring='roc_auc'
)

# Fit the model
rf_random.fit(X_train_scaled, y_train)

# Get best model
best_rf = rf_random.best_estimator_

# Make predictions
val_pred = best_rf.predict(X_val_scaled)
test_pred = best_rf.predict(X_test_scaled)

# Calculate probabilities for ROC AUC
val_prob = best_rf.predict_proba(X_val_scaled)[:, 1]
test_prob = best_rf.predict_proba(X_test_scaled)[:, 1]

# Print results
print("\nBest Parameters:", rf_random.best_params_)

print("\nValidation Performance:")
print(classification_report(y_val, val_pred))
print(f"Validation ROC AUC: {roc_auc_score(y_val, val_prob):.4f}")

print("\nTest Performance:")
print(classification_report(y_test, test_pred))
print(f"Test ROC AUC: {roc_auc_score(y_test, test_prob):.4f}")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': features,
    'importance': best_rf.feature_importances_
})
feature_importance = feature_importance.sort_values('importance', ascending=False)

print("\nTop 10 Most Important Features:")
print(feature_importance.head(10))

# Save the model and scaler
import joblib
joblib.dump(best_rf, 'best_rf_model.joblib')
joblib.dump(scaler, 'feature_scaler.joblib')
print("\nModel and scaler saved as 'best_rf_model.joblib' and 'feature_scaler.joblib'")