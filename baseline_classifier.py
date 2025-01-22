import pandas as pd
from joblib import load

# Load the serialized model and scaler
model = load('best_rf_model.joblib')
scaler = load('feature_scaler.joblib')

# Get the features the scaler and model were trained on
expected_features = scaler.feature_names_in_

# Example new data for prediction
new_data = pd.DataFrame([{
    'length': 120,
    'num_subdomains': 2,
    'has_ip_address': 0,
    'has_special_char': 1,  # This might not be part of training
    'hyphens_count': 3,
    'special_chars_ratio': 0.15,
    'digits_ratio': 0.1,
    'path_length': 40,
    'url_length': 160,
    'dots_count': 4,
    'query_length': 20,
    'has_suspicious_words': 0,
    'fragment_length': 0,  # Add default value if missing
    'has_data_uri': 0,     # Add default value if missing
    'has_hexadecimal': 0   # Add default value if missing
}])

# Ensure new_data matches the expected features
# Add missing features with default values
for feature in expected_features:
    if feature not in new_data.columns:
        new_data[feature] = 0

# Remove any extra features not expected by the scaler
new_data = new_data[expected_features]

# Scale the features using the saved scaler
new_data_scaled = scaler.transform(new_data)

# Make predictions with the loaded model
predictions = model.predict(new_data_scaled)
predicted_probabilities = model.predict_proba(new_data_scaled)

# Output the results
print("Predictions:", predictions)
print("Prediction Probabilities:", predicted_probabilities)
