import pandas as pd
import random
from joblib import load
import re
from urllib.parse import urlparse

# Load the serialized model and scaler
model = load('best_rf_model.joblib')
scaler = load('feature_scaler.joblib')

# Function to extract additional features from URLs
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

# Function to get input method
def get_user_choice():
    print("Choose an input method:")
    print("1. Enter input data manually.")
    print("2. Select one random malicious and one random non-malicious row from the dataset.")
    choice = input("Enter 1 or 2: ").strip()
    return choice

# Function to manually enter data
def get_manual_input():
    print("Enter the required data fields:")
    input_data = {
        'url': input("Enter URL: "),
        'length': float(input("Enter length: ")),
        'num_subdomains': float(input("Enter num_subdomains: ")),
        'has_ip_address': int(input("Enter has_ip_address (1 for Yes, 0 for No): ")),
        'has_special_char': int(input("Enter has_special_char (1 for Yes, 0 for No): "))
    }
    additional_features = extract_additional_features(input_data['url'])
    input_data.update(additional_features.to_dict())
    return pd.DataFrame([input_data])

# Function to select random rows from the dataset
def get_random_rows():
    # Load the dataset
    dataset = pd.read_csv('Preprocessed_URL_Dataset.csv')

    # Select one malicious and one non-malicious row
    malicious_row = dataset[dataset['is_malicious'] == 1].sample(n=1, random_state=random.randint(1, 100)).iloc[0]
    non_malicious_row = dataset[dataset['is_malicious'] == 0].sample(n=1, random_state=random.randint(1, 100)).iloc[0]

    # Extract additional features for each row
    malicious_features = extract_additional_features(malicious_row['url'])
    non_malicious_features = extract_additional_features(non_malicious_row['url'])

    # Combine features into complete rows
    malicious_row = pd.concat([malicious_row, malicious_features])
    non_malicious_row = pd.concat([non_malicious_row, non_malicious_features])
    
    print("\nRandom Malicious Row:")
    print(malicious_row)
    print("\nRandom Non-Malicious Row:")
    print(non_malicious_row)

    return pd.DataFrame([malicious_row, non_malicious_row])

def predict_url(url):
    additional_features = extract_additional_features(url)
    input_data = additional_features.to_dict()
    input_data['url'] = url

    # Ensure input_data matches the features the scaler and model expect
    expected_features = scaler.feature_names_in_
    for feature in expected_features:
        if feature not in input_data:
            input_data[feature] = 0  # Add missing features with default values
    input_data = pd.DataFrame([input_data])[expected_features]  # Keep only expected features

    # Scale the data
    input_data_scaled = scaler.transform(input_data)

    # Make predictions
    predictions = model.predict(input_data_scaled)
    predicted_probabilities = model.predict_proba(input_data_scaled)

    # Output the results
    print("\nData Row:")
    print(input_data.iloc[0].to_dict())
    print("Prediction: Malicious" if predictions[0] == 1 else "Prediction: Non-Malicious")
    print(f"Prediction Probabilities: {predicted_probabilities[0]}")

# Main prediction logic
def main():
    choice = get_user_choice()

    if choice == '1':
        new_data = get_manual_input()
        for url in new_data['url']:
            predict_url(url)
    elif choice == '2':
        new_data = get_random_rows()
        for url in new_data['url']:
            predict_url(url)
    else:
        print("Invalid choice. Exiting.")
        return

if __name__ == "__main__":
    main()
