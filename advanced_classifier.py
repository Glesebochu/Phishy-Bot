import re
import joblib
import pandas as pd

# Load your trained model
model = joblib.load("advanced_xgb_model.pkl")

def extract_features_from_url(url: str) -> pd.DataFrame:
    """
    Extract the necessary numeric/categorical features from the URL.
    Return a 1-row DataFrame (or the same structure your model expects).
    """
    # Example features:
    has_ip = 1 if re.match(r"^(\d{1,3}\.){3}\d{1,3}", url) else 0
    num_subdomains = url.count('.') - 1 if has_ip == 0 else 0
    length = len(url)
    
    # Additional features
    has_https = 1 if url.startswith("https://") else 0
    num_digits = sum(c.isdigit() for c in url)
    num_special_chars = sum(not c.isalnum() for c in url)
    path_length = len(re.findall(r"/", url))
    query_length = len(re.findall(r"\?", url))
    fragment_length = len(re.findall(r"#", url))
    tld_length = len(url.split('.')[-1])
    

    data = {
        'length': [length],
        'num_subdomains': [num_subdomains],
        'has_ip_address': [has_ip],
        'has_https': [has_https],
        'num_digits': [num_digits],
        'num_special_chars': [num_special_chars],
        'path_length': [path_length],
        'query_length': [query_length],
        'fragment_length': [fragment_length],
        'tld_length': [tld_length]
    }

    df = pd.DataFrame(data)
    return df

def predict_url(url: str) -> str:
    """
    Given a URL, return "Malicious" or "Legitimate" (or similar labels).
    """
    # 1. Extract features
    features_df = extract_features_from_url(url)

    # 2. Predict
    prediction = model.predict(features_df)[0]  # 0 or 1

    # 3. Convert numeric prediction to a string label
    if prediction == 1:
        return "Malicious"
    else:
        return "Legitimate"
