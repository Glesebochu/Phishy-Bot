import re
import joblib
import pandas as pd
from urllib.parse import urlparse

# Load your trained model
model = joblib.load("advanced_xgb_model.pkl")

# ------------------------------------------------------
# 1) List of TLDs from your training data:
TRAINING_TLDS = [
    "org",
    "com",
    "theshoppe.com",
    "it",
    "blogspot.com",
    "dcs.st-and.ac.uk",
    "03.ibm.com",
    "allthelyrics.com",
    "allvoices.com",
    "allwwewrestlers.com",
    "allyoucanread.com",
    "alpsroads.net",
    "alternatehistory.com",
    "alternet.org",
    "alterthepress.com",
    "altfg.com",
    "altiusdirectory.com",
    "altoonamirror.com",
    "altosport.com",
    "altpress.com",
    "alumni.net",
    "alumniclass.com",
    "alwaysontherun.net",
    "amazon.ca",
    "amazon.co.uk",
    "amazon.com",
    "mylife.com",
    "mylifeofcrime.wordpress.com",
    "mylifetime.com",
    "mylocalservices.us",
    "mylovedpornstars.com",
    "mymovies.net",
    "mynewplace.com",
    "mynhldraft.com",
    "mynhltraderumors.com",
    "ottawasun.com",
    "ougrizzlies.com",
    "ourairports.com",
    "ourbis.ca",
    "ourfaves.com",
    "ourhistory.canadiens.com",
    "youtube.com",
    "177.22.179",
    "nl",
    "000webhostapp.com",
    "171.169.193:35516",
    "net",
    "de",
    "248.170.218",
    "top",
    "ru",
    "info",
    "com.br",
    "200.14.110"
]
# TRAINING_TLDS = [
#     "com", "org", "com.br", "it", "us", "net", "co.uk", "de", "ru", "info", "top", "nl", "ca", "gov", 
#      "ac.uk", "co.uk", "wordpress.com", 
#     # etc. Add the entire list your model expects
# ]

def extract_features_from_url(url: str) -> pd.DataFrame:
    """
    Extract the exact numeric/binary features your model was trained on.
    Return a 1-row DataFrame with columns matching the training phase.
    """
    # 1. Parse the URL to get domain, TLD, etc.
    parsed = urlparse(url)
    domain = parsed.netloc.lower()  # e.g., "81.17.25.149" or "example.com"
    path_query_fragment = parsed.path + parsed.query + parsed.fragment
    
    # 2. has_ip_address (binary)
    ip_match = re.match(r"^(\d{1,3}\.){3}\d{1,3}$", domain)
    has_ip_address = 1 if ip_match else 0
    
    # 3. length (full URL length)
    url_length = len(url)
    
    # 4. num_subdomains (count '.' minus 1 if not IP) 
    if not has_ip_address:
        num_subdomains = domain.count('.') - 1 if domain.count('.') > 1 else 0
    else:
        num_subdomains = 0
    
    # 5. has_special_char (binary): if URL contains any non-alphanumeric or non-'.' or '-'
    if re.search(r"[^a-zA-Z0-9\.\-/:_]", url):
        has_special_char = 1
    else:
        has_special_char = 0
    
    # Additional features
    special_chars_ratio = sum(1 for c in url if not c.isalnum() and c not in '.-') / len(url)
    digits_ratio = sum(1 for c in url if c.isdigit()) / len(url)
    path_length = len(parsed.path)
    query_length = len(parsed.query)
    fragment_length = len(parsed.fragment)
    dots_count = url.count('.')
    hyphens_count = url.count('-')
    has_suspicious_words = int(any(word in url for word in ['login', 'secure', 'account', 'update', 'free', 'lucky', 'bonus']))
    has_hexadecimal = int(bool(re.search(r'%[0-9a-fA-F]{2}', url)))
    has_data_uri = int(url.startswith('data:'))

    # Extract TLD
    tld = domain.split('.')[-1]
    tld_org_ = int(tld == 'org')
    tld_com_ = int(tld == 'com')
    tld_net_ = int(tld == 'net')
    other_tlds = int(tld not in ['org', 'com', 'net'])

    # Build a feature dictionary
    feature_dict = {
        "length": [url_length],
        "num_subdomains": [num_subdomains],
        "has_ip_address": [has_ip_address],
        "has_special_char": [has_special_char],
        "special_chars_ratio": [special_chars_ratio],
        "digits_ratio": [digits_ratio],
        "path_length": [path_length],
        "query_length": [query_length],
        "fragment_length": [fragment_length],
        "dots_count": [dots_count],
        "hyphens_count": [hyphens_count],
        "has_suspicious_words": [has_suspicious_words],
        "has_hexadecimal": [has_hexadecimal],
        "has_data_uri": [has_data_uri],
        "tld_org_": [tld_org_],
        "tld_com_": [tld_com_],
        "tld_net_": [tld_net_],
        "other_tlds": [other_tlds],
    }
    
    # Convert to DataFrame
    df = pd.DataFrame(feature_dict)
    return df


def predict_url(url: str, threshold: float = 0.5) -> str:
    """
    Given a URL, produce a prediction using the model that
    expects features like [has_ip_address, length, num_subdomains, 
    has_special_char, tldorg, tldcom, ...].
    The threshold parameter allows adjusting the sensitivity of the classification.
    """
    # 1. Extract features with the matching schema
    features_df = extract_features_from_url(url)

    # 2. Predict probability
    prediction_proba = model.predict_proba(features_df)[0]  # [prob_legitimate, prob_malicious]

    # 3. Convert numeric prediction to a string label with probability
    prob_malicious = prediction_proba[1]
    if prob_malicious >= threshold:
        return f"Malicious ({prob_malicious:.2f})"
    else:
        return f"Legitimate ({prob_malicious:.2f})"


def main():
    test_urls = [
        "http://www.google.com",
        "http://malicious-site.com",
        "http://192.168.0.1",
        "http://subdomain.example.com",
        "http://example.com/path?query=1"
    ]
    
    threshold = 0.5  # Adjust the threshold as needed
    for url in test_urls:
        result = predict_url(url, threshold)
        print(f"URL: {url} -> Prediction: {result}")

if __name__ == "__main__":
    main()

