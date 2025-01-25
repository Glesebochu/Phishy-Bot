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
    #    But watch out for something like 'co.uk' -> subdomain logic can vary
    #    Example approach:
    if not has_ip_address:
        num_subdomains = domain.count('.') - 1 if domain.count('.') > 1 else 0
    else:
        num_subdomains = 0
    
    # 5. has_special_char (binary): if URL contains any non-alphanumeric or non-'.' or '-'
    #    This is just an example rule. Adapt to your training logic.
    if re.search(r"[^a-zA-Z0-9\.\-/:_]", url):
        has_special_char = 1
    else:
        has_special_char = 0
    
    # 6. Extract the TLD from domain (the part after the last '.')
    #    e.g., domain = "example.com" -> tld = "com"
    #          domain = "sub.theshoppe.com" -> tld = "theshoppe.com"
    parts = domain.split('.')
    # A naive approach: try the last two parts for known multi-part TLDs
    tld = None
    
    # Check from longest to shortest
    for i in range(len(parts)):
        candidate = ".".join(parts[i:])
        if candidate in TRAINING_TLDS:
            tld = candidate
            break
    
    if tld is None:
        # If we can't match, fallback to the single last part
        tld = parts[-1]  # might not match the training TLD list though
    
    # 7. Build a feature dictionary. Start with the core numeric ones:
    feature_dict = {
        "has_ip_address": [has_ip_address],
        "length": [url_length],
        "num_subdomains": [num_subdomains],
        "has_special_char": [has_special_char],
    }
    
    # 8. For each TLD in TRAINING_TLDS, add a binary column tldX
    #    1 if it matches the current URL's TLD, else 0
    for training_tld in TRAINING_TLDS:
        col_name = f"tld_{training_tld}_"
        if tld == training_tld:
            feature_dict[col_name] = [1]
        else:
            feature_dict[col_name] = [0]
    
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

