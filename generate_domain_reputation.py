import requests
import pandas as pd
import time

# Input your VirusTotal API key here
API_KEY = "ce36dd5b275914ab8fc9796e8484b1182524ad392d740e3d8cb2a7c5e825e0c2"
url = "https://www.virustotal.com/api/v3/domains/"
# Load dataset with domains
data = pd.read_csv("Preprocessed_URL_Dataset.csv")

# Extract unique domains
unique_domains = data["domain"].unique()[
    :1
]  # Replace 'domain' with your actual column name

# Prepare lists for storing results
domains = []
credibility_scores = []

# API request rate limit settings
RATE_LIMIT = 4  # Requests per minute
DELAY = 60 / RATE_LIMIT  # Delay between requests

# Fetch credibility scores
for domain in unique_domains:
    try:
        headers = {"x-apikey": API_KEY}
        response = requests.get(url + domain, headers=headers)

        try:
            json_response = response.json()
        except requests.exceptions.JSONDecodeError:
            print("Invalid JSON response received")
            json_response = {}  # Handle as empty response

        if response.status_code == 401:
            print("Unauthorized: Check your API Key")
        elif response.status_code == 429:
            print("Rate limit exceeded. Try again later.")
        elif response.status_code != 200:
            print(f"Unexpected error: {response.status_code}")

        # Extract reputation score
        if "data" in json_response:
            reputation = json_response["data"]["attributes"]["reputation"]
            score = 1 if reputation < 0 else 0  # 1 = Malicious, 0 = Benign
        else:
            score = -1

        # Append results
        domains.append(domain)
        credibility_scores.append(score)

    except Exception as e:
        # Handle errors
        print("Error occurred for domain:", e)
        domains.append(domain)
        credibility_scores.append(0)  # Default to benign on errors
    
    # Enforce rate limiting
    time.sleep(DELAY)

# Create DataFrame with results
credibility_df = pd.DataFrame({"Domain": domains, "Credibility": credibility_scores})

# Save to domain_names.csv
credibility_df.to_csv("domain_names.csv", index=False)

print("Credibility scores (binary) added and saved in 'domain_names.csv'")
