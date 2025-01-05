import requests
import pandas as pd
import time
import os
import json

# Input your VirusTotal API key here
API_KEY = "ce36dd5b275914ab8fc9796e8484b1182524ad392d740e3d8cb2a7c5e825e0c2"
domain_url = "https://www.virustotal.com/api/v3/domains/"
ip_url = "https://www.virustotal.com/api/v3/ip_addresses/"

# Load dataset with domains
data = pd.read_csv("Yan's_Domains.csv") 

# Extract unique domains
unique_domains = data["domain"].unique()[:5] 

# Prepare lists for storing results
domains = []
credibility_scores = []
vendors = []

# API request rate limit settings
RATE_LIMIT = 4  # Requests per minute
DELAY = 60  # Delay in seconds after every RATE_LIMIT requests
LAST_REQUEST_FILE = "last_request_time.txt"

# Check the last request time
if os.path.exists(LAST_REQUEST_FILE):
    with open(LAST_REQUEST_FILE, "r") as file:
        last_request_time = float(file.read().strip())
    elapsed_time = time.time() - last_request_time
    if elapsed_time < DELAY:
        time.sleep(DELAY - elapsed_time)

# Fetch credibility scores
for count, domain in enumerate(unique_domains, start=1):
    try:
        # Determine if input is an IP or domain
        url = ip_url if domain.replace('.', '').isdigit() else domain_url

        headers = {"x-apikey": API_KEY}
        response = requests.get(url + domain, headers=headers)

        try:
            json_response = response.json()        
        except json.decoder.JSONDecodeError:
            print("Invalid JSON response received")
            json_response = {}  # Handle as empty response
        if response.status_code != 200:
            if response.status_code == 401:
                print("Unauthorized: Check your API Key")
            elif response.status_code == 429:
                print("Rate limit exceeded. Try again later.")
            elif response.status_code == 400:
                print(f"Bad request for domain: {domain}, {response.text}")

        # Extract reputation score
        if "data" in json_response:
            analysis_stats = json_response["data"]["attributes"]["last_analysis_stats"]
            malicius_count= analysis_stats.get('malicious', 0)
            print(f"Analysis:{analysis_stats}")
            score = 1 if malicius_count > 0 else 0  # 1 = Malicious, 0 = Benign
        else:
            score = response.text
            print(f"No data found for domain: {domain}")

        # Append results
        domains.append(domain)
        credibility_scores.append(score)
        vendors.append(malicius_count)

    except Exception as e:
        # Handle errors
        print(f"Error occurred for {domain}: {e}")

    # Enforce rate limiting
    if count % RATE_LIMIT == 0:
        time.sleep(DELAY)
        # Update the last request time
        with open(LAST_REQUEST_FILE, "w") as file:
            file.write(str(time.time()))
    # Print progress
    print(f"Processed {count}/{len(unique_domains)}: {domain} -> {score}\n")

# Create DataFrame with results
credibility_df = pd.DataFrame({"domain": domains, "credibility": credibility_scores, "security_vendors": vendors})

# Save to Credibility_scores.csv
credibility_df.to_csv("Credibility_scores.csv", index=False)

print("Credibility scores added and saved in 'Credibility_scores.csv'")
