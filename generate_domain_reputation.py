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

# Extract all domains (not just unique ones)
domains_list = data["domain"]

# Check if previous results exist
if os.path.exists("Credibility_scores.csv"):
    existing_data = pd.read_csv("Credibility_scores.csv")
else:
    existing_data = pd.DataFrame(columns=["domain", "credibility", "security_vendors"])

# Prepare lists for storing new results
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
for count, domain in enumerate(domains_list, start=1):
    # Skip already processed domains
    if domain in existing_data["domain"].values:
        print(f"Skipping {domain}, already processed.")
        continue

    try:
        # Handle cases with and without a port
        if ":" in domain:
            domain = domain.split(":")[0]  # Keep only the part before the colon

        # Determine if input is an IP or domain
        if domain.replace(".", "").isdigit():
            url = ip_url
        elif "." in domain:  # Valid domain pattern check
            url = domain_url
        else:
            # Invalid domain or IP, mark as suspicious
            print(f"Invalid domain: {domain}")
            domains.append(domain)
            credibility_scores.append(-1)  # -1 for invalid entries
            vendors.append(-1)
            continue

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
            malicius_count = analysis_stats.get("malicious", 0)
            score = 1 if malicius_count > 0 else 0  # 1 = Malicious, 0 = Benign
        else:
            score = -1  # Treat as invalid
            malicius_count = -1
            print(f"No data found for domain: {domain}")

        # Append results
        domains.append(domain)
        credibility_scores.append(score)
        vendors.append(malicius_count)

    except Exception as e:
        # Handle errors
        print(f"Error occurred for {domain}: {e}")

    # Enforce rate limiting after processing 4 requests
    if count % RATE_LIMIT == 0:
        print(f"Applying Delay...\n")
        time.sleep(DELAY)
        # Update the last request time
        with open(LAST_REQUEST_FILE, "w") as file:
            file.write(str(time.time()))
    # Print progress
    print(
        f"Processed {count}/{len(domains_list)}: {domain} -> {score}, {malicius_count}\n"
    )

# Create DataFrame with new results
credibility_df = pd.DataFrame(
    {"domain": domains, "credibility": credibility_scores, "security_vendors": vendors}
)

# Append new data to existing data
updated_data = pd.concat([existing_data, credibility_df], ignore_index=True)

# Save to Credibility_scores.csv
updated_data.to_csv("Credibility_scores.csv", index=False)

print("Credibility scores added and saved in 'Credibility_scores.csv'")
