import requests
import pandas as pd

# Input your VirusTotal API key here
API_KEY = 'ce36dd5b275914ab8fc9796e8484b1182524ad392d740e3d8cb2a7c5e825e0c2'
url = 'https://www.virustotal.com/api/v3/domains/'

# Load dataset
data = pd.read_csv('')  # Replace with your file path
domains = data['Domain']           # Replace 'Domain' with your column name

# Create empty list for scores
scores = []

# Fetch credibility scores
for domain in domains:
    try:
        headers = {"x-apikey": API_KEY}
        response = requests.get(url + domain, headers=headers)
        json_response = response.json()

        # Extract reputation score
        if 'data' in json_response:
            score = json_response['data']['attributes']['reputation']
            scores.append(score)
        else:
            scores.append('Error')  # Error handling for invalid responses
    except Exception as e:
        scores.append('Error')  # Error handling for API failure

# Add scores to dataframe
data['Credibility_Score'] = scores

# Save the updated data to a new Excel file
data.to_excel('Credibility_Scored.xlsx', index=False)
print("Scores added and saved as 'Credibility_Scored.xlsx'")