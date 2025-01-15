import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Load the dataset
data = pd.read_csv('URL-dataset.csv')  

# Normalize column names
data.columns = data.columns.str.strip().str.lower().str.replace(' ', '_')

# Handle missing or erroneous values
required_columns = ['url', 'is_malicious', 'domain', 'has_ip_address', 'path', 
                    'length', 'num_subdomains', 'has_special_char', 'tld']
data = data.dropna(subset=required_columns)  # Drop rows with missing required columns

# Convert 'is_malicious' column from TRUE/FALSE to 1/0
data['is_malicious'] = data['is_malicious'].map({'TRUE': 1, 'FALSE': 0})

# Drop rows with NaN values in 'is_malicious'
data = data.dropna(subset=['is_malicious'])

# Convert 'has_ip_address' and 'has_special_char' from 'yes/no' to 1/0
boolean_columns = ['has_ip_address', 'has_special_char']
for col in boolean_columns:
    data[col] = data[col].map({'Yes': 1, 'No': 0})
    data[col] = pd.to_numeric(data[col], errors='coerce')

# Drop rows where boolean columns contain NaN
data = data.dropna(subset=boolean_columns)

# Ensure numerical columns are of the correct data type
numerical_columns = ['length', 'num_subdomains']
scaler = MinMaxScaler()

# Convert numerical columns to numeric type and handle invalid values
for col in numerical_columns:
    data[col] = pd.to_numeric(data[col], errors='coerce')

# Drop rows with invalid numerical data (NaN values in numerical columns)
data = data.dropna(subset=numerical_columns)

# Normalize the 'length' and 'num_subdomains' columns
data[numerical_columns] = scaler.fit_transform(data[numerical_columns])

# Split the data into train, test, and validation sets
train_data, temp_data = train_test_split(
    data, test_size=0.3, stratify=data['is_malicious'], random_state=42
)
val_data, test_data = train_test_split(
    temp_data, test_size=0.5, stratify=temp_data['is_malicious'], random_state=42
)

# Populate the 'split' column
train_data['split'] = 'training'
val_data['split'] = 'validation'
test_data['split'] = 'test'

# Concatenate the split datasets back together for potential use
processed_data = pd.concat([train_data, val_data, test_data])

# Save the processed dataset to a new CSV file
processed_data.to_csv('processed_URL-datasetV2.csv', index=False)

print("Preprocessing complete. The dataset has been saved to 'processed_URL-dataset.csv'.")
