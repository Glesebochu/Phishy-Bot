import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split

# Load the dataset
data = pd.read_csv('Unprocessed_URL_Dataset.csv')  

# Normalize column names
data.columns = data.columns.str.strip().str.lower().str.replace(' ', '_')

# Handle missing or erroneous values
required_columns = ['url', 'is_malicious', 'domain', 'has_ip_address', 'path', 
                    'length', 'num_subdomains', 'has_special_char', 'tld']
data = data.dropna(subset=required_columns)  # Drop rows with missing required columns

# Convert 'is_malicious' column from TRUE/FALSE to 1/0
data['is_malicious'] = data['is_malicious'].map({'TRUE': 1, 'FALSE': 0})

# Convert 'has_ip_address' and 'has_special_char' columns from yes/no to 1/0
data['has_ip_address'] = data['has_ip_address'].map({'yes': 1, 'no': 0})
data['has_special_char'] = data['has_special_char'].map({'yes': 1, 'no': 0})

# Ensure numerical columns are of the correct data type
numerical_columns = ['length', 'num_subdomains', 'has_ip_address', 'has_special_char']
for col in numerical_columns:
    data[col] = pd.to_numeric(data[col], errors='coerce')

# Drop rows with invalid numerical data (non-numeric entries converted to NaN)
data = data.dropna(subset=numerical_columns)

# Normalize 'length' and 'num_subdomains' columns
scaler = MinMaxScaler()
data[['length', 'num_subdomains']] = scaler.fit_transform(data[['length', 'num_subdomains']])

# Split the data into training, validation, and test sets (70% train, 15% validation, 15% test)
train_data, temp_data = train_test_split(data, test_size=0.3, stratify=data['is_malicious'], random_state=42)
val_data, test_data = train_test_split(temp_data, test_size=0.5, stratify=temp_data['is_malicious'], random_state=42)

# Populate the 'split' column
train_data['split'] = 'training'
val_data['split'] = 'validation'
test_data['split'] = 'test'

# Concatenate the split datasets back together for potential use
processed_data = pd.concat([train_data, val_data, test_data])

# Define feature columns and target
features = ['length', 'num_subdomains', 'has_ip_address', 'has_special_char']
target = 'is_malicious'

X_train = train_data[features]
y_train = train_data[target]
X_val = val_data[features]
y_val = val_data[target]
X_test = test_data[features]
y_test = test_data[target]

# Standardize features for logistic regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Train logistic regression model
log_reg = LogisticRegression(random_state=42, max_iter=1000)
log_reg.fit(X_train_scaled, y_train)

# Evaluate model on validation set
val_predictions = log_reg.predict(X_val_scaled)
print("Validation Performance:")
print(classification_report(y_val, val_predictions))

# Evaluate model on test set
test_predictions = log_reg.predict(X_test_scaled)
print("Test Performance:")
print(classification_report(y_test, test_predictions))

# Accuracy score for comparison
val_accuracy = accuracy_score(y_val, val_predictions)
test_accuracy = accuracy_score(y_test, test_predictions)

print(f"Validation Accuracy: {val_accuracy:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# Save the processed dataset with the 'split' column
processed_data.to_csv('Processed_URL_Dataset.csv', index=False)
print("Processed dataset saved as 'Processed_URL_Dataset.csv'.")
