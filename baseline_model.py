import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Load the dataset
data = pd.read_csv('Preprocessed_URL_Dataset.csv')

# Normalize column names
data.columns = data.columns.str.strip().str.lower().str.replace(' ', '_')

# Verify required columns
required_columns = ['length', 'num_subdomains', 'has_ip_address', 'is_malicious', 'split']
missing_columns = [col for col in required_columns if col not in data.columns]
if missing_columns:
    raise ValueError(f"The following required columns are missing from the dataset: {missing_columns}")

# Print unique values in is_malicious column for debugging
print("Unique values in is_malicious column:", data['is_malicious'].unique())

# Convert is_malicious to numeric
def convert_to_binary(value):
    if isinstance(value, bool):
        return 1 if value else 0
    elif isinstance(value, str):
        return 1 if value.lower() == 'true' else 0
    elif isinstance(value, (int, float)):
        return 1 if value == 1 or value == 1.0 else 0
    return None

data['is_malicious'] = data['is_malicious'].apply(convert_to_binary)

# Verify conversion
print("Unique values after conversion:", data['is_malicious'].unique())

# Ensure 'has_ip_address' is numeric
data['has_ip_address'] = pd.to_numeric(data['has_ip_address'], errors='coerce')

# Handle missing or invalid rows
initial_row_count = len(data)
data = data.dropna(subset=required_columns)
final_row_count = len(data)

if final_row_count == 0:
    raise ValueError("No valid data available after preprocessing. Exiting.")
elif final_row_count < initial_row_count:
    print(f"Warning: {initial_row_count - final_row_count} rows were dropped due to missing or invalid values.")

# Validate and standardize the 'split' column
data['split'] = data['split'].str.capitalize()
valid_splits = {'Training', 'Testing', 'Validation'}
data = data[data['split'].isin(valid_splits)]
if data.empty:
    raise ValueError("No data available after filtering for valid 'split' values (Training, Testing, Validation).")

# Split data based on the 'split' column
train_data = data[data['split'] == 'Training']
val_data = data[data['split'] == 'Validation']
test_data = data[data['split'] == 'Testing']

# Print split sizes for debugging
print(f"\nDataset splits:")
print(f"Training samples: {len(train_data)}")
print(f"Validation samples: {len(val_data)}")
print(f"Testing samples: {len(test_data)}")

# Ensure the splits are not empty
if train_data.empty or test_data.empty:
    raise ValueError("Training or testing split is empty. Check the 'split' column in the dataset.")

# Define feature columns and target
features = ['length', 'num_subdomains', 'has_ip_address']
target = 'is_malicious'

X_train = train_data[features]
y_train = train_data[target]
X_test = test_data[features]
y_test = test_data[target]

# Initialize validation sets
X_val = pd.DataFrame()
y_val = pd.Series(dtype=float)

if not val_data.empty:
    X_val = val_data[features]
    y_val = val_data[target]

# Train logistic regression model
log_reg = LogisticRegression(random_state=42, max_iter=1000)
log_reg.fit(X_train, y_train)

# Evaluate model on validation set if available
if not val_data.empty:
    val_predictions = log_reg.predict(X_val)
    print("\nValidation Performance:")
    print(classification_report(y_val, val_predictions))
    val_accuracy = accuracy_score(y_val, val_predictions)
    print(f"Validation Accuracy: {val_accuracy:.4f}")

# Evaluate model on test set
test_predictions = log_reg.predict(X_test)
print("\nTest Performance:")
print(classification_report(y_test, test_predictions))
test_accuracy = accuracy_score(y_test, test_predictions)
print(f"Test Accuracy: {test_accuracy:.4f}")

print("\nProcessed dataset saved as 'Processed_URL_DatasetV2.csv'.")