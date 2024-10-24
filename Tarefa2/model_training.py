import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression

# Load the training data
train_data = pd.read_csv('../datasets/train_radiomics_hipocamp.csv')

# Handle categorical variables: Encoding 'Transition' (target variable)
encoder = LabelEncoder()
train_data['Transition'] = encoder.fit_transform(train_data['Transition'])

# Automatically drop non-numeric columns
X_train = train_data.select_dtypes(include=[float, int]).drop(columns=['Transition'])
y_train = train_data['Transition']

# Load the test data
test_data = pd.read_csv('../datasets/test_radiomics_hipocamp.csv')

# Ensure the same preprocessing for the test data
X_test = test_data.select_dtypes(include=[float, int])

# Feature scaling (standardizing the data)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

# Predict on the test dataset
y_pred = model.predict(X_test_scaled)

# Add a sequential RowId column starting from 1
test_data['RowId'] = range(1, len(test_data) + 1)

# Save the predictions as 'Result'
test_data['Result'] = encoder.inverse_transform(y_pred)

# Save the predictions to a CSV file
test_data[['RowId', 'Result']].to_csv('predictions.csv', index=False)

print("Predictions saved to 'predictions.csv'")