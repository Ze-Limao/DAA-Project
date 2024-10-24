import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load the training data
train_data = pd.read_csv('../datasets/train_radiomics_hipocamp.csv')

# Handle the target variable: Encoding 'Transition'
encoder = LabelEncoder()
train_data['Transition'] = encoder.fit_transform(train_data['Transition'])

# Separate features and target
X_train = train_data.drop(columns=['Transition'])
y_train = train_data['Transition']

# Load the test data
test_data = pd.read_csv('../datasets/test_radiomics_hipocamp.csv')

# Ensure both training and test sets have the same columns
missing_cols_in_test = set(X_train.columns) - set(test_data.columns)
missing_cols_in_train = set(test_data.columns) - set(X_train.columns)

# Add missing columns to the test dataset and fill them with 0 (or another appropriate value)
for col in missing_cols_in_test:
    test_data[col] = 0

# Add missing columns to the training dataset and fill them with 0 (if necessary)
for col in missing_cols_in_train:
    X_train[col] = 0

# Ensure the same column order between train and test
X_train = X_train[test_data.columns]

# Identify categorical and numeric columns
categorical_columns = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
numeric_columns = X_train.select_dtypes(include=[float, int]).columns.tolist()

# Preprocessing: One-hot encode categorical features and scale numeric features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_columns),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_columns)])  # Updated line

# Fit and transform training data
X_train_processed = preprocessor.fit_transform(X_train)

# Ensure the same preprocessing for the test data
X_test_processed = preprocessor.transform(test_data)

# Train the Decision Tree Classifier
model = DecisionTreeClassifier(max_depth=5, random_state=42)  # You can adjust the depth
model.fit(X_train_processed, y_train)

# Predict on the test dataset
y_pred = model.predict(X_test_processed)

# Add a sequential RowId column starting from 1
test_data['RowId'] = range(1, len(test_data) + 1)

# Save the predictions as 'Result'
test_data['Result'] = encoder.inverse_transform(y_pred)

# Save the predictions to a CSV file
test_data[['RowId', 'Result']].to_csv('predictions.csv', index=False)

print("Predictions saved to 'predictions.csv'")

# Visualize the decision tree
plt.figure(figsize=(20,10))
feature_names = numeric_columns + list(preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_columns))
plot_tree(model, feature_names=feature_names, class_names=encoder.classes_, filled=True)
plt.show()

# Feature importance
importances = model.feature_importances_
feature_importances = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_importances = feature_importances.sort_values(by='Importance', ascending=False)

# Display feature importance
print(feature_importances)

# Calculate and print accuracy
accuracy = accuracy_score(y_train, model.predict(X_train_processed))
print(f"Training Accuracy: {accuracy:.4f}")
