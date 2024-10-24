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
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_columns)
    ]
)

# Fit and transform training data
X_train_processed = preprocessor.fit_transform(X_train)

# Ensure the same preprocessing for the test data
X_test_processed = preprocessor.transform(test_data)

# Train the Decision Tree Classifier
model = DecisionTreeClassifier(max_depth=5, random_state=42)
model.fit(X_train_processed, y_train)

# Identify features at the last level of the tree
n_nodes = model.tree_.node_count
children_left = model.tree_.children_left
children_right = model.tree_.children_right

# Find the last level nodes
last_level_features = set()
for node in range(n_nodes):
    if children_left[node] == children_right[node]:  # Leaf node
        feature_index = model.tree_.feature[node]
        if feature_index != -2:  # Not a leaf node
            last_level_features.add(feature_index)

# Convert indices back to feature names
features_to_drop = [numeric_columns[i] if i < len(numeric_columns) else 
                    preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_columns)[i - len(numeric_columns)]
                    for i in last_level_features]

# Drop last level features from the training and test sets
X_train_reduced = X_train.drop(columns=features_to_drop)
X_test_reduced = test_data.drop(columns=features_to_drop)

# Preprocess the reduced feature sets
X_train_processed_reduced = preprocessor.fit_transform(X_train_reduced)
X_test_processed_reduced = preprocessor.transform(X_test_reduced)

# Train the Decision Tree Classifier again with the reduced features
model_reduced = DecisionTreeClassifier(max_depth=5, random_state=42)
model_reduced.fit(X_train_processed_reduced, y_train)

# Predict on the reduced test dataset
y_pred_reduced = model_reduced.predict(X_test_processed_reduced)

# Add a sequential RowId column starting from 1
test_data['RowId'] = range(1, len(test_data) + 1)

# Save the new predictions as 'Result'
test_data['Result'] = encoder.inverse_transform(y_pred_reduced)

# Save the predictions to a CSV file
test_data[['RowId', 'Result']].to_csv('predictions_reduced.csv', index=False)

print("Predictions saved to 'predictions_reduced.csv'")

# Get the remaining feature names after dropping last level features
remaining_features = [col for col in X_train.columns if col not in features_to_drop]

# Visualize the decision tree for the reduced model
plt.figure(figsize=(20,10))
plot_tree(model_reduced, feature_names=remaining_features, class_names=encoder.classes_, filled=True)
plt.show()

# Calculate and print new accuracy
accuracy_reduced = accuracy_score(y_train, model_reduced.predict(X_train_processed_reduced))
print(f"Training Accuracy with Reduced Features: {accuracy_reduced:.4f}")
