import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv('datasets/train_radiomics_hipocamp.csv')

# Check for missing values - no missing values
missing_values = data.isnull().sum()
print("Missing values in each column:\n", missing_values[missing_values > 0])  # Show only columns with missing values

# Visualize the distribution of the target variable (Transition)
plt.figure(figsize=(10,6))
sns.countplot(data['Transition'], palette='coolwarm')
plt.title('Distribution of Cognitive Status Transitions', fontsize=16)
plt.xlabel('Transition', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.xticks(rotation=45)
plt.show()

# Inspect the dataset's basic statistics
print("Dataset basic statistics:\n")
print(data.describe())

# Additional info on data types and other attributes
print("\nDataset information:")
data.info()
