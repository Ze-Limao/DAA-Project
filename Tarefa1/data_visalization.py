import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

train_data = pd.read_csv('../datasets/train_radiomics_hipocamp.csv')

print("Basic Information about the Dataset:")
print(train_data.info())

def plot_missing_values(data):
    missing_values = data.isnull().sum()
    missing_values = missing_values[missing_values > 0]
    missing_values.sort_values(inplace=True)

    if not missing_values.empty:
        plt.figure(figsize=(10, 6))
        sns.barplot(x=missing_values.index, y=missing_values)
        plt.xticks(rotation=90)
        plt.title('Missing Values by Feature')
        plt.xlabel('Features')
        plt.ylabel('Number of Missing Values')
        plt.show()
    else:
        print("No missing values detected in the dataset.")

plot_missing_values(train_data)

def plot_feature_distributions(data):
    numeric_features = data.select_dtypes(include=[np.number]).columns.tolist()
    num_plots = len(numeric_features)
    num_rows = (num_plots // 3) + 1

    plt.figure(figsize=(20, num_rows * 5))
    for i, feature in enumerate(numeric_features):
        plt.subplot(num_rows, 3, i + 1)
        sns.histplot(data[feature], kde=True, bins=30)
        plt.title(f'Distribution of {feature}')
    plt.tight_layout()
    plt.show()

plot_feature_distributions(train_data)

def plot_correlation_heatmap(data):
    plt.figure(figsize=(15, 12))
    correlation_matrix = data.corr()
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1)
    plt.title('Correlation Heatmap')
    plt.show()

plot_correlation_heatmap(train_data)

def plot_boxplots(data):
    numeric_features = data.select_dtypes(include=[np.number]).columns.tolist()
    num_plots = len(numeric_features)
    num_rows = (num_plots // 3) + 1

    plt.figure(figsize=(20, num_rows * 5))
    for i, feature in enumerate(numeric_features):
        plt.subplot(num_rows, 3, i + 1)
        sns.boxplot(x=data[feature])
        plt.title(f'Boxplot of {feature}')
    plt.tight_layout()
    plt.show()

plot_boxplots(train_data)

def plot_pairplot(data, features=None, target=None):
    if features is None:
        features = data.select_dtypes(include=[np.number]).columns.tolist()
    if target:
        sns.pairplot(data, vars=features, hue=target, palette='Set2')
    else:
        sns.pairplot(data, vars=features)
    plt.show()

plot_pairplot(train_data)

def display_summary_statistics(data):
    print("\nSummary Statistics of Numerical Features:")
    print(data.describe())

display_summary_statistics(train_data)

def plot_categorical_value_counts(data):
    categorical_features = data.select_dtypes(include=['object', 'category']).columns.tolist()
    num_plots = len(categorical_features)
    num_rows = (num_plots // 3) + 1

    plt.figure(figsize=(20, num_rows * 5))
    for i, feature in enumerate(categorical_features):
        plt.subplot(num_rows, 3, i + 1)
        sns.countplot(y=data[feature], order=data[feature].value_counts().index, palette='viridis')
        plt.title(f'Value Counts of {feature}')
    plt.tight_layout()
    plt.show()

plot_categorical_value_counts(train_data)
