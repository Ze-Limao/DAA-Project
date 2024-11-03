import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt

train_data = pd.read_csv('../datasets/train_radiomics_hipocamp.csv')

constant_columns = [col for col in train_data.columns if train_data[col].nunique() == 1]
train_data.drop(columns=constant_columns, inplace=True)

numerical_features = train_data.select_dtypes(include=['float64', 'int64']).columns.tolist()

output_dir = "outlier_visualizations"
os.makedirs(output_dir, exist_ok=True)

def plot_boxplots(data, features):
    for feature in features:
        plt.figure(figsize=(10, 5))
        sns.boxplot(x=data[feature])
        plt.title(f'Box Plot of {feature}')
        plt.savefig(f'{output_dir}/boxplot_{feature}.png')
        plt.close()

def plot_scatter_pairs(data, features):
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            plt.figure(figsize=(10, 5))
            sns.scatterplot(x=data[features[i]], y=data[features[j]], alpha=0.6)
            plt.title(f'Scatter Plot of {features[i]} vs {features[j]}')
            plt.xlabel(features[i])
            plt.ylabel(features[j])
            plt.savefig(f'{output_dir}/scatter_{features[i]}_vs_{features[j]}.png')
            plt.close()

print("📊 Generating visualizations...")
plot_boxplots(train_data, numerical_features)
plot_scatter_pairs(train_data, numerical_features)
print("✅ Visualizations saved.")

print("\n🎉 All tasks completed!")
