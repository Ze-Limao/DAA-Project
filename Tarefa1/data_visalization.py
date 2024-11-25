import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy.stats import spearmanr

def advanced_radiomics_visualization(train_data):
    """
    Comprehensive visualization strategy for high-dimensional radiomics dataset
    
    Parameters:
    -----------
    train_data : pandas.DataFrame
        Input radiomics dataset
    """
    plt.rcParams["figure.figsize"] = (12, 8)
    
    # 1. Basic Dataset Overview
    print("Dataset Shape:", train_data.shape)
    print("\nColumn Types:\n", train_data.dtypes.value_counts())
    print("\nTarget Variable Distribution:\n", train_data['Transition'].value_counts(normalize=True))
    
    # 2. Feature Selection and Dimensionality Reduction Techniques
    # Select numerical features
    numerical_features = train_data.select_dtypes(include=['float64', 'int64']).columns.tolist()
    numerical_features = [col for col in numerical_features if col != 'Transition']
    X = train_data[numerical_features]
    
    # Encode target variable
    le = LabelEncoder()
    y = le.fit_transform(train_data['Transition'])
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 3. Principal Component Analysis (PCA)
    pca = PCA(n_components=0.95)  # Retain 95% of variance
    X_pca = pca.fit_transform(X_scaled)
    
    plt.figure(figsize=(10, 6))
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('PCA - Cumulative Variance Explained')
    plt.show()
    
    # 4. Feature Importance with Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_scaled, y)
    
    # Top 20 most important features
    feature_importances = pd.Series(rf.feature_importances_, index=numerical_features)
    top_features = feature_importances.nlargest(20)
    
    plt.figure(figsize=(12, 6))
    top_features.plot(kind='bar')
    plt.title('Top 20 Most Important Features')
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
    
    # 5. High-Dimensional Correlation Analysis
    # Use Spearman correlation for non-linear relationships
    correlation_matrix = np.zeros((len(top_features), len(top_features)))
    for i, feat1 in enumerate(top_features.index):
        for j, feat2 in enumerate(top_features.index):
            correlation_matrix[i, j], _ = spearmanr(X[feat1], X[feat2])
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, 
                xticklabels=top_features.index, 
                yticklabels=top_features.index, 
                cmap='coolwarm', 
                center=0, 
                annot=True, 
                fmt='.2f', 
                linewidths=0.5)
    plt.title('Correlation Heatmap of Top 20 Features')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
    
    # 6. Distribution of Top Features by Transition
    plt.figure(figsize=(15, 10))
    for i, feature in enumerate(top_features.index[:6], 1):
        plt.subplot(2, 3, i)
        sns.boxplot(x='Transition', y=feature, data=train_data)
        plt.title(f'{feature} by Transition')
        plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    # 7. 2D Visualization of First Two Principal Components
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], 
                c=y, 
                cmap='viridis', 
                alpha=0.7)
    plt.colorbar(scatter, label='Transition Class')
    plt.title('2D PCA Visualization with Transition')
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    
    # Custom legend with actual class names
    plt.show()
    
    # 8. Statistical Summary of Top Features
    top_features_df = train_data[list(top_features.index) + ['Transition']]
    print("\nStatistical Summary of Top Features by Transition:")
    print(top_features_df.groupby('Transition').mean())

    # 9. Separate Legend for Classes
    plt.figure(figsize=(8, 6))
    for i, cls in enumerate(le.classes_):
        plt.scatter([], [], c=plt.cm.viridis(i / (len(le.classes_) - 1)), label=cls)
    plt.legend(title='Transition Classes', loc='center')
    plt.axis('off')
    plt.title('Transition Classes Color Legend')
    plt.tight_layout()
    plt.show()

# Usage
train_data = pd.read_csv('../datasets/train_radiomics_hipocamp.csv')
advanced_radiomics_visualization(train_data)