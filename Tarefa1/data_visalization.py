import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, classification_report
from scipy.stats import spearmanr
import shap

def advanced_radiomics_visualization(train_data):
    """
    Comprehensive visualization strategy for high-dimensional radiomics dataset
    
    Parameters:
    -----------
    train_data : pandas.DataFrame
        Input radiomics dataset
    """
    st.title("Advanced Radiomics Visualization Dashboard")
    
    st.header("1. Basic Dataset Overview")
    st.write("Dataset Shape:", train_data.shape)
    st.write("Column Types:", train_data.dtypes.value_counts())
    st.write("Target Variable Distribution:", train_data['Transition'].value_counts(normalize=True))
    
    numerical_features = train_data.select_dtypes(include=['float64', 'int64']).columns.tolist()
    numerical_features = [col for col in numerical_features if col != 'Transition']
    X = train_data[numerical_features]
    
    le = LabelEncoder()
    y = le.fit_transform(train_data['Transition'])
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    st.header("2. Principal Component Analysis (PCA)")
    pca = PCA(n_components=0.95)
    X_pca = pca.fit_transform(X_scaled)
    
    plt.figure(figsize=(10, 6))
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('PCA - Cumulative Variance Explained')
    st.pyplot()
    
    st.header("3. Feature Importance with Random Forest")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_scaled, y)
    
    feature_importances = pd.Series(rf.feature_importances_, index=numerical_features)
    top_features = feature_importances.nlargest(20)
    
    plt.figure(figsize=(12, 6))
    top_features.plot(kind='bar')
    plt.title('Top 20 Most Important Features')
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot()
    
    st.header("4. High-Dimensional Correlation Analysis")
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
    st.pyplot()
    
    st.header("5. Distribution of Top Features by Transition")
    plt.figure(figsize=(15, 10))
    for i, feature in enumerate(top_features.index[:6], 1):
        plt.subplot(2, 3, i)
        sns.boxplot(x='Transition', y=feature, data=train_data)
        plt.title(f'{feature} by Transition')
        plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot()
    
    st.header("6. Pairwise Feature Relationships")
    available_features = [feature for feature in top_features.index[:6] if feature in train_data.columns]
    sns.pairplot(train_data[available_features + ['Transition']], hue='Transition', palette='viridis')
    st.pyplot()
    
    st.header("7. Feature Distributions Across All Samples")
    plt.figure(figsize=(15, 10))
    for i, feature in enumerate(top_features.index[:6], 1):
        plt.subplot(2, 3, i)
        sns.kdeplot(train_data[feature], shade=True, color="blue", label='All Data')
        plt.title(f'Distribution of {feature}')
        plt.xlabel(feature)
        plt.ylabel('Density')
        plt.legend()
    plt.tight_layout()
    st.pyplot()
    
    st.header("8. t-SNE Visualization")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X_tsne = tsne.fit_transform(X_scaled)
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis', alpha=0.7)
    plt.colorbar(scatter, label='Transition Class')
    plt.title('t-SNE Visualization')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    st.pyplot()
    
    st.header("9. Class-wise PCA Clustering Heatmap")
    centroids = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(X_pca.shape[1])]).groupby(y).mean()
    sns.heatmap(centroids, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Class-wise PCA Centroids')
    st.pyplot()
    
    st.header("10. Class Balance")
    train_data['Transition'].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=sns.color_palette('viridis'))
    plt.title('Class Distribution')
    st.pyplot()
    
    st.header("11. Model Performance Metrics")
    y_pred = rf.predict(X_scaled)
    conf_matrix = confusion_matrix(y, y_pred)
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    st.pyplot()
    st.text("Classification Report:")
    st.text(classification_report(y, y_pred, target_names=le.classes_))
    
    st.header("12. SHAP Summary Plot")
    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X_scaled)
    shap.summary_plot(shap_values, pd.DataFrame(X_scaled, columns=numerical_features), plot_type="bar")
    st.pyplot()

train_data = pd.read_csv('../datasets/train_radiomics_hipocamp.csv')
advanced_radiomics_visualization(train_data)
