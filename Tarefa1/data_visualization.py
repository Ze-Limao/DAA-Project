import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import TSNE
from sklearn.feature_selection import mutual_info_classif
from scipy.stats import spearmanr
from sklearn.cluster import KMeans

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
    st.markdown("""
        This section provides a high-level overview of the dataset, including its shape, 
        the distribution of data types, and the class balance for the `Transition` variable.
        Understanding the basic structure and balance of the dataset is crucial for further analysis.
    """)
    st.write("Dataset Shape:", train_data.shape)
    st.write("Column Types:", train_data.dtypes.value_counts())
    st.write("Target Variable Distribution (Transition):")
    st.bar_chart(train_data['Transition'].value_counts(normalize=True))

    st.header("2. Missing Values")
    st.markdown("""
        This section checks for missing values in the dataset and visualizes their distribution.
        Missing values can significantly impact the analysis and model performance, so it's important to identify and handle them appropriately.
    """)
    missing_values = train_data.isnull().sum()
    st.write("Missing Values per Column:")
    st.write(missing_values[missing_values > 0])
    st.write("Total Missing Values:", missing_values.sum())
    if missing_values.sum() > 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(train_data.isnull(), cbar=False, cmap='viridis')
        ax.set_title('Missing Values Heatmap')
        st.pyplot(fig)
        
    st.header("3. Distribution of Target Classes Through Age")
    st.markdown("""
        This section visualizes the distribution of the target classes (`Transition`) across different age groups.
        Understanding how the target variable is distributed through age can provide insights into potential age-related patterns.
    """)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(data=train_data, x='Age', hue='Transition', multiple='stack', palette='viridis', ax=ax)
    ax.set_title('Distribution of Target Classes Through Age')
    ax.set_xlabel('Age')
    ax.set_ylabel('Count')
    st.pyplot(fig)

    st.header("4. Summary Statistics")
    st.markdown("""
        This section provides summary statistics for numerical features in the dataset.
        Summary statistics give a quick overview of the central tendency, dispersion, and shape of the dataset's distribution.
    """)
    st.write(train_data.describe())

    numerical_features = train_data.select_dtypes(include=['float64', 'int64']).columns.tolist()
    numerical_features = [col for col in numerical_features if col != 'Transition']
    X = train_data[numerical_features]

    le = LabelEncoder()
    y = le.fit_transform(train_data['Transition'])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    st.header("5. Skewness and Kurtosis")
    st.markdown("""
        This section provides skewness and kurtosis values for numerical features in the dataset.
        Skewness measures the asymmetry of the data distribution, while kurtosis measures the tailedness.
    """)
    skewness = train_data[numerical_features].skew()
    kurtosis = train_data[numerical_features].kurtosis()
    st.write("Skewness of Numerical Features:")
    st.write(skewness)
    st.write("Kurtosis of Numerical Features:")
    st.write(kurtosis)


    st.header("6. Principal Component Analysis (PCA)")
    st.markdown("""
        PCA is used for dimensionality reduction, helping to identify patterns by projecting the data onto
        components that explain the most variance. This plot shows how much variance is explained
        by the principal components cumulatively. Understanding the variance explained by each component
        helps in determining the number of components to retain for further analysis.
    """)
    pca = PCA(n_components=0.95)
    X_pca = pca.fit_transform(X_scaled)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(np.cumsum(pca.explained_variance_ratio_))
    ax.set_xlabel('Number of Components')
    ax.set_ylabel('Cumulative Explained Variance')
    ax.set_title('PCA - Cumulative Variance Explained')
    st.pyplot(fig)

    st.header("7. Feature Importance with Random Forest")
    st.markdown("""
        Feature importance helps identify the most influential features for classification. 
        Here, the top 20 most important features are shown based on a Random Forest model.
        Understanding which features are most important can guide feature selection and improve model performance.
    """)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_scaled, y)

    feature_importances = pd.Series(rf.feature_importances_, index=numerical_features)
    top_features = feature_importances.nlargest(20)
    fig, ax = plt.subplots(figsize=(12, 6))
    top_features.plot(kind='bar', ax=ax)
    ax.set_title('Top 20 Most Important Features')
    ax.set_xlabel('Features')
    ax.set_ylabel('Importance')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig)

    st.header("8. High-Dimensional Correlation Analysis")
    st.markdown("""
        This heatmap shows the Spearman correlation between the top 20 features, which can help identify
        relationships or redundancies in the dataset. High correlation between features may indicate redundancy,
        which can be addressed by feature selection or dimensionality reduction techniques.
    """)
    correlation_matrix = np.zeros((len(top_features), len(top_features)))
    for i, feat1 in enumerate(top_features.index):
        for j, feat2 in enumerate(top_features.index):
            correlation_matrix[i, j], _ = spearmanr(X[feat1], X[feat2])
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(correlation_matrix, 
                xticklabels=top_features.index, 
                yticklabels=top_features.index, 
                cmap='coolwarm', 
                center=0, 
                annot=True, 
                fmt='.2f', 
                linewidths=0.5, 
                ax=ax)
    ax.set_title('Correlation Heatmap of Top 20 Features')
    st.pyplot(fig)

    st.header("9. t-SNE Visualization")
    st.markdown("""
        t-SNE is a non-linear dimensionality reduction technique that helps visualize high-dimensional
        data in a two-dimensional space. It is particularly useful for visualizing clusters or separability between classes.
        This plot helps in understanding the inherent structure and clustering of the data.
    """)
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X_tsne = tsne.fit_transform(X_scaled)
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis', alpha=0.7)
    fig.colorbar(scatter, ax=ax, label='Transition Class')
    ax.set_title('t-SNE Visualization')
    ax.set_xlabel('t-SNE Component 1')
    ax.set_ylabel('t-SNE Component 2')
    st.pyplot(fig)

    st.header("10. Feature Relevance Using Mutual Information")
    st.markdown("""
        Mutual Information (MI) quantifies the dependency between features and the target variable.
        Higher scores indicate stronger relevance of a feature for classification. This plot shows the top 20 features
        by MI score, helping to identify which features are most informative for the target variable.
    """)
    mi_scores = mutual_info_classif(X_scaled, y)
    mi_scores_series = pd.Series(mi_scores, index=numerical_features).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 6))
    mi_scores_series.head(20).plot(kind='bar', color='teal', ax=ax)
    ax.set_title('Top 20 Features by Mutual Information')
    ax.set_ylabel('MI Score')
    ax.set_xlabel('Feature')
    st.pyplot(fig)

    st.header("11. Correlation Matrix of Top 20 Features by Mutual Information")
    st.markdown("""
        This heatmap shows the correlation matrix of the top 20 features selected by Mutual Information.
        Correlation analysis helps in understanding the relationships between features and identifying potential redundancies.
    """)
    top_mi_features = mi_scores_series.head(20).index
    correlation_matrix = train_data[top_mi_features].corr(method='spearman')
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(correlation_matrix, 
                xticklabels=top_mi_features, 
                yticklabels=top_mi_features, 
                cmap='coolwarm', 
                center=0, 
                annot=True, 
                fmt='.2f', 
                linewidths=0.5, 
                ax=ax)
    ax.set_title('Correlation Matrix of Top 20 Features by Mutual Information')
    st.pyplot(fig)

    st.header("12. Top Features Distribution")
    st.markdown("""
        Box plots show the distribution of the top features, highlighting the spread and outliers.
        This helps in understanding the range and variability of the features, as well as identifying potential outliers.
    """)
    fig, axes = plt.subplots(4, 5, figsize=(20, 16))
    for i, feature in enumerate(top_features.index[:20]):
        sns.boxplot(x='Transition', y=feature, data=train_data, ax=axes[i//5, i%5], palette='viridis')
        axes[i//5, i%5].set_title(f'Distribution of {feature}')
        axes[i//5, i%5].set_xticklabels(axes[i//5, i%5].get_xticklabels(), rotation=45)
    plt.tight_layout()
    st.pyplot(fig)

    st.header("13. Histogram of Top Features")
    st.markdown("""
        Histograms show the distribution of the top features, providing insights into their spread and central tendency.
        This helps in understanding the overall distribution and identifying any skewness or abnormalities in the data.
    """)
    fig, axes = plt.subplots(4, 5, figsize=(20, 16))
    for i, feature in enumerate(top_features.index[:20]):
        sns.histplot(train_data[feature], kde=True, ax=axes[i//5, i%5], color='teal')
        axes[i//5, i%5].set_title(f'Histogram of {feature}')
    plt.tight_layout()
    st.pyplot(fig)

    st.header("14. Class Distribution by Feature")
    st.markdown("""
        Violin plots show the distribution of each class for the top features, highlighting the overlap and separability.
        This helps in understanding how well the features can distinguish between different classes.
    """)
    fig, axes = plt.subplots(4, 5, figsize=(20, 16))
    for i, feature in enumerate(top_features.index[:20]):
        sns.violinplot(x='Transition', y=feature, data=train_data, ax=axes[i//5, i%5], palette='viridis')
        axes[i//5, i%5].set_title(f'Class Distribution of {feature}')
        axes[i//5, i%5].set_xticklabels(axes[i//5, i%5].get_xticklabels(), rotation=45)
    plt.tight_layout()
    st.pyplot(fig)

    st.header("15. Pair Plot of Top Features")
    st.markdown("""
        Pair plots show the relationships between the top features, helping to identify patterns and correlations.
        This is useful for visualizing interactions between features and understanding their joint distributions.
        Pair plots can reveal potential linear or non-linear relationships between features, which can be important for model building.
    """)
    pairplot_data = train_data[top_features.index[:5].tolist() + ['Transition']]
    fig = sns.pairplot(pairplot_data, hue='Transition', palette='viridis')
    st.pyplot(fig.fig)

    st.header("16. K-means Clustering")
    st.markdown("""
        K-means clustering is used to identify clusters in the data. This plot shows the clusters identified by K-means.
        Clustering helps in understanding the inherent grouping in the data and can be useful for identifying patterns or subgroups.
        In this plot, we use t-SNE components to visualize the clusters identified by K-means. Each point represents a sample,
        and the color indicates the cluster assignment. This helps in understanding the structure and separability of the clusters.
    """)
    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c=clusters, cmap='viridis', alpha=0.7)
    fig.colorbar(scatter, ax=ax, label='Cluster')
    ax.set_title('K-means Clustering')
    ax.set_xlabel('t-SNE Component 1')
    ax.set_ylabel('t-SNE Component 2')
    st.pyplot(fig)
    
    st.header("17. Feature Importance for Each Transition Class")
    st.markdown("""
        This section shows the feature importance for each class of the target variable `Transition`.
        Understanding the most important features for each class can help in identifying class-specific patterns.
    """)

    numerical_features = train_data.select_dtypes(include=['float64', 'int64']).columns.tolist()
    numerical_features = [col for col in numerical_features if col != 'Transition']
    X = train_data[numerical_features]

    le = LabelEncoder()
    y = le.fit_transform(train_data['Transition'])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    unique_classes = train_data['Transition'].unique()
    for cls in unique_classes:
        st.subheader(f"Feature Importance for Transition Class: {cls}")
        cls_mask = train_data['Transition'] == cls
        y_cls = cls_mask.astype(int)

        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_scaled, y_cls)

        feature_importances = pd.Series(rf.feature_importances_, index=numerical_features)
        top_features = feature_importances.nlargest(20)
        fig, ax = plt.subplots(figsize=(12, 6))
        top_features.plot(kind='bar', ax=ax)
        ax.set_title(f'Top 20 Most Important Features for Transition Class: {cls}')
        ax.set_xlabel('Features')
        ax.set_ylabel('Importance')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig)

train_data = pd.read_csv('../datasets/train_radiomics_hipocamp.csv')
advanced_radiomics_visualization(train_data)