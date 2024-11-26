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
from sklearn.feature_selection import mutual_info_classif
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
    st.markdown("""
        This section provides a high-level overview of the dataset, including its shape, 
        the distribution of data types, and the class balance for the `Transition` variable.
    """)
    st.write("Dataset Shape:", train_data.shape)
    st.write("Column Types:", train_data.dtypes.value_counts())
    st.write("Target Variable Distribution (Transition):")
    st.bar_chart(train_data['Transition'].value_counts(normalize=True))

    numerical_features = train_data.select_dtypes(include=['float64', 'int64']).columns.tolist()
    numerical_features = [col for col in numerical_features if col != 'Transition']
    X = train_data[numerical_features]

    le = LabelEncoder()
    y = le.fit_transform(train_data['Transition'])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    st.header("2. Principal Component Analysis (PCA)")
    st.markdown("""
        PCA is used for dimensionality reduction, helping to identify patterns by projecting the data onto
        components that explain the most variance. This plot shows how much variance is explained
        by the principal components cumulatively.
    """)
    pca = PCA(n_components=0.95)
    X_pca = pca.fit_transform(X_scaled)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(np.cumsum(pca.explained_variance_ratio_))
    ax.set_xlabel('Number of Components')
    ax.set_ylabel('Cumulative Explained Variance')
    ax.set_title('PCA - Cumulative Variance Explained')
    st.pyplot(fig)

    st.header("3. Feature Importance with Random Forest")
    st.markdown("""
        Feature importance helps identify the most influential features for classification. 
        Here, the top 20 most important features are shown based on a Random Forest model.
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

    st.header("4. High-Dimensional Correlation Analysis")
    st.markdown("""
        This heatmap shows the Spearman correlation between the top 20 features, which can help identify
        relationships or redundancies in the dataset.
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

    st.header("5. Feature Overlap Between Classes")
    st.markdown("""
        Violin plots show the distribution of each feature for different classes, highlighting the overlap
        between classes and the separability of features.
    """)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for i, feature in enumerate(top_features.index[:6]):
        sns.violinplot(x='Transition', y=feature, data=train_data, hue='Transition', palette='viridis', legend=False, ax=axes[i//3, i%3])
        axes[i//3, i%3].set_title(f'Overlap of {feature} Across Classes')
        axes[i//3, i%3].set_xticks(axes[i//3, i%3].get_xticks())
        axes[i//3, i%3].set_xticklabels(axes[i//3, i%3].get_xticklabels(), rotation=45)
    plt.tight_layout()
    st.pyplot(fig)

    st.header("6. Pairwise Feature Relationships")
    st.markdown("""
        This scatterplot shows interactions between pairs of top features, helping us understand
        how different features contribute to class separation.
    """)
    feature_pairs = [(top_features.index[0], top_features.index[1]),
                     (top_features.index[2], top_features.index[3])]
    for feat1, feat2 in feature_pairs:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(x=train_data[feat1], y=train_data[feat2], hue=train_data['Transition'], palette='viridis', ax=ax)
        ax.set_title(f'{feat1} vs {feat2}')
        ax.set_xlabel(feat1)
        ax.set_ylabel(feat2)
        st.pyplot(fig)

    st.header("7. t-SNE Visualization")
    st.markdown("""
        t-SNE is a non-linear dimensionality reduction technique that helps visualize high-dimensional
        data in a two-dimensional space. It is particularly useful for visualizing clusters or separability between classes.
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

    st.header("8. Feature Relevance Using Mutual Information")
    st.markdown("""
        Mutual Information (MI) quantifies the dependency between features and the target variable.
        Higher scores indicate stronger relevance of a feature for classification.
    """)
    mi_scores = mutual_info_classif(X_scaled, y)
    mi_scores_series = pd.Series(mi_scores, index=numerical_features).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 6))
    mi_scores_series.head(20).plot(kind='bar', color='teal', ax=ax)
    ax.set_title('Top 20 Features by Mutual Information')
    ax.set_ylabel('MI Score')
    ax.set_xlabel('Feature')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig)

    st.header("9. SHAP Summary Plot")
    st.markdown("""
        SHAP values explain the impact of each feature on the model's predictions.
        This bar plot shows the average absolute SHAP value for each feature.
    """)
    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X_scaled)
    fig, ax = plt.subplots()
    shap.summary_plot(shap_values, pd.DataFrame(X_scaled, columns=numerical_features), plot_type="bar", show=False)
    st.pyplot(fig)

    st.header("10. Model Performance Evaluation")
    st.markdown("""
        The confusion matrix and classification report provide insights into the model's performance,
        including precision, recall, F1-score, and accuracy for each class.
    """)
    y_pred = rf.predict(X_scaled)
    cm = confusion_matrix(y, y_pred)
    cr = classification_report(y, y_pred, target_names=le.classes_)

    st.subheader("Confusion Matrix")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')
    st.pyplot(fig)

    st.subheader("Classification Report")
    st.text(cr)

train_data = pd.read_csv('../datasets/train_radiomics_hipocamp.csv')
advanced_radiomics_visualization(train_data)