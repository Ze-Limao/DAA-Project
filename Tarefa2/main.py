import os
import pandas as pd
import numpy as np
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    BaggingClassifier, StackingClassifier
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, Normalizer
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import RFECV
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
import xgboost as xgb
from colorama import Fore, init
import matplotlib.pyplot as plt
import seaborn as sns

init(autoreset=True)

def winsorize_outliers(data, lower_percentile=0.01, upper_percentile=0.99):
    """Apply Winsorization to reduce the impact of outliers."""
    for col in data.select_dtypes(include=['float64', 'int64']).columns:
        lower_bound = data[col].quantile(lower_percentile)
        upper_bound = data[col].quantile(upper_percentile)
        data[col] = np.where(data[col] < lower_bound, lower_bound, data[col])
        data[col] = np.where(data[col] > upper_bound, upper_bound, data[col])
    return data

def encode_transition(data):
    label_encoder = LabelEncoder()
    data['Transition'] = label_encoder.fit_transform(data['Transition'])
    return data, label_encoder

def decode_transition(predictions, label_encoder):
    return label_encoder.inverse_transform(predictions)

print(Fore.BLUE + "⏳ Loading and Preprocessing Dataset...")
train_data = pd.read_csv('../datasets/train_radiomics_hipocamp.csv')
test_data = pd.read_csv('../datasets/test_radiomics_hipocamp.csv')
train_data.dropna(inplace=True)
print(Fore.GREEN + "✅ Train dataset loaded and cleaned.")

train_data = winsorize_outliers(train_data)
print(Fore.GREEN + "✅ Winsorization applied to outliers in the training data.")

constant_columns = [col for col in train_data.columns if train_data[col].nunique() == 1]
train_data.drop(columns=constant_columns, inplace=True)
test_data.drop(columns=constant_columns, inplace=True)
print(Fore.RED + f"✅ Removed constant columns: {constant_columns}")

train_data, transition_encoder = encode_transition(train_data)
print(Fore.GREEN + "✅ 'Transition' column encoded.")
print(Fore.BLUE + f"ℹ️ 'Transition' column info: unique values = {train_data['Transition'].unique()}, data type = {train_data['Transition'].dtype}")

numerical_features = train_data.select_dtypes(include=['float64', 'int64']).columns.tolist()
numerical_features.remove('Transition')
categorical_features = train_data.select_dtypes(include=['object', 'category']).columns.tolist()

train_data.drop(columns=categorical_features, inplace=True)
test_data.drop(columns=categorical_features, inplace=True)
print(Fore.RED + f"✅ Removed categorical columns: {categorical_features}")

preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('scaler', StandardScaler()),
            ('normalizer', Normalizer())
        ]), numerical_features),
    ]
)

label_encoder = LabelEncoder()
X_train_full = train_data.drop(columns=['Transition'])
y_train_full = train_data['Transition']

X_train_transformed = preprocessor.fit_transform(X_train_full)
X_test_full = test_data
X_test_transformed = preprocessor.transform(X_test_full)

print(Fore.BLUE + "\n🌟 Applying RFECV for feature selection...")
rf = RandomForestClassifier(random_state=42)
rfecv = RFECV(estimator=rf, step=1, cv=3, scoring='f1_macro')
X_train_rfe = rfecv.fit_transform(X_train_transformed, y_train_full)
X_test_rfe = rfecv.transform(X_test_transformed)
print(Fore.GREEN + f"✅ RFECV applied. Optimal number of features: {rfecv.n_features_}")

print(Fore.BLUE + "\n🌟 Applying PCA for dimensionality reduction...")
pca = PCA(n_components=0.95)
X_train_pca = pca.fit_transform(X_train_rfe)
X_test_pca = pca.transform(X_test_rfe)
print(Fore.GREEN + f"✅ PCA applied. Number of components chosen: {pca.n_components_}")

X_train, X_val, y_train, y_val = train_test_split(X_train_pca, y_train_full, test_size=0.2, random_state=42)
print(Fore.GREEN + "✅ Data split into training and validation sets.")

models = {
    "RandomForest": Pipeline([
        ('classifier', RandomForestClassifier(random_state=42))
    ]),
    "GradientBoosting": Pipeline([
        ('classifier', GradientBoostingClassifier(random_state=42))
    ]),
    "Bagging": Pipeline([
        ('classifier', BaggingClassifier(random_state=42))
    ]),
    "DecisionTree": Pipeline([
        ('classifier', DecisionTreeClassifier(random_state=42))
    ]),
    "SVM": Pipeline([
        ('classifier', SVC(probability=True))
    ]),
    "XGBoost": Pipeline([
        ('classifier', xgb.XGBClassifier(eval_metric='logloss', random_state=42))
    ])
}

param_grids = {
    "RandomForest": {'classifier__n_estimators': [100, 200], 'classifier__max_depth': [None, 10, 20], 'classifier__min_samples_split': [2, 5, 10],'classifier__min_samples_leaf': [1, 2, 4]},
    "GradientBoosting": {'classifier__n_estimators': [100, 200, 300], 'classifier__learning_rate': [0.01, 0.05, 0.1], 'classifier__max_depth': [3, 4, 5], 'classifier__subsample': [0.7, 0.8, 1.0]},
    "Bagging": {'classifier__n_estimators': [50, 100, 150]},
    "DecisionTree": {'classifier__max_depth': [None, 10, 20]},
    "SVM": {'classifier__C': [0.5, 1, 5, 10]},
    "XGBoost": {
        'classifier__n_estimators': [100, 200, 300],
        'classifier__learning_rate': [0.05, 0.1, 0.2],
        'classifier__max_depth': [1, 2, 3, 4],
    }
}

"""
"XGBoost": {
        'classifier__n_estimators': [100, 200, 500],
        'classifier__learning_rate': [0.01, 0.05, 0.1, 0.2],
        'classifier__max_depth': [3, 5, 7, 9, 12],
        'classifier__gamma': [0, 0.1, 0.5, 1],
        'classifier__min_child_weight': [1, 5, 10],
        'classifier__subsample': [0.5, 0.7, 1.0],
        'classifier__booster': ['gbtree', 'dart']
    }
"""

results = {}
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

for model_name, model in models.items():
    print(Fore.BLUE + f"🔍 Tuning {model_name}...")
    clf = GridSearchCV(model, param_grids[model_name], cv=cv, scoring='f1_macro', n_jobs=-1)
    clf.fit(X_train, y_train)
    best_model = clf.best_estimator_
    y_pred = best_model.predict(X_val)
    results[model_name] = {
        'accuracy': accuracy_score(y_val, y_pred),
        'f1': f1_score(y_val, y_pred, average='macro')
    }
    print(Fore.GREEN + f"✅ {model_name} - Best Params: {clf.best_params_} - Accuracy: {results[model_name]['accuracy']:.2%}, F1 Score: {results[model_name]['f1']:.2f}")

print(Fore.BLUE + "\n🔀 Training optimized Stacking model...")
stacking_model = StackingClassifier(
    estimators=[
        ('random_forest', models['RandomForest']),
        ('gradient_boosting', models['GradientBoosting']),
        ('bagging', models['Bagging']),
        ('decision_tree', models['DecisionTree']),
        ('svm', models['SVM']),
        ('xgboost', models['XGBoost'])
    ],
    final_estimator=GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
)
stacking_model.fit(X_train, y_train)
y_pred_stacking = stacking_model.predict(X_val)
results['Stacking'] = {
    'accuracy': accuracy_score(y_val, y_pred_stacking),
    'f1': f1_score(y_val, y_pred_stacking, average='macro')
}
print(Fore.GREEN + f"✅ Optimized Stacking - Accuracy: {results['Stacking']['accuracy']:.2%}, F1 Score: {results['Stacking']['f1']:.2f}")

print(Fore.WHITE + "\n📄 Generating submission files...")
submissions_dir = "submissions"
os.makedirs(submissions_dir, exist_ok=True)

for model_name, model in models.items():
    model.fit(X_train_pca, y_train_full)
    y_pred_submission = model.predict(X_test_pca)
    y_pred_stacking_submission_decoded = decode_transition(y_pred_submission, transition_encoder)
    submission = pd.DataFrame({
        "RowId": range(1, len(y_pred_submission) + 1),
        "Result": y_pred_stacking_submission_decoded
    })
    filename = os.path.join(submissions_dir, f"{model_name.lower()}_submission.csv")
    submission.to_csv(filename, index=False)
    print(Fore.GREEN + f"✅ Submission for {model_name} saved: {filename}")

stacking_model.fit(X_train_pca, y_train_full)
y_pred_stacking_submission = stacking_model.predict(X_test_pca)
y_pred_stacking_submission_decoded = decode_transition(y_pred_stacking_submission, transition_encoder)
stacking_submission = pd.DataFrame({
    "RowId": range(1, len(y_pred_stacking_submission) + 1),
    "Result": y_pred_stacking_submission_decoded
})
stacking_submission_file = f"{submissions_dir}/Stacking_submission.csv"
stacking_submission.to_csv(stacking_submission_file, index=False)
print(Fore.GREEN + f"✅ Submission created for Stacking: {stacking_submission_file}")

print(Fore.WHITE + "\n📊 Creating performance comparison graphs...")
graphs_dir = "graphs"
os.makedirs(graphs_dir, exist_ok=True)

model_names = list(results.keys())
accuracies = [results[model]['accuracy'] for model in model_names]
f1_scores = [results[model]['f1'] for model in model_names]

plt.style.use('default')
sns.set_palette("husl")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

sns.barplot(x=accuracies, y=model_names, ax=ax1, palette='viridis')
ax1.set_title('Model Accuracy Comparison', pad=20)
ax1.set_xlabel('Accuracy')
ax1.grid(True, axis='x')
for i, v in enumerate(accuracies):
    ax1.text(v, i, f' {v:.3f}', va='center')

sns.barplot(x=f1_scores, y=model_names, ax=ax2, palette='viridis')
ax2.set_title('Model F1-Score Comparison', pad=20)
ax2.set_xlabel('F1-Score')
ax2.grid(True, axis='x')
for i, v in enumerate(f1_scores):
    ax2.text(v, i, f' {v:.3f}', va='center')

plt.tight_layout()
graph_path = os.path.join(graphs_dir, 'model_performance_comparison.png')
plt.savefig(graph_path, dpi=300, bbox_inches='tight')
plt.close()

plt.figure(figsize=(12, 6))
x = np.arange(len(model_names))
width = 0.35

plt.bar(x - width/2, accuracies, width, label='Accuracy', color='skyblue')
plt.bar(x + width/2, f1_scores, width, label='F1-Score', color='lightgreen')

plt.xlabel('Models')
plt.ylabel('Score')
plt.title('Model Performance Comparison (Accuracy vs F1-Score)')
plt.xticks(x, model_names, rotation=45, ha='right')
plt.legend()
plt.grid(True, axis='y')

for i, (acc, f1) in enumerate(zip(accuracies, f1_scores)):
    plt.text(i - width/2, acc, f'{acc:.3f}', ha='center', va='bottom')
    plt.text(i + width/2, f1, f'{f1:.3f}', ha='center', va='bottom')

plt.tight_layout()
combined_graph_path = os.path.join(graphs_dir, 'model_performance_combined.png')
plt.savefig(combined_graph_path, dpi=300, bbox_inches='tight')
plt.close()

print(Fore.GREEN + f"✅ Performance comparison graphs saved in {graphs_dir}:")
print(Fore.GREEN + f"   - {graph_path}")
print(Fore.GREEN + f"   - {combined_graph_path}")

print(Fore.WHITE + "\n🎉 All tasks completed!")