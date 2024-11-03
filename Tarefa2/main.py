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
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
from sklearn.pipeline import Pipeline
import xgboost as xgb

def winsorize_outliers(data, lower_percentile=0.01, upper_percentile=0.99):
    """Apply Winsorization to reduce the impact of outliers."""
    for col in data.select_dtypes(include=['float64', 'int64']).columns:
        lower_bound = data[col].quantile(lower_percentile)
        upper_bound = data[col].quantile(upper_percentile)
        data[col] = np.where(data[col] < lower_bound, lower_bound, data[col])
        data[col] = np.where(data[col] > upper_bound, upper_bound, data[col])
    return data

print("⏳ Loading and Preprocessing Dataset...")
train_data = pd.read_csv('../datasets/train_radiomics_hipocamp.csv')
test_data = pd.read_csv('../datasets/test_radiomics_hipocamp.csv')
train_data.dropna(inplace=True)
print("✅ Train dataset loaded and cleaned.")

train_data = winsorize_outliers(train_data)
print("✅ Winsorization applied to outliers in the training data.")

constant_columns = [col for col in train_data.columns if train_data[col].nunique() == 1]
train_data.drop(columns=constant_columns, inplace=True)
test_data.drop(columns=constant_columns, inplace=True)
print(f"✅ Removed constant columns: {constant_columns}")

numerical_features = train_data.select_dtypes(include=['float64', 'int64']).columns.tolist()
categorical_features = train_data.select_dtypes(include=['object', 'category']).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        # ('cat', OneHotEncoder(), categorical_features)
    ]
)

label_encoder = LabelEncoder()
X_train_full = train_data.drop(columns=['Transition'])
y_train_full = label_encoder.fit_transform(train_data['Transition'])

X_train_transformed = preprocessor.fit_transform(X_train_full)
X_test_full = test_data
X_test_transformed = preprocessor.transform(X_test_full)

print("\n🌟 Performing Recursive Feature Elimination (RFE) with Random Forest...")
rfe_selector = RFE(RandomForestClassifier(random_state=42), n_features_to_select=500, step=100)
X_train_selected = rfe_selector.fit_transform(X_train_transformed, y_train_full)
X_test_selected = rfe_selector.transform(X_test_transformed)
print(f"✅ RFE complete with {X_train_selected.shape[1]} features retained.")

print("\n🌟 Applying PCA for dimensionality reduction...")
n_components = min(300, X_train_selected.shape[1])
pca = PCA(n_components=n_components)
X_train_pca = pca.fit_transform(X_train_selected)
X_test_pca = pca.transform(X_test_selected)

print("\n⚙️ Splitting data and training models with hyperparameter tuning...")
X_train, X_val, y_train, y_val = train_test_split(X_train_selected, y_train_full, test_size=0.2, random_state=42)

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
    "RandomForest": {'classifier__n_estimators': [100, 200], 'classifier__max_depth': [None, 10, 20]},
    "GradientBoosting": {'classifier__n_estimators': [100, 200], 'classifier__learning_rate': [0.05, 0.1]},
    "Bagging": {'classifier__n_estimators': [10, 50]},
    "DecisionTree": {'classifier__max_depth': [None, 10, 20]},
    "SVM": {'classifier__C': [0.1, 1, 10], 'classifier__kernel': ['linear', 'rbf']},
    "XGBoost": {
        'classifier__n_estimators': [100, 150],
        'classifier__learning_rate': [0.05, 0.1],
        'classifier__max_depth': [3, 4],
        'classifier__gamma': [0, 0.1],
        'classifier__subsample': [0.7, 1.0],
        'classifier__booster': ['gbtree']
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
    print(f"🔍 Tuning {model_name}...")
    clf = GridSearchCV(model, param_grids[model_name], cv=cv, scoring='f1_macro', n_jobs=-1)
    clf.fit(X_train, y_train)
    best_model = clf.best_estimator_
    y_pred = best_model.predict(X_val)
    results[model_name] = {
        'accuracy': accuracy_score(y_val, y_pred),
        'f1': f1_score(y_val, y_pred, average='macro')
    }
    print(f"✅ {model_name} - Best Params: {clf.best_params_} - Accuracy: {results[model_name]['accuracy']:.2%}, F1 Score: {results[model_name]['f1']:.2f}")

print("\n🔀 Training optimized Stacking model...")
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
print(f"✅ Optimized Stacking - Accuracy: {results['Stacking']['accuracy']:.2%}, F1 Score: {results['Stacking']['f1']:.2f}")

print("\n📄 Generating submission files...")
submissions_dir = "submissions"
os.makedirs(submissions_dir, exist_ok=True)

for model_name, model in models.items():
    model.fit(X_train_selected, y_train_full)
    y_pred_submission = model.predict(X_test_selected)
    submission = pd.DataFrame({
        "RowId": range(1, len(y_pred_submission) + 1),
        "Result": label_encoder.inverse_transform(y_pred_submission)
    })
    submission_file = f"{submissions_dir}/{model_name}_submission.csv"
    submission.to_csv(submission_file, index=False)
    print(f"   ✅ Submission created for {model_name}: {submission_file}")

stacking_model.fit(X_train_selected, y_train_full)
y_pred_stacking_submission = stacking_model.predict(X_test_selected)
stacking_submission = pd.DataFrame({
    "RowId": range(1, len(y_pred_stacking_submission) + 1),
    "Result": label_encoder.inverse_transform(y_pred_stacking_submission)
})
stacking_submission.to_csv(f"{submissions_dir}/stacking_submission.csv", index=False)
print("   ✅ Submission created for Stacking: submissions/stacking_submission.csv")

print("\n🎉 All tasks completed!")