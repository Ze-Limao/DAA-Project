import os
import pandas as pd
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, 
    BaggingClassifier, StackingClassifier
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression  # For stacking meta-model
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

print("⏳ Loading and Preprocessing Dataset...")
train_data = pd.read_csv('../datasets/train_radiomics_hipocamp.csv')
test_data = pd.read_csv('../datasets/test_radiomics_hipocamp.csv')
train_data.dropna(inplace=True)
print("✅ Train dataset loaded and cleaned.")

numerical_features = train_data.select_dtypes(include=['float64', 'int64']).columns.tolist()
categorical_features = train_data.select_dtypes(include=['object', 'category']).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        # ('cat', OneHotEncoder(), categorical_features)
    ]
)

print("Columns in X_train_full:", train_data.columns)
X_train_full = train_data.drop(columns=['Transition'])
y_train_full = train_data['Transition']
X_train_transformed = preprocessor.fit_transform(X_train_full)

X_test_full = test_data
X_test_transformed = preprocessor.transform(X_test_full)

print("\n🌟 Performing feature selection with Random Forest...")
feature_selector = RandomForestClassifier(random_state=42)
feature_selector.fit(X_train_transformed, y_train_full)
importances = feature_selector.feature_importances_

top_features = sorted(range(len(importances)), key=lambda i: importances[i], reverse=True)[:1755]
X_train_selected = X_train_transformed[:, top_features]
X_test_selected = X_test_transformed[:, top_features]
print(f"✅ Feature selection complete with {len(top_features)} features retained.")

print("\n⚙️ Splitting data and training models...")
X_train, X_val, y_train, y_val = train_test_split(X_train_selected, y_train_full, test_size=0.2, random_state=42)

models = {
    "RandomForest": RandomForestClassifier(random_state=42),
    "DecisionTree": DecisionTreeClassifier(random_state=42),
    "SVM": SVC(),
    "GradientBoosting": GradientBoostingClassifier(random_state=42),
    "Bagging": BaggingClassifier(random_state=42)
}

results = {}
for model_name, model in models.items():
    clf = Pipeline(steps=[('classifier', model)])
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_val)
    results[model_name] = {
        'accuracy': accuracy_score(y_val, y_pred),
        'f1': f1_score(y_val, y_pred, average='macro')
    }
    print(f"✅ {model_name} - Accuracy: {results[model_name]['accuracy']:.2%}, F1 Score: {results[model_name]['f1']:.2f}")

print("\n🔀 Training Stacking model...")
stacking_model = StackingClassifier(
    estimators=[
        ('random_forest', models['RandomForest']),
        ('decision_tree', models['DecisionTree']),
        ('gradient_boosting', models['GradientBoosting'])
    ],
    final_estimator=LogisticRegression()
)
stacking_model.fit(X_train, y_train)
y_pred_stacking = stacking_model.predict(X_val)
results['Stacking'] = {
    'accuracy': accuracy_score(y_val, y_pred_stacking),
    'f1': f1_score(y_val, y_pred_stacking, average='macro')
}
print(f"✅ Stacking - Accuracy: {results['Stacking']['accuracy']:.2%}, F1 Score: {results['Stacking']['f1']:.2f}")

print("\n📄 Generating submission files...")
submissions_dir = "submissions"
os.makedirs(submissions_dir, exist_ok=True)

for model_name, model in models.items():
    model.fit(X_train_selected, y_train_full)
    y_pred_submission = model.predict(X_test_selected)
    submission = pd.DataFrame({
        "RowId": range(1, len(y_pred_submission) + 1),
        "Result": y_pred_submission
    })
    submission_file = f"{submissions_dir}/{model_name}_submission.csv"
    submission.to_csv(submission_file, index=False)
    print(f"   ✅ Submission created for {model_name}: {submission_file}")

stacking_model.fit(X_train_selected, y_train_full)
y_pred_stacking_submission = stacking_model.predict(X_test_selected)
stacking_submission = pd.DataFrame({
    "RowId": range(1, len(y_pred_stacking_submission) + 1),
    "Result": y_pred_stacking_submission
})
stacking_submission.to_csv(f"{submissions_dir}/stacking_submission.csv", index=False)
print("   ✅ Submission created for Stacking: submissions/stacking_submission.csv")

print("\n🎉 All tasks completed!")
