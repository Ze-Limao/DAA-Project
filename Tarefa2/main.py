import tkinter as tk
from tkinter import messagebox, filedialog
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
from imblearn.over_sampling import SMOTE
from collections import Counter
import torch
import neural_network
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from ttkbootstrap.icons import Icon
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import json
from pathlib import Path

class ModelTrainingGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Model Training Interface")
        self.root.geometry("1200x900")
        
        self.X_train_pca = None
        self.X_test_pca = None
        self.y_train_full = None
        self.transition_encoder = None
        self.results = {}

        self.create_widgets()

    def create_widgets(self):
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)

        title_label = ttk.Label(main_frame, text="Model Training Interface", 
                                font=("Helvetica", 28, "bold"), bootstyle="info")
        title_label.pack(pady=(0, 20))

        model_frame = ttk.Labelframe(main_frame, text="Available Models", padding="10")
        model_frame.pack(fill=tk.X, pady=(0, 20))
        
        self.models = {
            "RandomForest": {
                'model': Pipeline([('classifier', RandomForestClassifier(random_state=42))]),
                'params': {
                    'classifier__n_estimators': [100, 200],
                    'classifier__max_depth': [None, 10, 20],
                    'classifier__min_samples_split': [2, 5, 10],
                    'classifier__min_samples_leaf': [1, 2, 4]
                }
            },
            "GradientBoosting": {
                'model': Pipeline([('classifier', GradientBoostingClassifier(random_state=42))]),
                'params': {
                    'classifier__n_estimators': [100, 200, 300],
                    'classifier__learning_rate': [0.01, 0.05, 0.1],
                    'classifier__max_depth': [3, 4, 5],
                    'classifier__subsample': [0.7, 0.8, 1.0]
                }
            },
            "Bagging": {
                'model': Pipeline([('classifier', BaggingClassifier(random_state=42))]),
                'params': {'classifier__n_estimators': [50, 100, 150]}
            },
            "DecisionTree": {
                'model': Pipeline([('classifier', DecisionTreeClassifier(random_state=42))]),
                'params': {'classifier__max_depth': [None, 10, 20]}
            },
            "SVM": {
                'model': Pipeline([('classifier', SVC(probability=True))]),
                'params': {'classifier__C': [0.5, 1, 5, 10]}
            },
            "XGBoost": {
                'model': Pipeline([('classifier', xgb.XGBClassifier(eval_metric='logloss', random_state=42))]),
                'params': {
                    'classifier__n_estimators': [100, 200, 300],
                    'classifier__learning_rate': [0.05, 0.1, 0.2],
                    'classifier__max_depth': [1, 2, 3, 4]
                }
            },
            "Neural Network": {'model': None, 'params': None},
            "Stacking": {'model': None, 'params': None}
        }
        
        self.model_vars = {}
        for i, model_name in enumerate(self.models.keys()):
            var = tk.BooleanVar(value=False)
            self.model_vars[model_name] = var
            cb = ttk.Checkbutton(model_frame, text=model_name, variable=var, bootstyle="round-toggle")
            cb.grid(row=i//4, column=i%4, sticky=tk.W, padx=10, pady=5)
        
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(pady=10)
        
        select_all_btn = ttk.Button(button_frame, text="Select All", 
                                    command=self.select_all, bootstyle="info-outline")
        select_all_btn.grid(row=0, column=0, padx=5)
        
        deselect_all_btn = ttk.Button(button_frame, text="Deselect All", 
                                      command=self.deselect_all, bootstyle="info-outline")
        deselect_all_btn.grid(row=0, column=1, padx=5)
        
        options_frame = ttk.Labelframe(main_frame, text="Training Options", padding="10")
        options_frame.pack(fill=tk.X, pady=(0, 20))
        
        self.save_preprocessed_var = tk.BooleanVar(value=False)
        save_preprocessed_cb = ttk.Checkbutton(options_frame, text="Save preprocessed data", 
                                               variable=self.save_preprocessed_var, 
                                               bootstyle="round-toggle")
        save_preprocessed_cb.grid(row=0, column=0, padx=10, pady=5, sticky=tk.W)
        
        self.use_preprocessed_var = tk.BooleanVar(value=False)
        use_preprocessed_cb = ttk.Checkbutton(options_frame, text="Use preprocessed data", 
                                              variable=self.use_preprocessed_var, 
                                              command=self.select_preprocessed_file, 
                                              bootstyle="round-toggle")
        use_preprocessed_cb.grid(row=0, column=1, padx=10, pady=5, sticky=tk.W)

        self.use_control_data = tk.BooleanVar(value=False)
        use_control_data_cb = ttk.Checkbutton(options_frame, text="Use control data", 
                                              variable=self.use_control_data, 
                                              bootstyle="round-toggle")
        use_control_data_cb.grid(row=0, column=2, padx=10, pady=5, sticky=tk.W)
        
        train_button = ttk.Button(main_frame, text="Train Selected Models", 
                                  command=self.train_models, bootstyle="success")
        train_button.pack(pady=20)
        
        self.progress_frame = ttk.Labelframe(main_frame, text="Training Progress", padding="10")
        self.progress_frame.pack(fill=tk.X, pady=(0, 20))
        
        self.progress_var = tk.StringVar(value="Ready to train...")
        self.progress_label = ttk.Label(self.progress_frame, textvariable=self.progress_var, 
                                        wraplength=900)
        self.progress_label.pack(fill=tk.X)
        
        self.progress_bar = ttk.Progressbar(self.progress_frame, length=900, mode='determinate', 
                                            bootstyle="success-striped")
        self.progress_bar.pack(pady=10)
        
        self.preprocessed_file_path = None

    def update_progress(self, message, progress_value):
        self.progress_var.set(message)
        self.progress_bar['value'] = progress_value
        self.root.update()

    def select_all(self):
        for var in self.model_vars.values():
            var.set(True)

    def deselect_all(self):
        for var in self.model_vars.values():
            var.set(False)

    def select_preprocessed_file(self):
        if self.use_preprocessed_var.get():
            self.preprocessed_file_path = filedialog.askopenfilename(
                title="Select Preprocessed Data File",
                filetypes=(("CSV files", "*.csv"), ("All files", "*.*"))
            )
        else:
            self.preprocessed_file_path = None

    def preprocess_data(self):
        self.update_progress("Loading and preprocessing dataset...", 5)
        
        train_data = None
        if self.use_control_data.get():
            train_data = pd.read_csv('../datasets/train_radiomics_occipital_CONTROL.csv')
        else:
            train_data = pd.read_csv('../datasets/train_radiomics_hipocamp.csv')
        
        test_data = pd.read_csv('../datasets/test_radiomics_hipocamp.csv')
        train_data.dropna(inplace=True)
        
        def winsorize_outliers(data, lower_percentile=0.01, upper_percentile=0.99):
            for col in data.select_dtypes(include=['float64', 'int64']).columns:
                lower_bound = data[col].quantile(lower_percentile)
                upper_bound = data[col].quantile(upper_percentile)
                data[col] = np.where(data[col] < lower_bound, lower_bound, data[col])
                data[col] = np.where(data[col] > upper_bound, upper_bound, data[col])
            return data
        
        # train_data = winsorize_outliers(train_data)
        
        constant_columns = [col for col in train_data.columns if train_data[col].nunique() == 1]
        train_data.drop(columns=constant_columns, inplace=True)
        test_data.drop(columns=constant_columns, inplace=True)
        
        self.transition_encoder = LabelEncoder()
        train_data['Transition'] = self.transition_encoder.fit_transform(train_data['Transition'])
        
        numerical_features = train_data.select_dtypes(include=['float64', 'int64']).columns.tolist()
        numerical_features.remove('Transition')
        categorical_features = train_data.select_dtypes(include=['object', 'category']).columns.tolist()
        
        train_data.drop(columns=categorical_features, inplace=True)
        test_data.drop(columns=categorical_features, inplace=True)
        
        self.update_progress("Applying preprocessing steps...", 15)
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', Pipeline([
                    ('scaler', StandardScaler()),
                    ('normalizer', Normalizer())
                ]), numerical_features),
            ]
        )
        
        X_train_full = train_data.drop(columns=['Transition'])
        self.y_train_full = train_data['Transition']
        
        X_train_transformed = preprocessor.fit_transform(X_train_full)
        X_test_transformed = preprocessor.transform(test_data)
        
        self.update_progress("Applying RFECV for feature selection...", 25)
        rf = RandomForestClassifier(random_state=42)
        rfecv = RFECV(estimator=rf, step=1, cv=3, scoring='f1_macro')
        X_train_rfe = rfecv.fit_transform(X_train_transformed, self.y_train_full)
        X_test_rfe = rfecv.transform(X_test_transformed)
        
        self.update_progress("Applying PCA...", 35)
        pca = PCA(n_components=0.95, random_state=42)
        self.X_train_pca = pca.fit_transform(X_train_rfe)
        self.X_test_pca = pca.transform(X_test_rfe)
        
        if self.save_preprocessed_var.get():
            preprocessed_data = {
                'X_train_pca': self.X_train_pca,
                'X_test_pca': self.X_test_pca,
                'y_train_full': self.y_train_full,
                'transition_encoder': self.transition_encoder
            }
            preprocessed_file_path = filedialog.asksaveasfilename(
                title="Save Preprocessed Data",
                defaultextension=".csv",
                filetypes=(("CSV files", "*.csv"), ("All files", "*.*"))
            )
            if preprocessed_file_path:
                pd.to_pickle(preprocessed_data, preprocessed_file_path)
        
        return True

    def load_preprocessed_data(self):
        if self.preprocessed_file_path:
            preprocessed_data = pd.read_pickle(self.preprocessed_file_path)
            self.X_train_pca = preprocessed_data['X_train_pca']
            self.X_test_pca = preprocessed_data['X_test_pca']
            self.y_train_full = preprocessed_data['y_train_full']
            self.transition_encoder = preprocessed_data['transition_encoder']
            return True
        return False

    def train_models(self):
        selected_models = [name for name, var in self.model_vars.items() if var.get()]
        
        if not selected_models:
            messagebox.showwarning("No Selection", "Please select at least one model to train.")
            return
        
        submissions_dir = "submissions"
        graphs_dir = "graphs"
        os.makedirs(submissions_dir, exist_ok=True)
        os.makedirs(graphs_dir, exist_ok=True)
        
        try:
            if self.use_preprocessed_var.get():
                if not self.load_preprocessed_data():
                    messagebox.showerror("Error", "Failed to load preprocessed data.")
                    return
            else:
                if self.X_train_pca is None:
                    if not self.preprocess_data():
                        return
            
            X_train, X_val, y_train, y_val = train_test_split(
                self.X_train_pca, self.y_train_full, test_size=0.2, random_state=42
            )
            
            self.update_progress("Applying SMOTE to balance classes...", 45)
            smote = SMOTE(random_state=42)
            X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
            
            total_steps = len(selected_models)
            current_step = 0
            
            for model_name in selected_models:
                current_step += 1
                progress = 45 + (current_step / total_steps) * 45
                
                if model_name == "Neural Network":
                    self.update_progress(f"Training Neural Network...", progress)
                    neural_network_model = neural_network.main(X_train_smote, y_train_smote)
                    
                    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
                    y_val_tensor = torch.tensor(y_val.to_numpy(), dtype=torch.long)
                    
                    neural_network_model.eval()
                    with torch.no_grad():
                        outputs = neural_network_model(X_val_tensor)
                        _, y_pred_val = torch.max(outputs, 1)
                    
                    y_pred_val_np = y_pred_val.numpy()
                    y_val_np = y_val_tensor.numpy()
                    
                    self.results[model_name] = {
                        'accuracy': accuracy_score(y_val_np, y_pred_val_np),
                        'f1': f1_score(y_val_np, y_pred_val_np, average='macro')
                    }
                    
                    neural_network_model.eval()
                    with torch.no_grad():
                        X_test_tensor = torch.tensor(self.X_test_pca, dtype=torch.float32)
                        outputs = neural_network_model(X_test_tensor)
                        _, y_pred_submission = torch.max(outputs, 1)
                    
                    y_pred_submission_np = y_pred_submission.numpy()
                    y_pred_submission_decoded = self.transition_encoder.inverse_transform(y_pred_submission_np)
                    
                    submission = pd.DataFrame({
                        "RowId": range(1, len(y_pred_submission_np) + 1),
                        "Result": y_pred_submission_decoded
                    })
                    submission.to_csv(f"{submissions_dir}/neural_network_submission.csv", index=False)
                
                elif model_name == "Stacking":
                    if len(selected_models) < 3:
                        messagebox.showwarning(
                            "Warning",
                            "Stacking requires at least 2 other models. Skipping Stacking."
                        )
                        continue
                    
                    self.update_progress(f"Training Stacking Classifier...", progress)
                    
                    base_models = [
                        (name, self.models[name]['model'])
                        for name in selected_models
                        if name not in ["Neural Network", "Stacking"]
                    ]
                    
                    if not base_models:
                        continue
                    
                    stacking_model = StackingClassifier(
                        estimators=base_models,
                        final_estimator=GradientBoostingClassifier(
                            n_estimators=100,
                            learning_rate=0.1,
                            random_state=42
                        )
                    )
                    
                    stacking_model.fit(X_train_smote, y_train_smote)
                    y_pred_val = stacking_model.predict(X_val)
                    
                    self.results[model_name] = {
                        'accuracy': accuracy_score(y_val, y_pred_val),
                        'f1': f1_score(y_val, y_pred_val, average='macro')
                    }
                    
                    stacking_model.fit(self.X_train_pca, self.y_train_full)
                    y_pred_stacking = stacking_model.predict(self.X_test_pca)
                    y_pred_stacking_decoded = self.transition_encoder.inverse_transform(y_pred_stacking)
                    
                    submission = pd.DataFrame({
                        "RowId": range(1, len(y_pred_stacking) + 1),
                        "Result": y_pred_stacking_decoded
                    })
                    submission.to_csv(f"{submissions_dir}/stacking_submission.csv", index=False)
                
                else:
                    self.update_progress(f"Training {model_name}...", progress)
                    
                    model = self.models[model_name]['model']
                    params = self.models[model_name]['params']
                    
                    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                    clf = GridSearchCV(model, params, cv=cv, scoring='f1_macro', n_jobs=-1)
                    clf.fit(X_train_smote, y_train_smote)
                    
                    best_model = clf.best_estimator_
                    y_pred = best_model.predict(X_val)
                    
                    self.results[model_name] = {
                        'accuracy': accuracy_score(y_val, y_pred),
                        'f1': f1_score(y_val, y_pred, average='macro')
                    }
                    
                    model_params = {model_name: clf.best_params_ if model_name not in ["Neural Network", "Stacking"] else "Default parameters used"}
                    if os.path.exists(f"{submissions_dir}/model_parameters.json"):
                        with open(f"{submissions_dir}/model_parameters.json", "r") as json_file:
                            existing_params = json.load(json_file)
                        existing_params.update(model_params)
                    else:
                        existing_params = model_params
                    
                    with open(f"{submissions_dir}/model_parameters.json", "w") as json_file:
                        json.dump(existing_params, json_file, indent=4)
                    
                    best_model.fit(self.X_train_pca, self.y_train_full)
                    y_pred_submission = best_model.predict(self.X_test_pca)
                    y_pred_submission_decoded = self.transition_encoder.inverse_transform(y_pred_submission)
                    submission = pd.DataFrame({
                        "RowId": range(1, len(y_pred_submission) + 1),
                        "Result": y_pred_submission_decoded
                    })
                    submission.to_csv(f"{submissions_dir}/{model_name.lower()}_submission.csv", index=False)
            
            self.update_progress("Generating performance comparison graphs...", 95)
            
            model_names = list(self.results.keys())
            accuracies = [self.results[model]['accuracy'] for model in model_names]
            f1_scores = [self.results[model]['f1'] for model in model_names]
            
            plt.style.use('default')
            sns.set_palette("husl")
            fig, ax = plt.subplots(figsize=(12, 6))
            
            x = np.arange(len(model_names))
            width = 0.35
            
            bars1 = ax.bar(x - width/2, accuracies, width, label='Accuracy', color='skyblue')
            bars2 = ax.bar(x + width/2, f1_scores, width, label='F1-Score', color='lightgreen')
            
            ax.set_xlabel('Models')
            ax.set_ylabel('Score')
            ax.set_title('Model Performance Comparison (Accuracy vs F1-Score)')
            ax.set_xticks(x)
            ax.set_xticklabels(model_names, rotation=45, ha='right')
            ax.legend()
            ax.grid(True, axis='y')
            
            for bar in bars1:
                height = bar.get_height()
                ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
            
            for bar in bars2:
                height = bar.get_height()
                ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
            
            plt.tight_layout()
            
            self.update_progress("Training completed! Check the graph above.", 100)
            messagebox.showinfo(
                "Success",
                "Model training and submission files generation completed!\n"
                "Performance graph is displayed in the UI."
            )
            
            for widget in self.progress_frame.winfo_children():
                widget.destroy()
            
            canvas = FigureCanvasTkAgg(fig, master=self.progress_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
            self.progress_var.set("Training failed!")


def main():
    init(autoreset=True)
    root = ttk.Window(themename="darkly")
    root.tk.call('tk', 'scaling', 2.0)
    app = ModelTrainingGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()