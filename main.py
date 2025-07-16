import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from tpot import TPOTClassifier, TPOTRegressor
#TPOT (Tree-based Pipeline Optimization Tool) is an open-source Python library for Automated Machine Learning (AutoML).
# It leverages genetic programming to optimize machine learning pipelines
import joblib #running Python functions as pipeline jobs
import os
from modules.preprocessor import preprocess_data
from modules.model_trainer import train_model
from modules.evaluator import evaluate_model

def load_file(file_path):
    df= pd.read_csv(file_path)
    print(f"File loaded successfully with {df.shape[0]} rows and {df.shape[1]} columns")
    return df

def save_model(model):
    os.makedirs("models", exist_ok=True)
    joblib.dump(model.fitted_pipeline_, "models/Bostonbest_model.pkl")
    joblib.dump(X.columns.tolist(), "models/columns.pkl")
    print("\n💾 Modelsaved to 'models/' successfully.")
    
if __name__ == "__main__":
    #user input
    file_path = input("📂 Enter the path of csv dataset: ")
    task = input("🗒️ Enter the task (classification or regression):").strip().lower()
    target_column = input("Enter the target column name: ").strip()
    
    #Pipeline steps
    df= load_file(file_path)
    X, y = preprocess_data(df, target_column)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model= train_model(X_train, y_train, task)
    evaluate_model(model, X_test, y_test, task)
    save_model(model)