import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from tpot import TPOTClassifier, TPOTRegressor
#TPOT (Tree-based Pipeline Optimization Tool) is an open-source Python library for Automated Machine Learning (AutoML).
# It leverages genetic programming to optimize machine learning pipelines
import joblib #running Python functions as pipeline jobs
import os 

def load_file(file_path):
    df= pd.read_csv(file_path)
    print(f"File loaded successfully with {df.shape[0]} rows and {df.shape[1]} columns")
    return df

def preprocess_data(df, target_column):
    df = df.dropna() # Drops missing value rows
    X= df.drop(columns=[target_column]) 
    y=df[target_column]
    
    #Encoding categorical variables(converting strings to numbers eg. TRUE-1, False-0)
    X=pd.get_dummies(X)
    
    return X, y

def train_model(X_train, y_train, task):
    if task=="classification":
        model=TPOTClassifier(generations=5, population_size=20, random_state=42)
        
    else:
        model=TPOTRegressor(generations=5, population_size=20, random_state=42)
        
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test, task):
    predictions=model.predict(X_test)
    print("\n📊 Evaluation Results:")
    
    if task=="classification":
        acc = accuracy_score(y_test, predictions)
        print(f"Accuracy: {acc:.4f}")
    else:
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        print(f"Mean Squared Error: {mse:.4f}")
        print(f"R^2 Score: {r2:.4f}")

def save_model(model):
    os.makedirs("models", exist_ok=True)
    joblib.dump(model.fitted_pipeline_, "models/best_model.pkl")
    print("\n💾 Modelsaved to 'models/best_model.pkl' successfully.")
    
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