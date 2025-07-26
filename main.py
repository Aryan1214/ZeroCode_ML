import pandas as pd
from sklearn.model_selection import train_test_split
import joblib #running Python functions as pipeline jobs
import os
from modules.preprocessor import preprocess_data
from modules.model_trainer import train_model
from modules.evaluator import evaluate_model

def load_file(file_path): # Function to load a CSV file
    df= pd.read_csv(file_path)
    print(f"File loaded successfully with {df.shape[0]} rows and {df.shape[1]} columns")
    return df

def make_prediction(model, input_values):
    # Check if model has fitted_pipeline_
    if hasattr(model, "fitted_pipeline_"):
        prediction = model.fitted_pipeline_.predict([input_values])
    else:
        prediction = model.predict([input_values])
    
    print(f"\n🏡 Predicted Value: {prediction[0]}")

def save_model(model):
    os.makedirs("models", exist_ok=True)
    joblib.dump(model.fitted_pipeline_, "models/Bostonbest_model.pkl")
    joblib.dump(X.columns.tolist(), "models/columns.pkl")
    print("\n💾 Modelsaved to 'models/' successfully.")
    
if __name__ == "__main__":
    #user input
    file_path = input("📂 Enter the path of csv dataset: ").strip()
    task = input("🗒️ Enter the task (classification or regression):").strip().lower()
    target_column = input("🎯Enter the target column name: ").strip()
    
    #Pipeline steps
    df= load_file(file_path)
    X, y = preprocess_data(df, target_column)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model= train_model(X_train, y_train, task)
    evaluate_model(model, X_test, y_test, task)
    save_model(model)
    
    try:
        input_values = [float(x) for x in input("\n🔢 Enter 13 comma-separated values for prediction: ").split(",")]
        if len(input_values) != len(X.columns):
            print(f"❌ Expected {len(X.columns)} values, but got {len(input_values)}.")
        else:
            make_prediction(model.fitted_pipeline_, input_values)
    except Exception as e:
        print(f"⚠️ Skipping prediction due to input error: {e}")