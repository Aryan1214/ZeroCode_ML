from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error, r2_score

def evaluate_model(model, X_test, y_test, task):
    predictions=model.predict(X_test)
    print("\n📊 Evaluation Results:")
    
    if task=="classification":
        acc = accuracy_score(y_test, predictions)
        print(f"Accuracy: {acc:.4f}")
        print("Confussion Matrix:")
        print(confusion_matrix(y_test, predictions))
    else:
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        print(f"Mean Squared Error: {mse:.4f}")
        print(f"R^2 Score: {r2:.4f}")