from tpot import TPOTClassifier, TPOTRegressor

def train_model(X_train, y_train, task):
    if task=="classification":
        model=TPOTClassifier(generations=5, population_size=20, random_state=42)
        
    else:
        model=TPOTRegressor(generations=5, population_size=20, random_state=42)
        
    model.fit(X_train, y_train)
    return model