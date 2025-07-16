import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_data(df, target_column):
    df = df.dropna() # Drops missing value rows
    X= df.drop(columns=[target_column]) 
    y=df[target_column]
    
    if y.dtype == object or y.dtype == 'str':
        le = LabelEncoder()
        y = le.fit_transform(y)
    
    #Encoding categorical variables(converting strings to numbers eg. TRUE-1, False-0)
    X=pd.get_dummies(X)
    
    return X, y
