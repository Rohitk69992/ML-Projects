import os
import sys
import numpy as np
import pandas as pd
import dill
from src.exception import CustomException
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

def save_object(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,'wb') as file_obj:
            dill.dump(obj,file_obj)
    except Exception as e:
        raise CustomException(e,sys)
    
def evaluate_model(X_train, X_test, y_train, y_test, models, param):
    try:
        report = {}
        
        for model_name, model in models.items():
            # Get the hyperparameter grid specifically for this model name
            model_param = param.get(model_name, {})  # Defaults to empty dict if missing
            
            # Hyperparameter Tuning using Grid Search
            gs = GridSearchCV(model, model_param, cv=3)
            gs.fit(X_train, y_train)
            
            # Configure model with the best hyperparameter setup
            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)
            
            # Generate predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            # Evaluate Performance
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)
            
            # Store the test score in our final report map
            report[model_name] = test_model_score
            
        return report
    
    except Exception as e:
        raise CustomException(e, sys)
