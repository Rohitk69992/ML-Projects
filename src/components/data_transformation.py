import sys
import numpy as np
import os
from dataclasses import dataclass
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_ob_file_path=os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
        
    def get_data_transformer_object(self):
        """This function is responsible for data transformation"""
        try:
            numerical_features=['writing_score','reading_score']
            categorical_featues=['gender','race_ethnicity','parental_level_of_education','lunch','test_preparation_course']
            
            num_pipeline=Pipeline(
                steps=[("Imputer",SimpleImputer(strategy='median')),("Scaler",StandardScaler())])
            cat_pipeline = Pipeline(
                        steps=[
                            ("Imputer", SimpleImputer(strategy='most_frequent')),
                            ("OneHotEncoder", OneHotEncoder(sparse_output=False)),   # Cleanly closed tuple
                            ("Scaler", StandardScaler())          # Cleanly closed tuple
                        ])
            logging.info("Numerical Columns and Standard Scaling completed")
            logging.info("Categorical Columns encoding completed")
            
            preprocessor=ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,numerical_features),
                    ("cat_pipeline",cat_pipeline,categorical_featues)
                ]
            )
            
            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)
            
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            logging.info("Reading of train and test data completed")
            logging.info("Obtaining preprocessing object")
            preprocessing_obj=self.get_data_transformer_object()
            target_column_name="math_score"
            numerical_features=['writing_score','reading_score']
            
            input_feature_train_df = train_df.drop(columns=[target_column_name])
            target_feature_train_df = train_df[target_column_name]
            
            input_feature_test_df = test_df.drop(columns=[target_column_name])
            target_feature_test_df = test_df[target_column_name]
            logging.info("Applying proprocessing object on the training and testing dataframe ")
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)
            
            train_arr = np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr,np.array(target_feature_test_df)]
            logging.info("Saving preprocessing object")
            save_object(file_path=self.data_transformation_config.preprocessor_ob_file_path,obj=preprocessing_obj)
            
            return(train_arr,test_arr,self.data_transformation_config.preprocessor_ob_file_path)
            
        except Exception as e:
            raise CustomException(e,sys)