import os
from pathlib  import Path
import sys
import numpy as np
from src.exceptions.exception import CustomException
from src.logger.logging import logging
from src.utils.utils import save_obj
from dataclasses import dataclass
import pandas as pd
from src.components.Data_ingestion import DataIngestion
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,OneHotEncoder
@dataclass
class DataTransformConfig:
    preprocesser=os.path.join("artifacts",'preprocesser.pkl')

class DataTransformation:
    def __init__(self):
        logging.info("Data Transformation have been started")
        self.data_transform=DataTransformConfig()
    
    def Preprocessor(self):
        logging.info("Creating the method to create pkl file")
        cat_cols=['gender', 'Partner', 'Dependents' 'PhoneService', 'MultipleLines', 'InternetService',
       'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
       'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
       'PaymentMethod']
        
        num_cols=['SeniorCitizen','tenure','MonthlyCharges']

        num_pipeline=Pipeline(steps=[
            ('SimpleImputer',SimpleImputer(strategy='median')),
            ('StandardScaler',StandardScaler())
        ])
        cat_pipeline=Pipeline(steps=[
            ('SimpleImputer',SimpleImputer(strategy='most_frequent')),
            ('OneHotEncoder',OneHotEncoder(handle_unknown='ignore')),
            ('StandardScaler',StandardScaler())
        ])

        preprocessor=ColumnTransformer([
            ('num_pipeline',num_pipeline,num_cols),
            ('cat_pipeline',cat_pipeline,cat_cols)
        ])

        return preprocessor
    

    def InstantiateDataPreprocessing(self,train_path,test_path):
        try:
         logging.info('Data is loading for preprocess')
         train_df=pd.read_csv(train_path)
         test_df=pd.read_csv(test_path)
         dropped_cols=['Unnamed :0','Churn']
         train_df_input=train_df.drop(dropped_cols,axis=1)
         test_df_input=test_df.drop(dropped_cols,axis=1)
         train_df_input.drop('Unnamed :0',axis=1,inplace=True)
         test_df_input.drop('Unnamed :0',axis=1,inplace=True)
         test_df_output=test_df['Churn']
         train_df_output=train_df['Churn']
         logging.info('Loading the Preprocessor')
         preprocessor=self.Preprocessor()
         train_df_input_arr=preprocessor.fit_transform(train_df_input)
         test_df_input_arr=preprocessor.fit_transform(test_df_input)
         train_arr=np.c_[train_df_input_arr,np.array(train_df_output)]
         test_arr=np.c_[test_df_input_arr,np.array(test_df_output)]
         logging.info('Data Preprocessing completed')

         save_obj(
             self.data_transform.preprocesser,preprocessor
         )

         return train_arr,test_arr
        
        except Exception as e:
            raise CustomException(e,sys)




obj= DataIngestion()

train_path,test_path = obj.data_ingestion_instatiate()

data_transform=DataTransformation()

train_arr,test_arr=data_transform.InstantiateDataPreprocessing(train_path,test_path)
