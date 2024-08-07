import os
import sys
from pathlib import Path
from src.exceptions.exception import CustomException
from src.logger.logging import logging
from src.components.Data_ingestion import DataIngestion
from src.components.Data_Transformation import DataTransformation
from src.utils.utils import Upsampling,evaluate_model,save_obj
from dataclasses import dataclass
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

@dataclass
class ModelTrainerConfig:
    model_config = os.path.join('artifacts','model.pkl')

class ModelTrainer:
    logging.info('Model Training have been started')
    
    def __init__(self):
        self.modelconfig=ModelTrainerConfig()
    
    def Instantiate_Model_Triner(self,train_arr,test_arr):
        try:
            logging.info('Loading and splitting the arrays')
            X_train,X_test,y_train,y_test=train_arr[:,:-1],train_arr[:,-1],test_arr[:,:-1],test_arr[:,-1]
            logging.info('Arrays have been loaded and splitted')
            X_train,X_test,y_train,y_test=Upsampling(X_train,X_test,y_train,y_test)
            logging.info('Models have been upsampled')
            Models={
                'LogisticRegression':LogisticRegression(C=100,penalty='l1',Literal='liblinear'),
                'RandomForest':RandomForestClassifier(criterion='entropy',n_estimators=100),
                'GradientBoosting':GradientBoostingClassifier(learning_rate=0.2,n_estimators=200),
                'NaiveBayes':GaussianNB(),
                'KNN': NearestNeighbors(n_neighbors=5),
                'Adaboost': AdaBoostClassifier(learning_rate=0.7,n_estimators=200)
            }

            model_report:dict=evaluate_model(Models,X_train,X_test,y_train,y_test)
            logging.info('Models training have been completed successfully')
            
            best_model_score=max(sorted(model_report.values()))

            best_model_name=model_report[list(model_report.keys())][list(model_report.values()).index(best_model_score)]
            best_model=model_report[best_model_name]
            logging.info(f'Best Model: {best_model_name} with Score: {best_model_score} and model is {best_model}')
            save_obj(self.modelconfig.model_config,best_model)
            logging.info(f'Model saved')
            return X_test,y_test
        except Exception as e:
            raise CustomException(e,sys)
                  

if __name__ == '__main__':
    data_obj=DataIngestion()
    train_path,test_path=data_obj.Ingest_data()
    data_transform=DataTransformation()
    train_arr,test_arr=data_transform.InstantiateDataPreprocessing(train_path,test_path)
    model_trainer=ModelTrainer()
    X_test,y_test=model_trainer.Instantiate_Model_Triner(train_arr,test_arr)