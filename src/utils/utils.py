import os
import sys
from pathlib import Path
from src.exceptions.exception import CustomException
from src.logger.logging import logging
from imblearn.combine import SMOTETomek
from sklearn.metrics import accuracy_score
import pickle
import pandas as pd
def save_obj(file_path,obj):
    try:
     file_path=os.path.dirname(file_path)
     os.makedirs(file_path, exist_ok=True)
     with open(file_path, 'wb') as f:
        pickle.dump(obj, f)
    except Exception as e:
       raise CustomException(e,sys)


def Upsampling(X_train,X_test,y_train,y_test):
    try:
     smote=SMOTETomek()
     X_train,y_train=smote.fit_resample(X_train,y_train)
     X_test,y_test=smote.fit_resample(X_test,y_test)
     return X_train,X_test,y_train,y_test
    except Exception as e:
       raise CustomException(e,sys)

def evaluate_model(X_train,y_train,X_test,y_test,models):
    try:
        report = {}
        for i in range(len(models)):
            model = list(models.values())[i]
            # Train model
            model.fit(X_train,y_train)

            

            # Predict Testing data
            y_test_pred =model.predict(X_test)

            # Get R2 scores for train and test data
            #train_model_score = r2_score(ytrain,y_train_pred)
            test_model_score = accuracy_score(y_test,y_test_pred)

            report[list(models.keys())[i]] =  test_model_score

        return report

    except Exception as e:
        logging.info('Exception occured during model training')
        raise CustomException(e,sys)

