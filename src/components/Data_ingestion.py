import os
from pathlib  import Path
import sys
from src.exceptions.exception import CustomException
from src.logger.logging import logging
from dataclasses import dataclass
import pandas as pd
from sklearn.model_selection import train_test_split
@dataclass
class DataIngestConfig:
    train_data_path=os.path.join('artifacts','train_data.csv')
    test_data_path=os.path.join('artifacts','test_data.csv')
    raw_data_path=os.path.join('artifacts','raw_data.csv')

class DataIngestion:
    def __init__(self):
        logging.info('Data Ingestion have been started')
        self.ingest_config=DataIngestConfig()
    
    def data_ingestion_instatiate(self):
        try:
         logging.info("Data loading.....")
         df=pd.read_csv("C:/Users/admin/Desktop/Data.csv")
         logging.info('Raw data creating....')
         os.makedirs(os.path.dirname(os.path.join(self.ingest_config.raw_data_path)), exist_ok=True)
         df.to_csv(self.ingest_config.raw_data_path,index=False)
         logging.info('Raw data have been created')
         train_data,test_data=train_test_split(df,test_size=0.3,random_state=45)
         logging.info('Training and testing data creation have been started')
         train_data.to_csv(self.ingest_config.train_data_path,index=False)
         test_data.to_csv(self.ingest_config.test_data_path,index=False)
         logging.info('Train and test data have been created')
         return(
            self.ingest_config.train_data_path,self.ingest_config.test_data_path
         )
        except Exception as e:
           raise CustomException(e,sys)
        


if __name__=="__main__":
   obj=DataIngestion()
   train_path, test_path = obj.data_ingestion_instatiate()