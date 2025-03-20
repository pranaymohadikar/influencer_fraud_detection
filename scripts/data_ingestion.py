import logging
import os
import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(filename = 'logs/data_ingestion.log',
                    level = logging.INFO,
                    format = "%(asctime)s - %(levelname)s - %(message)s")

class DataIngestion:
    def __init__(self, raw_data_path, output_dir = 'data/'):
        self.raw_data_path = raw_data_path
        self.output_dir = output_dir
        self.train_data_path = os.path.join(output_dir, "train/train_data.csv")
        self.test_data_path = os.path.join(output_dir, "test/test_data.csv")
        self.df = None
    def load_data(self):
        try:
            logging.info('loading raw data')
            if not os.path.exists(self.raw_data_path):
                raise FileNotFoundError(f'file not found:{self.raw_data_path}')
            
            self.df = pd.read_csv(self.raw_data_path)
            
            if self.df is None or self.df.empty:
                raise ValueError("raw data not found")
            
            logging.info('raw data loaded successfully')
            print('raw data: \n',  self.df.head())
            return self.df
        except Exception as e:
            logging.error(f'taw data loading error: {e}')
            print(f'taw data loading error: {e}')
    
    def data_split(self, df, test_size = 0.2, random_state = 42):
        try:
            logging.info('splitting of the data into train and test')
            train_df, test_df = train_test_split(self.df, test_size = test_size, random_state = random_state)
            logging.info("data split successful")
            return train_df, test_df
        except Exception as e:
            logging.error(f'error in splitting: {e}')
            print(f'error in splitting: {e}') 
    
    def save_data(self, train_df, test_df):
        try:
            logging.info('saving th train anad test data files')
            os.makedirs(os.path.dirname(self.train_data_path), exist_ok = True)
            os.makedirs(os.path.dirname(self.test_data_path), exist_ok = True)
            
            train_df.to_csv(self.train_data_path, index = False)
            test_df.to_csv(self.test_data_path, index = False)
            
            logging.info(f'train data saved to: {self.train_data_path}')
            logging.info(f'test data saved to: {self.test_data_path}')
            
            print(f'train data saved to: {self.train_data_path}')
            print(f'test data saved to: {self.test_data_path}')
        
        except Exception as e:
            logging.error(f'error in saving data: {e}')
            print(f'error in saving data: {e}')
            
    
    def run(self):
        try:
            logging.info('starting data ingestion pipeline')
            self.df = self.load_data()
            train_df, test_df = self.data_split(self.df)
            self.save_data(train_df,test_df)
            logging.info('data ingestion completed')
        except Exception as e:
            logging.info(f'error in data ingestion: {e}')
            print(f'error in data ingestion: {e}')


if __name__ == "__main__":
    raw_data_path =  "E:\KaaM\Influencer_Fraud_Detection\data\influencer_fraud_detection_synthetic_data.csv"  
    
    data_ingestion =  DataIngestion(raw_data_path)
    
    try:
        data_ingestion.run()
        print('data ingestion completed')
    except Exception as e:
        print('error in data ingestion')    
                      
            