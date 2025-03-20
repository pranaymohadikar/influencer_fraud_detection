import logging
import pandas as pd
import numpy as np
import os
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

#logging configuration

logging.basicConfig(filename = 'logs/feature_engineering_instanceMethod.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

#using instance method

class FeatureEngineeringInstance:
    
    def __init__(self, data_path):
        #initializing class with the dataset path
        self.data_path = data_path
        self.df = None
        
    def processing_data(self):
        #processing the dataset with feature engineering, standardization and balancing the data
        try:
            logging.info('Starting feature engineering')
            
            #to check if file exist
            if not os.path.exists(self.data_path):
                raise FileNotFoundError(f'file not found:{self.data_path}')
        
            #reading file
            # try:
            #     self.df = pd.read_csv(self.data_path)
            # except Exception as e:
            #     print(f'error as {e}')
            # Try reading the CSV with different encodings
            try:
                self.df = pd.read_csv(self.data_path, encoding='utf-8')
            except UnicodeDecodeError:
                self.df = pd.read_csv(self.data_path, encoding='ISO-8859-1')  # Alternative encoding

                
                #to check if dataframe of empty or not
            if self.df is None or self.df.empty:
                raise ValueError('file is empty')
                
            print('raw data preview',self.df.head())
            
            #creating new features
            
            logging.info('creating engagement rate')
            self.df['engagement_rate'] = (self.df['avg_likes'] + self.df['avg_comments'])/self.df['followers']
            
            logging.info('creating follower growth rate')
            self.df['follower_growth_rate'] = np.random.normal(-0.1, 0.5, size=len(self.df))
            
            logging.info('creating suspicious comment ratio')
            self.df['suspicious_comment_ratio'] = np.random.uniform(0, 0.3, size= len(self.df))
            
            # Label as fraudulent (1) or genuine (0)
            # Fraudulent if: low engagement rate or high suspicious comment ratio
            logging.info('generating labels for the fraud detection')
            self.df['fraudulent'] = ((self.df['engagement_rate'] < 0.01) | (self.df['suspicious_comment_ratio'] > 0.2)).astype(int)
            
            #standardizing numerical columns
            num_cols = ['followers', 'avg_likes', 'avg_comments', 
                            'engagement_rate', 'follower_growth_rate', 'suspicious_comment_ratio']
            
            logging.info('applying standard scaler to numerical columns')
            scaler = StandardScaler()
            self.df[num_cols] = scaler.fit_transform(self.df[num_cols])
            
            print('standardized dataset',self.df[num_cols])
            
            #handling imbalanced dataset using SMOTE
            
            logging.info('handling class imbalance by using SMOTE')
            X = self.df[num_cols]
            y = self.df['fraudulent']
            
            smote = SMOTE(sampling_strategy='auto', random_state=42)
            X_resampled,y_resampled = smote.fit_resample(X,y)
            
            #creating new balanced dataset
            
            df_resampled = pd.DataFrame(X_resampled, columns = num_cols)
            df_resampled['fraudulent'] = y_resampled
            
            logging.info('data processing completed successfully')
            
            print('final data preview',df_resampled.head())
            
            #save data to new file
            
            output_dir = 'data/processed/'
            os.makedirs(output_dir,exist_ok=True) #create dir if not exist
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(output_dir, f'processed_data_{timestamp}.csv')
            
            df_resampled.to_csv(output_file, index = False)
            
            logging.info(f'process data saved to : {output_file}')
            print(f'processed data saved to : {output_file}')
            
            return df_resampled
    
        except Exception as e:
            logging.error(f'error found during processing : {e}')
            print(f'error found during processing: {e}')
            return None
    
    
if __name__ == "__main__":
    data_file = r"E:\KaaM\Influencer_Fraud_Detection\data\influencer_fraud_detection_synthetic_data.csv"  # Update with the correct path
    
    #create an instance of featureEngineering
    feature_engineering = FeatureEngineeringInstance(data_file)
    
    #call the instance
    processed_df = feature_engineering.processing_data()
    
    if processed_df is not None:
        print('data processing completed successfully')
    else:
        print('error in data processing. check the logs for details. ')
        

        
        
        
        
        
        
        
        
        