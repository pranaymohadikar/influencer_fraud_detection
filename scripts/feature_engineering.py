import logging
import pandas as pd
import numpy as np
import os
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# Logging configuration
logging.basicConfig(filename='logs/feature_engineering.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

class FeatureEngineering:
    @staticmethod
    def processed_data(data):
        try:
            logging.info("Starting feature enginnering")

            # Check if the file exists
            if not os.path.exists(data):
                raise FileNotFoundError(f" File not found: {data}")

            # Try reading the CSV with different encodings
            try:
                df = pd.read_csv(data, encoding='utf-8')
            except UnicodeDecodeError:
                df = pd.read_csv(data, encoding='ISO-8859-1')  # Alternative encoding

            # Check if the DataFrame is empty
            if df is None or df.empty:
                raise ValueError(" CSV file is empty!")

            # Debug: Print first few rows to check data
            print("\nRaw Data Preview:\n", df.head())

            # # Check if required columns exist
            # required_columns = ['followers', 'avg_likes', 'avg_comments']
            # missing_columns = [col for col in required_columns if col not in df.columns]

            # if missing_columns:
            #     raise ValueError(f"Missing required columns: {missing_columns}")

            # Removing duplicates
            #df.drop_duplicates(inplace=True)

            # Creating new features
            logging.info("Creating engagement_rate")
            df['engagement_rate'] = (df['avg_likes'] + df['avg_comments']) / df['followers']

            logging.info("Creating follower_growth_rate")
            df['follower_growth_rate'] = np.random.uniform(-0.1, 0.5, size=len(df))

            logging.info("Creating suspicious_comment_ratio")
            df['suspicious_comment_ratio'] = np.random.uniform(0, 0.3, size=len(df))
            
            logging.info('generating labels for the data, ')
             # Label as fraudulent (1) or genuine (0)
             # Fraudulent if: low engagement rate or high suspicious comment ratio
            df["Fraudulent"] = ((df["engagement_rate"] < 0.01) | (df["suspicious_comment_ratio"] > 0.2)).astype(int)

            
            
            #adding standard scaler
            
            num_cols = ['followers', 'avg_likes', 'avg_comments', 
                            'engagement_rate', 'follower_growth_rate', 'suspicious_comment_ratio']
            logging.info("using standard scaler")
            scaler = StandardScaler()
            df[num_cols] = scaler.fit_transform(df[num_cols])
            print(df.head())
            
            
            logging.info('handling imbalanced data')
            X=df[num_cols]
            y =df['Fraudulent']
            
            smote = SMOTE(sampling_strategy='auto', random_state=42)
            X_resampled,y_resampled = smote.fit_resample(X,y)
            
            #creating new balanced dataset
            df_resampled = pd.DataFrame(X_resampled, columns = num_cols)
            df_resampled['Fraudulent'] = y_resampled
            
            logging.info("Data processing completed successfully")

            # Debug: Print processed data preview
            print("\n Processed Data Preview:\n", df_resampled.head())
            

            # Save processed data to a new CSV file
            output_dir = "data/processed/"
            os.makedirs(output_dir, exist_ok=True)  # Create directory if it doesn't exist

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(output_dir, f"processed_data_{timestamp}.csv")

            df_resampled.to_csv(output_file, index=False)

            logging.info(f"Processed data saved to: {output_file}")
            print(f"\n Processed data saved to: {output_file}")

            return df
        except Exception as e:
            logging.error(f" Error found during processing: {e}")
            print(f"\n Error found during processing: {e}")
            return None

if __name__ == "__main__":
    data_file = r"E:\KaaM\Influencer_Fraud_Detection\data\influencer_fraud_detection_synthetic_data.csv"  # Update with the correct path
    processed_df = FeatureEngineering.processed_data(data_file)
    
    if processed_df is not None:
        print("\nData processing completed successfully.")
    else:
        print("\nError in data preocessing. Check the logs for details.")
