import logging
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import os

#configure logging
logging.basicConfig(filename = 'logs/model_training_instanceMeethod.log',
                    level = logging.INFO,
                    format = '%(asctime)s - %(levelname)s - %(message)s')

class ModelTrainer:
    def __init__(self, df, target_col):
        '''initalize the model trainer class with the data
        df = geature dataframe
        target_col = name of the target column
        '''
        
        self.df = df
        self.target_col = target_col
        self.model = None
    
    def train(self, model, model_name):
        '''
        training a ML model
        model = model instance
        model_name = name of the model
        return trained model
        '''
        
        try:
            logging.info(f'startng model training for {model_name}')
            X= self.df.drop(columns = [self.target_col])
            y = self.df[self.target_col]
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
            
            self.model = model
            self.model.fit(X_train, y_train)
            
            #evaluate the model
            y_pred = self.model.predict(X_test)
            report = classification_report(y_test, y_pred)
            logging.info(f"model evaluation for {model_name}: \n{report}")
            print(f"model evaluation for {model_name}: \n{report}")
            
            return self.model
        
        except Exception as e:
            logging.error(f'Error during model training for {model_name}: {e}')
            print(f'Error during model training for {model_name}: {e}')
            return None
        
    def save_model(self, model_name, file_path = "models/"):
        
        try:
            if self.model is None:
                raise ValueError('no trained model found. train the model first')
            
            #create the directory if it doesnt exist
            os.makedirs(file_path, exist_ok = True)
            
            #generate file name
            file_name = os.path.join(file_path, f'{model_name}.pkl')
            
            #save the model
            
            joblib.dump(self.model, file_name)
            logging.info(f'model {model_name} saved successfully at {file_name}')
            print(f'model {model_name} saved successfully at {file_name}')
            
        except Exception as e:
            logging.error(f'error in saving the model {model_name}: {e}')
            print(f'error in saving the model {model_name}: {e}')
            


def get_latest_processed_file(dir):
        try:
            files = [os.path.join(dir, f) for f in os.listdir(dir) if f.endswith('.csv')]
            
            if not files:
                raise FileNotFoundError('no processed file found')
            
            latest_file = max(files, key = os.path.getmtime)
            logging.info(f'latest processed file: {latest_file}')
            print(f'latest processed file: {latest_file}')
            
            return latest_file
        except Exception as e:
            logging.error(f'file not found: {e}')
            print(f'file not found: {e}')
            return None    
            
if __name__ == "__main__":
    import pandas as pd
    import os
    
        # Example usage
    processed_dir = 'data/processed/'
    latest_file = get_latest_processed_file(processed_dir)
    if latest_file:
        
    #data_file = "data/processed/processed_data.csv"  # Ensure this path is correct
        df = pd.read_csv(latest_file)
        
        # Define multiple models to train
        models = {
            "Logistic regression" : LogisticRegression(random_state = 42)
            
            #"RandomForest": RandomForestClassifier(random_state=42),
            #"GradientBoosting": GradientBoostingClassifier(random_state=42),
            #"SVC": SVC(random_state=42)
            
        }
        
        # Train and save each model
        for model_name, model in models.items():
            model_trainer = ModelTrainer(df, target_col="fraudulent")
            trained_model = model_trainer.train(model, model_name)
            
            if trained_model:
                model_trainer.save_model(model_name)
    else:
        print('no processed file found')
    
            
            