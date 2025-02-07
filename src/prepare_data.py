import pandas as pd
import joblib
import numpy as np
from pathlib import Path
import logging
import os
from sklearn.model_selection import train_test_split  
from sklearn.preprocessing import MinMaxScaler

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_FOLDER = BASE_DIR / "data" / "raw"
OUTPUT_FOLDER = BASE_DIR / "data" / "processed"
MODEL_FOLDER = BASE_DIR / "models"

def main():
    """ Performs Train/test split on data
    """
    
    logger = logging.getLogger(__name__)
    logger.info('Cleaning data and splitting into Test/Train')
    
    data_filepath = os.path.join(INPUT_FOLDER, "admission.csv")
    output_folderpath = OUTPUT_FOLDER
    
    # Call the main data processing function with the provided file paths
    create_datasets(data_filepath, output_folderpath)

def create_datasets(data_filepath, output_folderpath):
    
    #--Importing dataset
    df = pd.read_csv(data_filepath, sep=",")
    df.drop(columns=["Serial No."], inplace=True)
    X = df.drop(columns=["Chance of Admit "])
    y = df["Chance of Admit "]
    X_column = X.columns
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    X_train = pd.DataFrame(X_train, columns = X_column)
    X_test = pd.DataFrame(X_test, columns = X_column)
    y_train = pd.DataFrame(y_train)
    y_test = pd.DataFrame(y_test)
    
    print("Data has been prepared.")

    #--Saving the dataframes to their respective output file paths
    for file, filename in zip([X_test, X_train, y_test, y_train], ['X_test', 'X_train', 'y_test', 'y_train']):
        output_filepath = os.path.join(output_folderpath, f'{filename}.csv')
        file.to_csv(output_filepath, index=False)
        
    joblib.dump(scaler, os.path.join(MODEL_FOLDER, 'scaler.pkl'))
    
            
       

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]


    main()
