import pandas as pd
import numpy as np
import json
from pathlib import Path
import logging
import joblib
import os
import bentoml
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import ElasticNet, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_FOLDER = BASE_DIR / "data" / "processed"
OUTPUT_FOLDER = BASE_DIR / "models" 
CONFIG_FOLDER = BASE_DIR / "config"

# Mapping for model names to classes
MODEL_MAPPING = {
    "ridge": Ridge,
    "elasticnet": ElasticNet,
    "lasso": Lasso,
    "randomforestregressor": RandomForestRegressor,
    "gradientboostingregressor": GradientBoostingRegressor
}

def load_config():
    """
    Load the model selection config (config.json).
    """
    with open(os.path.join(CONFIG_FOLDER, "config.json")) as f:
        config = json.load(f)
    return config

def load_param_grid():
    """a
    Load the parameter grid config (param_grid.json).
    """
    with open(os.path.join(CONFIG_FOLDER, "param_grid.json")) as f:
        param_grid = json.load(f)
    return param_grid

def main():
    """ Runs a GridSearchCV with the selected model from config.json and saves the best parameters in models/best_params.skl
    """
    config = load_config()
    param_grid = load_param_grid()
    model_name = config.get("model_name")
    projectname = "admission"
    logger = logging.getLogger(__name__)
    logger.info(f'finding best parameters using GridSearch with model {model_name}')
    
    
    input_filepath_test_x = os.path.join(INPUT_FOLDER, "X_test.csv")
    input_filepath_train_x = os.path.join(INPUT_FOLDER, "X_train.csv")
    input_filepath_test_y = os.path.join(INPUT_FOLDER, "y_test.csv")
    input_filepath_train_y = os.path.join(INPUT_FOLDER, "y_train.csv")
    
    try:
        # Get the model and parameter grid based on user input
        model, param_grid = get_model_and_params(model_name, param_grid)
    except ValueError as e:
        # Handle invalid model names
        print(e)
        return
    
    # Call the main data processing function with the provided file paths
    best_params = find_best_params(input_filepath_test_x, input_filepath_train_x, 
                input_filepath_test_y, input_filepath_train_y,
                model, param_grid)
    
    trained_model = train_model(input_filepath_train_x, 
                input_filepath_train_y,
                model, best_params, projectname, model_name)
    
    metrics = evaluate_model(trained_model, input_filepath_test_x, input_filepath_test_y)
    
    print(f"Model evaluation metrics: {metrics}")
    
def train_model(input_filepath_train_x, 
                input_filepath_train_y, 
                model, best_params, projectname, model_name):
 
    #--Importing dataset
    X_train = pd.read_csv(input_filepath_train_x, sep=",")
    y_train = pd.read_csv(input_filepath_train_y, sep=",")
    
    y_train = y_train.values.ravel()
    
    #trained_model = model(best_params)
    model.set_params(**best_params)
    model.fit(X_train, y_train)

    print(f"Training complete with parameters: {best_params}")
    
    model_ref = bentoml.sklearn.save_model(projectname+"_"+model_name, model)
    
    output_filepath = os.path.join(OUTPUT_FOLDER, f"{projectname}_{model_name}.pkl")
    joblib.dump(model, output_filepath)
    print(f"Model saved as: {model_ref}")
    
    return model


def find_best_params(input_filepath_test_x, input_filepath_train_x, 
                 input_filepath_test_y, inpút_filepath_train_y, 
                 model, param_grid):
 
    #--Importing dataset
    X_train = pd.read_csv(input_filepath_train_x, sep=",")
    X_test = pd.read_csv(input_filepath_test_x, sep=",")
    y_train = pd.read_csv(inpút_filepath_train_y, sep=",")
    y_test = pd.read_csv(input_filepath_test_y, sep=",")
    
    y_train = y_train.values.ravel()
    y_test = y_test.values.ravel()
    
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, 
                               scoring='neg_mean_squared_error', cv=5, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    
    print(f"Best parameters: {best_params}")
    print(f"Best score: {best_score}")
    
    return best_params

def evaluate_model(model, input_filepath_test_x, input_filepath_test_y):
    """
    Evaluate the model using test data and calculate various metrics.
    """
    # Predict with the trained model
    X_test = pd.read_csv(input_filepath_test_x, sep=",")
    y_test = pd.read_csv(input_filepath_test_y, sep=",")
    
    y_pred = model.predict(X_test)
    y_columns = y_test.columns
    # Calculate MSE, R^2, and optionally more metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    # Store metrics in a dictionary
    metrics = {
        "mse": mse,
        "r2": r2,
        "mae": mae,
    }
    
    y_pred = pd.DataFrame(y_pred, columns = y_columns)

    return metrics   
         
def get_model_and_params(model_name, param_grid):
    """
    Returns the model and parameter grid for a given model name.
    """
    model_name = model_name.lower()
    if model_name in param_grid:
        model_config = param_grid[model_name]
        model_class = MODEL_MAPPING.get(model_name)
        
        if not model_class:
            raise ValueError(f"Model '{model_name}' not found in MODEL_MAPPING.")
        
        model = model_class()  # Instantiate the model
        param_grid = model_config['param_grid']  # Extract the parameter grid
        
        return model, param_grid
    else:
        raise ValueError(f"Model '{model_name}' not found in param_grid.")


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]


    main()