import pandas as pd
import os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import logging
import yaml
import pickle

# Ensure the "logs" directory exists
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# logging configuration
logger = logging.getLogger('model_training')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'model_training.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# logging handlers added to the logger

logger.addHandler(console_handler)  
logger.addHandler(file_handler)

def load_params(params_path: str) -> dict: # parameters path as input and output is dictionary
    """Load parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug('Parameters retrieved from %s', params_path)
        return params
    except FileNotFoundError:
        logger.error('File not found: %s', params_path)
        raise
    except yaml.YAMLError as e:
        logger.error('YAML error: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error: %s', e)
        raise

def load_data(file_path: str) -> tuple:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        logger.debug('Data loaded from %s', file_path)
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        return X, y
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the data: %s', e)
        raise

def train_model(X_train, y_train, random_state: int = 42):
    """Train a logistic regression model."""
    try:
        model = LogisticRegression(max_iter=1000, random_state=random_state)
        model.fit(X_train, y_train)
        logger.debug('Model trained successfully')
        return model
    except Exception as e:
        logger.error('Error during model training: %s', e)
        raise

def evaluate_model(model, X_test, y_test):
    """Evaluate the model on test data."""
    try:
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        logger.debug('Model evaluation completed')
        logger.debug('Accuracy: %f', accuracy)
        logger.debug('Precision: %f', precision)
        logger.debug('Recall: %f', recall)
        logger.debug('F1 Score: %f', f1)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    except Exception as e:
        logger.error('Error during model evaluation: %s', e)
        raise

def save_model(model, file_path: str) -> None:
    """Save the model to a file using pickle."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as f:
            pickle.dump(model, f)
        logger.debug('Model saved to %s', file_path)
    except Exception as e:
        logger.error('Unexpected error occurred while saving the model: %s', e)
        raise

def main():
    try:
        params = load_params(params_path='params.yaml')
        random_state = params['model_building']['random_state']
        
        # Load data
        X_train, y_train = load_data('./data/processed/train_tfidf.csv')
        X_test, y_test = load_data('./data/processed/test_tfidf.csv')
        
        # Train model
        model = train_model(X_train, y_train, random_state=random_state)
        
        # Evaluate model
        metrics = evaluate_model(model, X_test, y_test)
        
        print(f"\nModel Performance:")
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1 Score:  {metrics['f1']:.4f}")
        
        # Save model
        save_model(model, './models/model.pkl')
        logger.debug('Model training completed successfully')
        
    except Exception as e:
        logger.error('Failed to complete the model training process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()
