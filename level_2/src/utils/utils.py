import os
import yaml
import joblib
import structlog

logger = structlog.get_logger()

def load_config():

    curr_path = os.path.dirname(os.path.join(os.path.abspath(__file__), ))    
    config_path = os.path.join(curr_path, '..', '..', 'config')    
    config_file_name = 'config.yaml'
    
    config_file_path = os.path.abspath(os.path.join(config_path, config_file_name))
    
    config_file = yaml.safe_load(open(config_file_path, 'rb'))
    
    return config_file

def save_model(model, model_name):
    curr_path = os.path.dirname(os.path.join(os.path.abspath(__file__), ))    
    models_path = os.path.join(curr_path, '..', '..', 'models')    
    
    models_file_path = os.path.abspath(os.path.join(models_path, model_name))
    
    try:
        joblib.dump(model, models_file_path)
        logger.info(f'Model saved successfully in {models_file_path}')
    except Exception as e:
        logger.error(f'Error saving model: {e}')