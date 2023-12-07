import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import structlog

from utils.utils import load_config, save_model

logger = structlog.get_logger()

class ModelTraining:
    def __init__(self, x_data: pd.DataFrame, y_data: pd.DataFrame) -> None:
        self.x_data = x_data
        self.y_data = y_data
        
    def train(self, model):               
        logger.info(f'Training model {model.__class__.__name__}')
        try:
            model.fit(self.x_data, self.y_data)
            logger.info(f'Model trained successfully.')
            save_model(model, load_config().get('model_name'))
            return model
        except Exception as e:
            logger.error(f'Error training model: {e}')