import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import structlog

from sklearn.pipeline import Pipeline

logger = structlog.get_logger()

class DataPreprocess:
    def __init__(self, pipe: Pipeline) -> None:
        self.pipe = pipe
        self.pipe_trained = None
        
    def train(self, df: pd.DataFrame) -> Pipeline:
        logger.info('Preprocessing started')
        self.pipe_trained = self.pipe.fit(df)
        return self.pipe_trained
        
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.pipe_trained is None:
            raise ValueError('Pipeline is not trained')
        
        logger.info('Transforming data')
        
        data_preprocessed = self.pipe_trained.transform(df)
        
        logger.info('Preprocessing finished')
        
        return data_preprocessed