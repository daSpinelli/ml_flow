import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import structlog

from sklearn.model_selection import train_test_split
from utils.utils import load_config

logger = structlog.get_logger()

class DataTransform():
    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df
        self.target = load_config().get('target_column')
        
    def train_test_split(self) -> tuple:
        x = self.df.drop(self.target, axis=1)
        y = self.df[self.target]
        
        x_train, x_test, y_train, y_test =  train_test_split(
                                                x, 
                                                y, 
                                                test_size=load_config().get('test_size'),
                                                stratify=y,
                                                random_state=load_config().get('random_state')
                                            )
        
        return x_train, x_test, y_train, y_test
        