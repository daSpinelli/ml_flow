import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import structlog

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score, StratifiedKFold

from utils.utils import load_config, save_model

logger = structlog.get_logger()

class ClassifierEvaluation:
    def __init__(self, model, x: np.ndarray, y: np.ndarray, k_fold: int = 5) -> None:
        self.model = model
        self.x = x
        self.y = y
        self.k_fold = k_fold
    
    def cross_val_eval(self):
        logger.info(f'Cross validation evaluation for model {self.model.__class__.__name__} started.')
        
        skf = StratifiedKFold(n_splits=self.k_fold, shuffle=True, random_state=load_config().get('random_state'))
        
        scores = cross_val_score(self.model, self.x, self.y, cv=skf, scoring='roc_auc')
        
        logger.info(f'Cross validation evaluation for model {self.model.__class__.__name__} finished.')
        
        return scores
    
    def roc_auc_scorer(self, model, x, y):
        y_pred = model.predict_proba(x)[:, 1]
        return roc_auc_score(y, y_pred)
    
    @staticmethod
    def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray):
        logger.info(f'Evaluation of predictions started.')
        try:
            roc_auc = roc_auc_score(y_true, y_pred)
            logger.info(f'ROC AUC score: {roc_auc}')
            return roc_auc
        except Exception as e:
            logger.error(f'Error evaluating predictions: {e}')