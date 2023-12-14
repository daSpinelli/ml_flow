import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import structlog
import mlflow
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from feature_engine.imputation import MeanMedianImputer
from feature_engine.discretisation import EqualFrequencyDiscretiser
from feature_engine.wrappers import SklearnTransformerWrapper
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier

from utils.utils import load_config, save_model
from evaluation.classifier_eval import ClassifierEvaluation

mlflow.set_tracking_uri(load_config().get('mlflow_uri'))
mlflow.set_experiment(load_config().get('mlflow_experiment_name'))

logger = structlog.get_logger()

class ModelTraining:
    def __init__(self, x_data: pd.DataFrame, y_data: pd.DataFrame) -> None:
        self.x_data = x_data
        self.y_data = y_data
    
    def get_best_model(self):
        logger.info('Getting best model')

        df_mlflow = mlflow.search_runs(
            filter_string='metrics.val_roc_auc_score < 1'
        ).sort_values(by='metrics.val_roc_auc_score', ascending=False)

        best_run_id = df_mlflow.iloc[0]['run_id']
        
        col_params = [
            'params.imputer',
            'params.discretiser',
            'params.scaler',
            'params.class_weight',
            'params.warm_start',
            'params.solver',
            'params.max_iter',
            'params.fit_intercept',
            'params.tol',
            'params.C',
            'params.multi_class'            
        ]
        
        df_best_params = df_mlflow.loc[df_mlflow['run_id'] == best_run_id, col_params]
        
        best_roc_auc_score = df_mlflow.iloc[0]['metrics.val_roc_auc_score']
        
        return df_best_params, best_roc_auc_score
            
    def run(self):
        df_best_params, _ = self.get_best_model()
        
        logger.info(f'Starting model training: {load_config().get("model_name")}')
        
        with mlflow.start_run(run_name='final model'):
            mlflow.set_tag('model_name', load_config().get('model_name'))
            
            model = LogisticRegression(
                warm_start=eval(df_best_params['params.warm_start'].values[0]),
                multi_class=df_best_params['params.multi_class'].values[0],
                class_weight=eval(df_best_params['params.class_weight'].values[0]),
                max_iter=int(df_best_params['params.max_iter'].values[0]),
                C=float(df_best_params['params.C'].values[0]),
                solver=df_best_params['params.solver'].values[0],
                tol=float(df_best_params['params.tol'].values[0])
            )
            
            pipe = Pipeline(
                [
                    ('imputer', eval(df_best_params['params.imputer'].values[0])),
                    ('discretiser', eval(df_best_params['params.discretiser'].values[0])),
                    ('scaler', eval(df_best_params['params.scaler'].values[0])),
                    ('model', model)
                ]
            )
            
            pipe.fit(self.x_data, self.y_data)
            
            y_pred = pipe.predict_proba(self.x_data)[:, 1]
            
            model_eval = ClassifierEvaluation(model, self.x_data, self.y_data)
            
            val_roc_auc = model_eval.evaluate_predictions(self.y_data, y_pred)
            
            mlflow.log_metric('val_roc_auc_score', val_roc_auc)
            
            mlflow.sklearn.log_model(
                pipe,
                load_config().get('model_name'),
                pyfunc_predict_fn='predict_proba',
                input_example=self.x_data.head(1),
                registered_model_name=load_config().get('model_name')
            )