import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname('__file__'), '..'))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

import sqlite3
import pandas as pd

from evidently import report
from evidently import metrics
from evidently import metric_preset
from evidently import test_preset

from level_2.src.data import data_load

class ModelMonitor():
    def __init__(self) -> None:
        self.query = 'SELECT * FROM predictions'
        
    def get_pred_data(self):
        conn = sqlite3.connect(os.path.join(ROOT_DIR, 'preds.db'))
        
        df = pd.read_sql_query(self.query, conn)
        conn.close()
        
        return df
    
    def get_training_data(self):
        dl = data_load.DataLoad()
        train_data_file = os.path.join(ROOT_DIR, 'level_2', 'data', 'raw', 'train.csv')
        df_train = dl.run(train_data_file, index_col=0)
        
        return df_train
    
    def run(self):
        df_cur = self.get_pred_data()
        df_ref = self.get_training_data().drop(columns=['target'])
        
        model_card = report.Report(
            metrics=[
                metrics.DatasetSummaryMetric(),
                metric_preset.DataDriftPreset(),
                metrics.DatasetMissingValuesMetric()
            ]
        )
        
        model_card.run(reference_data=df_ref, current_data=df_cur)
        
        model_card.save(os.path.join(ROOT_DIR, 'level_2', 'docs', 'model_monitoring.html'))
        
mm = ModelMonitor()
mm.run()
    