import requests
import json
import mlflow
import mlflow.pyfunc
import pandas as pd
import numpy as np
import structlog
import sqlite3


conn = sqlite3.connect('../../preds.db')
cursor = conn.cursor()

logger = structlog.get_logger()

mlflow.set_tracking_uri('http://localhost:5000')
mlflow.set_experiment('prob_loan')

class Predict():
    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df
        self.endpoint = 'http://localhost:5001/invocations'
        
    def run(self):
        logger.info('Starting model prediction')
        
        
        to_inference = {
            'dataframe_split': {
                'columns': self.df.columns.tolist(),
                'data': self.df.head(5).replace(np.nan, None).values.tolist()
            }
        }
        
        response = requests.post(
            url=self.endpoint,
            json=to_inference,
            headers={'Content-Type': 'application/json'}
        )
        
        if response.status_code != 200:
            logger.error(f'Status code: {response.status_code}\n{response.text}')
            raise Exception('Error connecting to endpoint')
        
        logger.info('Model prediction completed')
        
        probabilities = np.array(json.loads(response.text).get('predictions', []))[:, 1]
        
        df_probs = self._results(probabilities)
        
        self._capture_inputs_and_predictions(to_inference, df_probs)
        
        logger.info('Model prediction results stored in database')
        
        return df_probs
    
    def _results(self, probabilities):
        return pd.DataFrame({'Preds_Prob': probabilities})
    
    def _capture_inputs_and_predictions(self, inputs, predictions):
        input_df = pd.DataFrame(
            inputs['dataframe_split']['data'],
            columns=inputs['dataframe_split']['columns']
        )
        
        input_df['Preds_Prob'] = predictions
        
        self._store_predictions(input_df)
        
    def _store_predictions(self, input_df):
        input_df.to_sql('predictions', conn, if_exists='append', index=False)
        conn.commit()
        conn.close()
        
    def test_endpoint_connection(self):
        try:
            
            df = pd.DataFrame(
                [
                    [0.0, 0.88551908, 43.0, 0.0, 0.177512717, 5700.0, 4.0, 0.0, 0.0, 0.0, 0.0], 
                    [1.0, 0.463295269, 57.0, 0.0, 0.527236928, 9141.0, 15.0, 0.0, 4.0, 0.0, 2.0], 
                    [2.0, 0.043275036, 59.0, 0.0, 0.687647522, 5083.0, 12.0, 0.0, 1.0, 0.0, 2.0], 
                    [3.0, 0.280308229, 38.0, 1.0, 0.925960637, 3200.0, 7.0, 0.0, 2.0, 0.0, 0.0], 
                    [4.0, 0.9999999, 27.0, 0.0, 0.019917227, 3865.0, 4.0, 0.0, 0.0, 0.0, 1.0]
                ],
                columns=[
                    'Unnamed: 0',
                    'TaxaDeUtilizacaoDeLinhasNaoGarantidas',
                    'Idade',
                    'NumeroDeVezes30-59DiasAtrasoNaoPior',
                    'TaxaDeEndividamento',
                    'RendaMensal',
                    'NumeroDeLinhasDeCreditoEEmprestimosAbertos',
                    'NumeroDeVezes90DiasAtraso',
                    'NumeroDeEmprestimosOuLinhasImobiliarias',
                    'NumeroDeVezes60-89DiasAtrasoNaoPior',
                    'NumeroDeDependentes'
                ]
            )
            
            example_data = {
                'dataframe_split': {
                    'columns': df.columns.tolist(),
                    'data': df.replace(np.nan, None).values.tolist()
                }
            }

            # Enviar a solicitação POST com os dados JSON
            response = requests.post(
                url=self.endpoint,
                json=example_data,
                headers={'Content-Type': 'application/json'}
            )

            logger.info(f'Response status code: {response.status_code}')
            logger.info(f'Response content: {response.text}')
            
            # Verificar se a solicitação foi bem-sucedida (código 200)
            response.raise_for_status()
        except Exception as e:
            logger.error(f'Error connecting to endpoint: {e}')
            raise e
        
    