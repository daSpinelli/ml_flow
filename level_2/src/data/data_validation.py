import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import pandera
import structlog

from pandera import Column, Check, DataFrameSchema
from utils.utils import load_config

logger = structlog.get_logger()

class DataValidation:
    def __init__(self) -> None:
        self.columns_to_use = load_config().get('columns_to_use')

    def check_shape_data(self, df: pd.DataFrame) -> bool:
        try:
            logger.info('Validation started')
            df.columns = self.columns_to_use
            return True
        except Exception as e:
            logger.error(f'Validation failed: {e}')
            return False
        
    def check_columns(self, df: pd.DataFrame) -> bool:
        schema = DataFrameSchema(
            {
                'target': Column(int, Check.isin([0, 1]), Check(lambda x: x > 0), coerce=True),
                'TaxaDeUtilizacaoDeLinhasNaoGarantidas': Column(float, nullable=True),
                'Idade': Column(int, nullable=True),
                'NumeroDeVezes30-59DiasAtrasoNaoPior': Column(int, nullable=True),
                'TaxaDeEndividamento': Column(float, nullable=True),
                'RendaMensal': Column(float, nullable=True),
                'NumeroDeLinhasDeCreditoEEmprestimosAbertos': Column(int, nullable=True),
                'NumeroDeVezes90DiasAtraso': Column(int, nullable=True),
                'NumeroDeEmprestimosOuLinhasImobiliarias': Column(int, nullable=True),
                'NumeroDeVezes60-89DiasAtrasoNaoPior': Column(int, nullable=True),
                'NumeroDeDependentes': Column(float, nullable=True),
            }
        )
        
        try:
            schema.validate(df)
            logger.info('Validation passed')
            return True
        except pandera.errors.SchemaErrors as e:
            logger.error(f'Validation failed: {e}')
            pandera.display(e.failure_cases)
            return False
        
    def run(self, df: pd.DataFrame) -> bool:
        if self.check_shape_data(df):
            if self.check_columns(df):
                logger.info(f'Validation successeful')
                return True
            else:
                return False
        else:
            return False