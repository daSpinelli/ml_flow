import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import structlog

from utils.utils import load_config

logger = structlog.get_logger()

class DataLoad:
    def __init__(self) -> None:
        pass

    def run(self, data_file: str, index_col: int = None) -> pd.DataFrame:
        """
        Reads a CSV file into a Pandas DataFrame.

        Parameters
        ----------
        data_file : str
            Path to the CSV file to read.
        index_col : int, optional
            Column to use as the index of the DataFrame.

        Returns
        -------
        pd.DataFrame
            The DataFrame containing the data from the CSV file.
        """
        logger.info('Reading data from CSV file...')
        
        try:
            df = pd.read_csv(data_file, index_col=index_col)
            logger.info('Data read successfully.')
            return df
        except FileNotFoundError:
            logger.error(f'File {data_file} not found.')
        except Exception as e:
            logger.error(f'Error reading data from {data_file}: {e}')
            raise e