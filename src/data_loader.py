from astropy.io   import fits
from pathlib      import Path

import pandas     as pd
import numpy      as np
import matplotlib as mp

import sys

import config_loader as ConfigLoader


class DataLoader:

    """
    
    """

    def __init__(self, config: ConfigLoader, verbosity = 0):
        self.config = config
        self.verbosity = verbosity


    def get_dataframe(self, path: str) -> pd.DataFrame:
        config = self.config
        verbosity = self.verbosity

        with fits.open(Path(path).resolve()) as hdul:
            if verbosity == 1: hdul.info()
            data = hdul[1].data
            df = pd.DataFrame(data)
            df = self._convert_system_endian(df)
            df = self._decode_byte_strings(df)

        return df


    def _decode_byte_strings(self, df: pd.DataFrame):
        for column in df.select_dtypes(include=['object']):
            df[column] = df[column].apply(
                lambda x: x.decode('utf-8', errors='ignore') 
                if isinstance(x, bytes) else x
            )
        return df


    def _convert_system_endian(self, df):

        """
        Method to convert the df to system endian.
        """

        system_endian = sys.byteorder

        def convert_endian(df):

            """
            A Private method to convert the dataframe to system endian. 
            Checks system endian and converts df if necessary.
            """
            
            if ((df.dtype.kind in 'iu') and
                (df.dtype.byteorder) not in 
                ('=', system_endian)
            ):
                return df.astype(
                    df.dtype.newbyteorder(system_endian)
                )
            elif ((df.dtype.kind == 'f') and
                (df.dtype.byteorder) not in 
                ('=', system_endian)
            ):
                return df.astype(
                    df.dtype.newbyteorder(system_endian)
                )
            return df
        
        return df.apply(convert_endian)
