from astropy.io   import fits
from pathlib      import Path

import pandas     as pd
import numpy      as np
import matplotlib as mp

import sys

import config_loader as ConfigLoader


class DataLoader:

    def __init__(self, config: ConfigLoader, verbosity = 0):
        self.config = config
        self.verbosity = verbosity


    def get_xROSITA(self) -> pd.DataFrame:
        config = self.config
        verbosity = self.verbosity

        with fits.open(Path(config.XROSITA_PATH).resolve()) as hdul:
            if verbosity == 1: hdul.info()
            data_xROSITA = hdul[1].data
            df_xROSITA = pd.DataFrame(data_xROSITA)
            df_xROSITA = self._convert_dataframe(df_xROSITA)
        
        return df_xROSITA

    
    
    def get_H20(self) -> pd.DataFrame:
        config = self.config
        verbosity = self.verbosity

        with fits.open(Path(config.H20_PATH).resolve()) as hdul:
            if verbosity == 1: hdul.info()
            df_H20 = hdul[1].data
            df_H20 = pd.DataFrame(df_H20)
            df_H20 = self._convert_dataframe(df_H20)
        
        return df_H20
    

    def _convert_dataframe(self, dataframe):

        """
        Method to convert the dataframe to system endian.
        """

        system_endian = sys.byteorder

        def convert_endian(series):
            """
            A Private method to convert the dataframe to system endian. 
            Checks system endian and converts dataframe if necessary.
            """
            if ((series.dtype.kind in 'iu') and
                (series.dtype.byteorder) not in 
                ('=', system_endian)
            ):
                return series.astype(
                    series.dtype.newbyteorder(system_endian)
                )
            elif ((series.dtype.kind == 'f') and
                (series.dtype.byteorder) not in 
                ('=', system_endian)
            ):
                return series.astype(
                    series.dtype.newbyteorder(system_endian)
                )
            return series
        
        return dataframe.apply(convert_endian)
