import uuid
import pandas as pd
import numpy as np

from exceptions import DataFrameLengthMismatchError


class CatalogStandardizer:


    """
    Takes several arrays of equal length for the right ascension, declination,
    flux, flux error and positional error, and returns a standardized DataFrame.
    """
    

    def __init__(self, df_name, df_ra, df_dec, df_pos_err, df_flux):
        self.df_name = df_name
        self.df_ra = df_ra
        self.df_dec = df_dec
        self.df_pos_err = df_pos_err
        self.df_flux = df_flux

        self.__validate_lengths()


    def __validate_lengths(self):
        dataframes = [self.df_name, self.df_ra, self.df_dec, self.df_pos_err, self.df_flux]
        length = len(dataframes[0])
        for df in dataframes:
            if len(df) != length:
                raise DataFrameLengthMismatchError("Length of all input dataframes do not match")
            

    def standardize(self):
        length = len(self.df_ra)
        uuids = [str(uuid.uuid4()) for _ in range(length)]
        df_UUID = pd.Series(uuids)
        
        df = pd.concat([
            df_UUID, 
            self.df_name,
            self.df_ra, 
            self.df_dec, 
            self.df_pos_err,
            self.df_flux,
        ], axis=1)

        df.columns = ['uuid', 'name', 'ra', 'dec', 'pos_err', 'flux']
        
        return df