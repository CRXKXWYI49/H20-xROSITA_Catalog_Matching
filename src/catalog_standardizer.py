import uuid
import pandas as pd

from exceptions import DataFrameLengthMismatchError


class CatalogStandardizer:

    """
    Takes several arrays of equal length for the right ascension, declination,
    flux, flux error and positional error.
    """
    
    def __init__(
            df_ra: pd.DataFrame, 
            df_dec: pd.DataFrame, 
            df_pos_err: pd.DataFrame,
            df_flux: pd.DataFrame,
            df_flux_err: pd.DataFrame            
        ) -> pd.Dataframe:

        dataframes = [df_ra, df_dec, df_pos_err, df_flux, df_flux_err]

        for df in dataframes:
            if df.shape[1] != 1: 
                raise ValueError(
                    f"DataFrame {df.columns[0]} is not 1-dimensional."
                )
        
        for df in dataframes:
            if(len(df) != dataframes[0]):
                raise DataFrameLengthMismatchError(
                    "Length of all input dataframes do not match"
                )  

        length = len(df_ra)
        uuids = [str(uuid.uuid4()) for _ in range(length)]
        df_UUID = pd.DataFrame(uuids)
        
        df = pd.concat([
            df_UUID, 
            df_ra, 
            df_dec, 
            df_pos_err,
            df_flux,
            df_flux_err
        ], axis=1)

        df.columns = (
            ['uuid'] + 
            ['ra'] +
            ['dec'] + 
            ['pos_err'] +
            ['flux'] +
            ['flux_err'] 
        )

        return df
    
    
    