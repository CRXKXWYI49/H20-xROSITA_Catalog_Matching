import uuid
import pandas as pd

from exceptions import DataFrameLengthMismatchError


class CatalogStandardizer:

    """
    Takes three arrays of equal length for the right ascension and declination
    of the 
    """
    
    def __init__(
            df_RA: pd.DataFrame, 
            df_DEC: pd.DataFrame, 
            df_RA_err: pd.DataFrame,
            df_DEC_err: pd.DataFrame,            
        ) -> pd.Dataframe:
        
        if(
            len(df_RA) != 
            len(df_DEC) != 
            len(df_RA_err) != 
            len(df_DEC_err)
        ):
            raise DataFrameLengthMismatchError(
                f"Length of input dataframes do not match"
            )

        length = len(df_RA)
        uuids = [uuid.uuid4() for _ in range(length)]
        df_UUID = pd.DataFrame(uuids, columns=['UUID'])
        
        df = pd.concat([
            df_UUID, 
            df_RA, 
            df_RA_err, 
            df_DEC, 
            df_DEC_err
        ], axis=1)

        return df
    
    
    