import math

import pandas        as pd
import numpy         as np

from typing          import Any
from numpy           import float64, float32, float8, int32, int8
from scipy.integrate import fixed_quad
from scipy.spatial   import KDTree


class CatalogMatcher:

    def __init__(
            self, 
            df_input: pd.DataFrame,
            df_output: pd.DataFrame
        ):
        self.df_input = df_input
        self.df_output = df_output
        self.matches = []

        self._build_database(df_input, df_output)
        self._build_KDTree
        

    def _build_database(self, df_input: pd.DataFrame, df_output: pd.DataFrame):
        self.DF_INPUT_BY_UUID = df_input.sort_values(by=df_output.columns[0])
        self.DF_INPUT_BY_UUID.reset_index(drop=True)
        self.DF_INPUT_BY_RA = df_input.sort_values(by=df_input.columns[1])
        self.DF_INPUT_BY_RA.reset_index(drop=True)
        self.DF_INPUT_BY_DEC = df_input.sort_values(by=df_input.columns[2])
        self.DF_INPUT_BY_DEC.reset_index(drop=True)
        self.DF_INPUT_BY_FLUX = df_input.sort_values(by=df_input.columns[4])
        self.DF_INPUT_BY_FLUX.reset_index(drop=True)

        self.DF_OUTPUT_BY_UUID = df_output.sort_values(by=df_output.columns[0])
        self.DF_OUTPUT_BY_UUID.reset_index(drop=True)
        self.DF_OUTPUT_BY_RA = df_output.sort_values(by=df_output.columns[1])
        self.DF_OUTPUT_BY_RA.reset_index(drop=True)
        self.DF_OUTPUT_BY_DEC = df_output.sort_values(by=df_output.columns[2])
        self.DF_OUTPUT_BY_DEC.reset_index(drop=True)
        self.DF_OUTPUT_BY_FLUX = df_output.sort_values(by=df_output.columns[4])
        self.DF_OUTPUT_BY_FLUX.reset_index(drop=True)


    def _build_KDTree(self):
        coordinates = self.DF_INPUT_BY_UUID.loc['ra','dec'].to_numpy()
        self.tree = KDTree(coordinates)


    def compute_matches(self):

        pass


    def compute_lr(
            self, 
            target_object: str, 
            annulus_in_rad: float64,
            annulus_out_rad: float64
        ) -> float64:

        def probability_dist(distance:float64, sigma_pos: float64) -> float64:
            coefficient = 1 / (2 * np.pi * sigma_pos**2)
            exponent = np.exp(-distance**2 / (2 * sigma_pos**2))
            return coefficient * exponent
        
        def surface_density(
                df_input: pd.DataFrame, 
                acceptable_flux_range: float8
            ) -> float64: 
            pass

        def real_flux_dist(
                total_flux: float64, 
                num_sources: int32, 
                surface_density: float64
            ) -> float64:
            background =  np.pi * (annulus_in_rad**2) * num_sources * surface_density
            real_flux_dist = total_flux - background
            return real_flux_dist
        
        def expected_flux_dist(self, range: float32):

            pass

        lr = probability_dist * expected_flux_dist / surface_density

        return lr


    def compute_distance(
            self, 
            input_obj: str, 
            output_obj: str
        ) -> float64:

        input_RA = self.binary_search(input_obj, 'uuid', 'RA', is_input=True)
        input_DEC = self.binary_search(input_obj, 'uuid', 'DEC', is_input=True)
        output_RA = self.binary_search(output_obj, 'uuid', 'RA', is_input=False)
        output_DEC = self.binary_search(output_obj, 'uuid', 'DEC', is_input=False)

        distance = math.sqrt(
            (input_RA - output_RA)**2 + (input_DEC - output_DEC)**2
        )
        return distance
    
        
    def binary_search(
            self, 
            search_key: Any, 
            search_key_type: str, 
            target_col: str,
            is_input: bool
        ):
        """
        Perform a binary search on a DataFrame where the first column is 
        sorted. Returns the value in column_index of the same row.
        """

        if (is_input):
            match search_key_type:
                case 'uuid':
                    df = self.DF_INPUT_BY_UUID
                case 'ra':
                    df = self.DF_INPUT_BY_RA
                case 'dec':
                    df = self.DF_INPUT_BY_DEC
                case 'flux':
                    df = self.DF_INPUT_BY_FLUX
                case _ :
                    raise ValueError(
                        f"{search_key_type} column does not exist in "
                        "input data"
                    )
        else:
            match search_key_type:
                case 'uuid':
                    df = self.DF_OUTPUT_BY_UUID
                case 'ra':
                    df = self.DF_OUTPUT_BY_RA
                case 'dec':
                    df = self.DF_OUTPUT_BY_DEC
                case 'flux':
                    df = self.DF_OUTPUT_BY_FLUX
                case _ :
                    raise ValueError(
                        f"{search_key_type} column does not exist in"
                        "output data"
                    )

        low, high = 0, len(df) - 1

        while low <= high:
            mid = (low + high) // 2
            mid_value = df.iloc[mid, 0]

            if mid_value == search_key:
                return df.iloc[mid, target_col]
            elif mid_value < search_key:
                low = mid + 1
            else:
                high = mid - 1

        raise ValueError(
            f"Search key {search_key} does not exist in " 
            f"{"input" if(is_input) else "output"} dataframe"
        )

