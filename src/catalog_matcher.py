"""
Maximum-Likeihood Ratio Catalog Matcher
---------------
"""
# Author: Trevin Lee <trevin.lee@protonmail.com>

import pandas as pd
import numpy as np

from typing import Annotated, Callable, List
from numpy.typing import NDArray
from scipy.spatial import KDTree, distance
from multiprocessing import Pool
from sklearn.neighbors import KernelDensity
from scipy.interpolate import interp1d

from config_loader import ConfigLoader



class MLRCatalogMatcher:

    def __init__(
            self, 
            df_inputs: pd.Series,
            df_outputs: pd.Series,
            num_bins: int = 20,
            adj_flux_rad: float = 2.0,
            match_search_rad: float = 5.0,
            background_inner_rad: float = 5.0,
            background_outer_rad: float = 30.0,
        ):
        self.num_bins = num_bins
        self.df_inputs = df_inputs
        self.df_outputs = df_outputs

        self.adj_flux_rad = adj_flux_rad
        self.match_search_rad = match_search_rad
        self.background_inner_rad = background_inner_rad
        self.background_outer_rad = background_outer_rad

        self.__build_datastore()        


    def __build_datastore( self ):

        """
        Constructs the necessary datastructures to facililate
        future computations.
        """

        print("Building Datastore")

        self.__output_uuids = self.df_outputs['uuid'].to_numpy()
        self.__output_coords = self.df_outputs[['ra','dec']].to_numpy()
        self.__output_flux = self.df_outputs['flux'].to_numpy()
        self.__output_pos_err = self.df_inputs['pos_err'].to_numpy()
        self.__input_uuids = self.df_inputs['uuid'].to_numpy()
        self.__input_dict = {
            uuid: index for index, uuid in enumerate(self.__input_uuids)
        }
        self.__input_coords = self.df_inputs[['ra','dec']].to_numpy()
        self.__num_output = len(self.__output_uuids)
        self.KDTree = KDTree(
            self.__output_coords,
            balanced_tree=True, 
            compact_nodes=True
        )

        print("Datastore complete")
    

    # The below functions are used in computing the matches of the
    # catalogs. compute_matches() returns a 2D array of each
    # object and a list of corresponding likelihoods for each 
    # output object


    def compute_matches( self, processes: int = 1 ):

        print("Computing Matches")

        with Pool(processes=processes) as pool:
            self.__real_flux_dists: List[Callable] = pool.map(
                self._real_dist_normalizer_process,
                self.__input_uuids
            )

        print("Completed getting real flux distributions for all inputs \n")
        print(self.__real_flux_dists)

        with Pool(processes=processes) as pool:
            results = pool.map(
                self._compute_matches_process, 
                self.__input_uuids 
            ) 
            pool.close()
            pool.join()

        return results
    

    def _real_dist_normalizer_process(
            self,
            input_uuid: str
        ) -> NDArray:
            nearest_neighbors = self.__get_nearest_neighbors(
                input_uuid, 
                self.background_outer_rad
            )
            real_flux_dist = self.__get_real_flux_dist(nearest_neighbors)

            return real_flux_dist
        

    def _compute_matches_process( 
        self,
        input_obj_uuid: str 
    ) -> Annotated[NDArray, (None, 2)]:    
        likelihoods = self.__match_object(input_obj_uuid)
        matches_arr = np.array([input_obj_uuid, likelihoods])
        return sorted(matches_arr, reverse=True)
    

    def __match_object( self, input_uuid: str ) -> NDArray:

        input_index = self.__input_dict[input_uuid]
        input_coords = self.__input_coords[input_index]
        
        nearest_neighbors = self.__get_nearest_neighbors(
            input_uuid, 
            self.background_outer_rad,
        )

        potential_match_indices = nearest_neighbors[
            (nearest_neighbors['distance'] < self.match_search_rad)
        ].to_numpy()
        annulus_src_fluxes = nearest_neighbors[
            (nearest_neighbors['distance'] > self.background_inner_rad) & 
            (nearest_neighbors['distance'] < self.background_outer_rad)
        ]['flux'].to_numpy()

        real_flux_dist = self.__real_flux_dists[
            self.__input_dict[input_uuid]
        ]
        real_counterpart_frac: float = self.__real_counterpart_frac(
            num_potential_matches=len(potential_match_indices)
        )
        expected_flux_dist: Callable[[float], float] = (
            self.__get_expected_flux_dist(
                real_flux_dist=real_flux_dist,
                real_counterpart_frac=real_counterpart_frac,
            )
        )
        surface_density_dist: Callable[[float], float] = (
            self.__get_surface_density_dist(
                annulus_src_fluxes=annulus_src_fluxes
            )
        )

        matches = []
        for output_obj_index in potential_match_indices:

            output_flux = self.__output_flux[output_obj_index]
            output_pos_err = self.__output_pos_err[output_obj_index]
            distance = distance.euclidean(
                input_coords, 
                self.__output_coords[output_obj_index],
            )

            expected_flux = expected_flux_dist(output_flux)
            surface_density = surface_density_dist(output_flux)
            probability = self.__probability_dist(
                distance=distance,
                output_pos_err=output_pos_err
            )

            likelihood_ratio = self.__likelihood_ratio(
                expected_flux=expected_flux,
                surface_density=surface_density,
                probability=probability,
            )

            matches.append([
                self.__output_uuids[output_obj_index], 
                likelihood_ratio
            ])

        return matches
    

    def __likelihood_ratio( 
            self,
            expected_flux: float,
            surface_density: float,
            probability: float,
        ) -> float:
        
        """
        The probability_dist is a scalar because it is computed by a
        continuous function defined below.
        """
        
        likelihood_ratio = (
            (probability * expected_flux) / 
            surface_density
        )

        return likelihood_ratio
    

    def __probability_dist(
            self, 
            distance: float, 
            output_pos_err: float
    ) -> float:
        
        coefficient = 1 / (2 * np.pi * output_pos_err**2)
        exponent = np.exp( -distance**2 / (2 * output_pos_err**2))
        probability_dist = coefficient * exponent

        return probability_dist
    

    def __get_expected_flux_dist( 
            self, 
            real_flux_dist: Callable[[float],float],
            real_counterpart_frac: float
        ) -> Callable[[float], float]:

        def expected_flux_dist( flux: float ):
            expected_flux = (
                (real_flux_dist(flux) * real_counterpart_frac) / 
                self.real_dist_normalizer
            )
            return expected_flux
    
        return expected_flux_dist
    

    def __real_dist_normalizer( self, flux: str ):
        real_flux_normalized: float = 0.0

        for real_flux_dist in self.__real_flux_dists:
            real_flux_normalizer += real_flux_dist(flux)

        return real_flux_normalizer


    def __get_real_flux_dist(
            self, 
            total_flux_dist: Callable[[float], float],
            surface_density_dist: Callable[[float], float],
        ) -> Callable[[float], float]:

        inner_area = np.pi * (self.match_search_rad**2)
        num_output = self.__num_output

        def real_flux_dist(flux: float) -> float:
            adj_flux_dist = total_flux_dist(flux)
            surface_density = surface_density_dist(flux)

            background_flux_dist = inner_area * num_output * surface_density
            real_flux = adj_flux_dist - background_flux_dist
            return real_flux
        
        return real_flux_dist
    

    def __get_total_flux_dist( 
            self, 
            proximity_src_fluxes: pd.Series 
        ) -> Callable[[float], float]:

        total_flux_dist = self.__interpolate_distribution(
            proximity_src_fluxes
        )

        return total_flux_dist
    

    def __real_counterpart_frac(self, num_potential_matches) -> float:
        return 1 - (num_potential_matches / self.__num_output)


    def __get_surface_density_dist( 
            self, 
            annulus_src_fluxes: pd.Series 
        ) -> Callable[[float], float]:

        inner_rad = self.background_inner_rad
        outer_rad = self.background_outer_rad
        annulus_area = np.pi * (outer_rad**2 - inner_rad**2)

        annulus_flux_dist = self.__interpolate_distribution(
            annulus_src_fluxes, 
            self.num_bins
        )
        
        def surface_density_dist(flux: float) -> float:
            return annulus_flux_dist(float) / annulus_area
    
        return surface_density_dist

     
    # Private utility methods


    def __get_nearest_neighbors(
            self,
            uuid: str, 
            search_radius: float
        ) -> pd.Series:

        input_obj_index = self.__input_dict[uuid]
        input_coord = self.__input_coords[input_obj_index]

        relevant_obj_indices = self.KDTree.query_ball_point(
            input_coord, 
            search_radius
        ) # Query the KDTree for a list of indices

        nearest_neighbors = pd.DataFrame(
            columns=[
                'distance', 
                'flux', 
                'pos_error',
                'uuid', 
                'obj_index'
            ]
        )

        print(relevant_obj_indices)

        for df_index, obj_index in enumerate(relevant_obj_indices):
            nearest_neighbors[df_index] = [
                distance.euclidean(
                    input_coord,
                    self.__output_coords[obj_index]
                ),
                self.__output_flux[obj_index],
                self.__output_pos_err[obj_index],
                self.__output_uuids[obj_index],
                obj_index,
            ] # Build a dataframe of required objects and their parameters

        return nearest_neighbors
    
    
    def __interpolate_distribution(values, interpolation_bins, kind='linear'):

        """
        Interpolate the distribution of values into N bins and return an interpolator function.

        Parameters:
        values (array-like): The input values to be binned and interpolated.
        interpolation_bins (int): The number of bins to use for the interpolation.
        kind (str): The kind of interpolation to use ('linear', 'cubic', etc.).

        Returns:
        interpolator (function): A function that interpolates the distribution.
        """

        values = np.array(values)
        
        min_val = np.min(values)
        max_val = np.max(values)
        bin_edges = np.linspace(min_val, max_val, interpolation_bins + 1)
        bin_indices = np.digitize(values, bin_edges) - 1
        
        bin_indices[bin_indices == interpolation_bins] = interpolation_bins - 1
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_counts = np.bincount(bin_indices, minlength=interpolation_bins)
        probabilities = bin_counts / np.sum(bin_counts)
        
        interpolator = interp1d(bin_centers, probabilities, kind=kind, fill_value="extrapolate")
        
        return interpolator
        
