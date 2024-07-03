import pandas        as pd
import numpy         as np

from multiprocessing import Pool
from typing          import Annotated, List
from numpy.typing    import NDArray
from numpy           import float16, float64, float32, float8, int32, int8
from scipy.integrate import fixed_quad
from scipy.spatial   import KDTree, distance

from config_loader   import ConfigLoader



class CatalogMatcher:

    def __init__(
            self, 
            df_inputs: pd.DataFrame,
            df_outputs: pd.DataFrame,
            acceptable_flux_range: float32,
            adj_flux_rad: float32,
            potential_match_rad: float32,
            background_inner_rad: float32,
            background_outer_rad: float32,
        ):
        self.df_inputs = df_inputs
        self.df_outputs = df_outputs

        self.acceptable_flux_range = acceptable_flux_range
        self.adj_flux_rad = adj_flux_rad
        self.potential_match_rad = potential_match_rad
        self.background_inner_rad = background_inner_rad
        self.background_outer_rad = background_outer_rad

        self._build_datastore()
        self._build_KDTree()
        

    """
    The below functions construct the necessary datastructures to facililate
    future computations.
    """

    def _build_datastore( self ):
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


    def _build_KDTree( self ):
        self.KDTree = KDTree(
            self.__output_coords,
            balanced_tree=True, 
            compact_nodes=True
        )
    

    """
    The below functions are used in computing the matches of the
    catalogs. compute_matches() returns a 2D array of each
    object and a list of corresponding likelihoods for each 
    output object
    """


    def compute_matches( self, processes: int = 1 ):

        def real_dist_normalizer_process(
            self,
            input_uuid:str
        ) -> NDArray:
            nearest_neighbors = self.__get_nearest_neighbors(
                input_uuid, 
                self.background_outer_rad
            )
            real_flux_dist = self.__real_flux_dist(nearest_neighbors)

            return real_flux_dist
        
        def compute_matches_process( 
            self, 
            input_obj_uuid: str 
        ) -> Annotated[NDArray, (None, 2)]:    
            likelihoods = self.__match_object(input_obj_uuid)
            matches_arr = np.array([input_obj_uuid, likelihoods])

            return sorted(matches_arr, reverse=True)

        with Pool(processes=processes) as pool:
            self.real_flux_dists = pool.map(
                real_dist_normalizer_process(),
                self.__input_uuids
            )

        self.real_dist_normalizer = np.sum(self.real_flux_dists)

        with Pool(processes=processes) as pool:
            results = pool.map(
                compute_matches_process(), 
                self.__input_uuids 
            ) 
            pool.close()
            pool.join()

        return results
    

    def __match_object( self, input_uuid: str ) -> pd.DataFrame:
        
        nearest_neighbors = self.__get_nearest_neighbors(
            input_uuid, 
            self.background_outer_rad,
        )

        num_outputs = self.__num_output

        "Computing input specific parameters"
        potential_match_indices = nearest_neighbors[
            (nearest_neighbors['distance'] < self.potential_match_rad)
        ].to_numpy()
        num_potential_matches = len(potential_match_indices)

        real_flux_dist = self.__real_flux_dist(nearest_neighbors)
        real_counterpart_frac = self.__real_counterpart_frac(
            num_potential_matches
        )
        expected_flux_dist = self.__expected_flux_dist(
            real_flux_dist,
            real_counterpart_frac,
        )
        surface_density = self.__surface_density(nearest_neighbors)

        matches = []
        
        for output_obj in nearest_neighbors:
            "Computing output specfic parameters"
            distance = output_obj['distance']
            output_pos_err = output_obj['pos_error']
            probability_dist = self.__probability_dist(
                distance, 
                output_pos_err
            )

            match = self.__likelihood_ratio(
                probability_dist,
                expected_flux_dist,
                surface_density,
                real_flux_dist
            )
            matches.append(match)

        return matches
    

    def __likelihood_ratio( 
            self,
            probability_dist: float32,
            expected_flux_dist: NDArray,
            surface_density: NDArray
        ) -> float32:
        """
        The probability_dist is a scalar because it is computed by a
        continuous function defined below.
        """
        likelihood_ratio = (
            (probability_dist * expected_flux_dist) / 
            surface_density
        )

        return likelihood_ratio
    

    def __probability_dist(
            self, 
            distance: float32, 
            output_pos_err: float32
    ) -> float32:
        coefficient = 1 / (2 * np.pi * output_pos_err**2)
        exponent = np.exp( -distance**2 / (2 * output_pos_err**2))
        probability_dist = coefficient * exponent

        return probability_dist

    

    def __total_flux_dist( self, nearest_neighbors: pd.DataFrame ) -> NDArray:
        proximity_src_indices = nearest_neighbors[
            (nearest_neighbors['distance'] < self.adj_flux_rad)
        ]['flux'].to_numpy()
        total_flux_dist = self.__build_distribution(proximity_src_indices)

        return total_flux_dist


    def __real_flux_dist(
            self, 
            nearest_neighbors: pd.DataFrame,
        ) -> NDArray:

        inner_area = np.pi * (self.potential_match_rad**2)
        num_output = self.__num_output

        adj_flux_dist = self.__total_flux_dist(nearest_neighbors)
        surface_density = self.__surface_density(nearest_neighbors)

        background_flux_dist = inner_area * num_output * surface_density
        real_flux_dist = adj_flux_dist - background_flux_dist

        return real_flux_dist
    

    def __real_counterpart_frac(self, num_potential_matches) -> float32:
        return 1 - (self.__num_output / num_potential_matches)
    

    def __expected_flux_dist( 
            self, 
            real_flux_dist: NDArray,
            real_counterpart_frac: float32
        ) -> NDArray:

        expected_flux_dist = (
            (real_flux_dist * real_counterpart_frac) / 
            self.real_dist_normalizer
        )
        
        return expected_flux_dist


    def __surface_density( self, nearest_neighbors: pd.DataFrame ) -> NDArray:
        inner_rad = self.background_inner_rad
        outer_rad = self.background_outer_rad
        annulus_area = np.pi * (outer_rad**2 - inner_rad**2)

        annulus_src_indices = nearest_neighbors[
            (nearest_neighbors['distance'] > self.background_inner_rad) & 
            (nearest_neighbors['distance'] < self.background_outer_rad)
        ]['flux'].to_numpy()

        annulus_flux_dist = self.__build_distribution(annulus_src_indices)
        surface_density = annulus_flux_dist / annulus_area
    
        return surface_density


    
    "The below functions are private utility methods"

    def __get_nearest_neighbors(
            self,
            uuid: str, 
            search_radius: float32
        ) -> pd.DataFrame:

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
            ] # Build a dataframe of required objects and thei parameters

        return nearest_neighbors

    

    def __sum_awkward_dists( distributions: List[NDArray]) -> NDArray:
        """
        A function that takes a python list of numpy arrays of different
        lengths and sums over the numpy arrays.
        """
        max_length = max(len(dist) for dist in distributions)
        padded_distributions = [
            np.pad(dist, (0, max_length - len(dist)), 'constant') 
            for dist in distributions
        ]
        summed_distribution = np.sum(padded_distributions, axis=0)

        return summed_distribution


    def __build_distribution( values: NDArray ) -> NDArray:
        """
        returns a distribution of values as a 1D array containing the
        number of objects whose values fall into discrete bins of
        width 1 which start at 0. The index of the array represents
        the minimum value of the bin.
        """
        max_value = int(np.ceil(np.max(values)))
        bins = np.arange(0, max_value + 1, 1)
        bin_indices = np.digitize(values, bins, right=False)
        bin_counts = np.bincount(bin_indices, minlength=len(bins) + 1)
        distribution = np.vstack((bins, bin_counts[1:len(bins)+1])).T
        
        return distribution
    
