from typing import Any, Dict
import numpy as np
import yaml 

class Parameters:

    _number_of_cities:int
    _immunity : int= 5
    _base_immunity: int
    _start_population_size: int = 80
    _population_size: int
    _restriction_inversion: float = 3
    _expexted_number_of_iterations_per_minute: int
    _expexted_number_of_iterations_per_minute_second_core: int
    _population_size_second_core: int
    _pheremone_offspring: int

    def __init__(self) -> None:
        self._population_size = self._start_population_size
        pass

    def set_number_of_cities(self, number_of_cities:int):
        self._number_of_cities = number_of_cities
        if number_of_cities <= 50:
            self._expexted_number_of_iterations_per_minute = 1400
            self._expexted_number_of_iterations_per_minute_second_core = 1500
            self._population_size_second_core = 10
            self._base_immunity = 5
            self._immunity = 5
            self._start_population_size = 80
            self._population_size = self._start_population_size
            self._pheremone_offspring = 15
        elif number_of_cities <= 100:
            self._expexted_number_of_iterations_per_minute = 600
            self._expexted_number_of_iterations_per_minute_second_core = 275
            self._population_size_second_core = 10
            self._base_immunity = 5
            self._immunity = 5
            self._start_population_size = 80
            self._population_size = self._start_population_size
            self._pheremone_offspring = 15
        elif number_of_cities <= 200:
            #index tsp: 116   best: 36874.07142825972  
            self._expexted_number_of_iterations_per_minute = 120
            #Second core index: 96    best: 36865.00479418358
            self._expexted_number_of_iterations_per_minute_second_core = 100
            self._population_size_second_core = 10
            self._base_immunity = 5
            self._immunity = 5
            self._start_population_size = 80
            self._population_size = self._start_population_size
            self._pheremone_offspring = 15
        elif number_of_cities <= 500:
            #index tsp: 107   best: 138310.08376986254 
            self._expexted_number_of_iterations_per_minute = 100
            #Second core index: 86    best: 133925.8788809998
            self._expexted_number_of_iterations_per_minute_second_core = 80
            self._population_size_second_core = 3
            self._base_immunity = 3
            self._immunity = 3
            self._start_population_size = 60
            self._population_size = self._start_population_size
            self._pheremone_offspring = 10
        elif number_of_cities <= 750:
            self._expexted_number_of_iterations_per_minute = 50
            self._expexted_number_of_iterations_per_minute_second_core = 50
            self._population_size_second_core = 3
            self._base_immunity = 3
            self._immunity = 3
            self._start_population_size = 60
            self._population_size = self._start_population_size
            self._pheremone_offspring = 10
        else:
            self._expexted_number_of_iterations_per_minute = 50
            self._expexted_number_of_iterations_per_minute_second_core = 60
            self._population_size_second_core = 3
            self._base_immunity = 3
            self._immunity = 3
            self._start_population_size = 60
            self._population_size = self._start_population_size
            self._pheremone_offspring = 10

    def get_number_of_cities(self) -> int:
        return self._number_of_cities

    def get_population_size(self) -> int:
        return self._population_size
    
    def get_offspring_size(self) -> int:
        return self.get_number_of_1_1_1_offspring() + self.get_number_of_1_1_para_offspring() + self.get_number_of_1_para_1_offspring()\
              + self.get_number_of_1_para_1_offspring() + self.get_number_of_1_para_para_offspring() \
              + self.get_number_of_para_para_para_offspring() + self.get_number_of_OX1_offspring() +\
              + self.get_number_of_POS_offspring() + self.get_number_of_heuristic_offspring()
    
    #number of parents to select for crossover
    #total offspring= paret_pairs*offspring size
    def get_number_of_parent_pairs(self) -> int:
        return 3
    
    def get_k_selection(self) -> int:
        return 5
    
    def get_k_elimination(self) -> int:
        return 30
    
    def get_number_of_random_initialisation(self) -> int:
        return int(self.get_population_size() * 0.3)
    
    def get_number_of_nearest_neighbour_initialisation(self) -> int:
        return int(self.get_population_size() * 0.3)

    def get_p_inherent_common_city_in_common_location(self) -> float:
        return 0.8

    def get_p_inherent_difference_city_in_common_location(self) -> float:
        return 0.6

    def get_p_inherent_common_path(self) -> float:
        return 0.9

    def get_number_of_1_1_1_offspring(self) -> int:
        return 0

    def get_number_of_1_1_para_offspring(self) -> int:
        return 0

    def get_number_of_1_para_1_offspring(self) -> int:
        return 0

    def get_number_of_1_para_para_offspring(self) -> int:
        return 0

    def get_number_of_para_para_para_offspring(self) -> int:
        return 0

    #always generating two at once
    def get_number_of_OX1_offspring(self) -> int:
        return 2

    
    #always generating two at once
    def get_number_of_POS_offspring(self) -> int:
        return 2
    
    #always generating two at once
    def get_number_of_heuristic_offspring(self) -> int:
        return 2
    
    def get_number_of_pheremone_offspring(self) -> int:
        return self._pheremone_offspring

    def get_OX1_constant_subpath_lenght(self) -> int:
        return int(np.ceil(0.15*self._number_of_cities))

    def get_number_of_random_fixed_positions_POS(self) -> int:
        return int(np.ceil(0.15*self._number_of_cities))

    def get_length_DM_subtour(self) -> int:
        return 3

    def get_length_SIM_subtour(self) -> int:
        return 3

    def get_length_IVM_subtour(self) -> int:
        return 3
    
    def maximum_number_of_mutations(self) -> int:
        return 6

    def get_mutation_rate(self) -> float:
        return 0.9

    def get_chance_on_DM_mutation(self) -> float:
        return 1/5

    def get_chance_on_SIM_mutation(self) -> float:
        return 2/5

    def get_chance_on_IVM_mutation(self) -> float:
        return 2/5
    
 
    
    def get_grouping_equality_for_elemintation(self) -> int:
        return int(np.ceil(0.95*self._number_of_cities))

    def get_range_for_nearest_neighbours(self) -> int:
        return int(np.ceil(0.05*self._number_of_cities))

    def get_pheromone_scaling(self) -> int:
        return 1000 * self._number_of_cities

    #more than this amount of equal edges results in penalty
    def get_threshold_for_likeness_penalty(self) -> int:
        return int(np.ceil(0.5*self._number_of_cities))

    # power of scaling =  2 - (non-similar-edges/cities)**a
    def get_scaling_likeness_penalty(self) -> float:
        return 1.5

    def get_weight_distance_probabilities_for_extention_with_pheremones(self) -> float:
        return 0.2

    def get_weight_pheremone_probabilities_for_extention_with_pheremones(self) -> float:
        return 1 - self.get_weight_distance_probabilities_for_extention_with_pheremones()
    
    def get_immunity_top_population(self) -> int:
        return self._immunity
    
    def set_population_size(self, amount: int = 1):
        self._population_size = self._start_population_size + amount


    def set_immunity(self, amount:int):
        self._immunity =  self._base_immunity + amount


    def get_composition_second_core(self) -> np.ndarray:
        random_init = 0
        NN_init = 1
        greedy_init = 1
        pheremone_init = 3
        return np.array([random_init, NN_init, greedy_init, pheremone_init])

    def get_max_converged(self) -> int:
        return 3* self._base_immunity
    
    def get_reset_converged(self) -> int:
        return self._base_immunity
    
    def get_population_size_second_core(self) -> int:
        return self._population_size_second_core
    
    def get_second_core_reset(self) -> int:
        return int(0.1 * self._expexted_number_of_iterations_per_minute_second_core)
    
    def tsp_increase_inversion_mutation_on_converged(self) -> int:
        return int(0.3 * self._expexted_number_of_iterations_per_minute)
    
    def index_to_sync_pheremone_matrix(self) -> int:
        return int(0.5 * self._expexted_number_of_iterations_per_minute)
    
    # def update_pheremone_matrix(self) -> int:
        # return int(0.2 * self._expexted_number_of_iterations_per_minute)
    
    def restriction_on_length_of_inversion_offspring(self) -> float:
        return self._restriction_inversion
    
    def increase_restriction_on_length_of_inversion_offspring(self):
        self._restriction_inversion += 1