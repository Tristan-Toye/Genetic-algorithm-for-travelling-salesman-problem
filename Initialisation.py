
import time
import numpy as np

from EvolutionaryFunctions import EvolutionaryFunctions
from Exceptions import InitialisationError, NormalisationError, PopulationSizeException, TravellingSalesMatrixException, CyclicNotationException
from Individual import Individual
from Parameters import Parameters
from Population import Population
from Debug import DEBUG
from numba import jit, int16, float64, boolean, int8, objmode

from local_search import improve, optimise

@jit(float64[:](float64[:]),nopython = True)
def transform_distances_to_probabilities(distances: np.ndarray):
   
    norm = np.sum(distances)
    if norm != 0 or norm != 1:
        for index, value in enumerate(distances):
            distances[index] = value/norm
        # if DEBUG and not np.isclose(np.sum(distances),nom):
        #     raise NormalisationError(f"The sum of the probabilities are not equal to 1.\n\t sum: {np.sum(distances)}!")
    return distances

@jit(float64[:](float64[:]), nopython = True)
def reset(array: np.ndarray):
    for index, value in np.ndenumerate(array):
        array[index[0]] = 0
    return array

@jit(boolean(int16[:], boolean[:], float64[:,:], float64[:,:], int16[:], float64[:], float64[:], int16, float64, float64),nopython = True)
def pheremone(candidate_solution: np.ndarray, not_already_used: np.ndarray, distance_matrix: np.ndarray, pheremone_matrix: np.ndarray, all_cities: np.ndarray,
              distances: np.ndarray, pheremones: np.ndarray,
               number_of_cities: int, weight_pheremones: float, weight_distances : float ) -> bool:
        
    begin_city = np.random.choice(all_cities,1)[0]
    last_city = begin_city
    not_already_used[begin_city] = False

    
    reset_time = 0
    distance_lookup_time = 0
    pheremone_lookup_time = 0
    transform_distances = 0
    transform_pheremones = 0
    multi = 0
    select = 0
    counter = 0
    while counter < number_of_cities - 1:
        # with objmode(start_time = 'f8'):
        #     start_time = time.time()
        reset(distances)
        reset(pheremones)
        # with objmode(reset_time = 'f8'):
        #     reset_time += time.time() - start_time

        # with objmode(start_time = 'f8'):
        #     start_time = time.time()   
        for index, value in np.ndenumerate(distance_matrix[last_city]):
            if index[0] != last_city and not_already_used[index[0]] and ~np.isinf(distance_matrix[last_city][index[0]]):
                distances[index[0]] = 1/value
        # with objmode(distance_lookup_time = 'f8'):
        #     distance_lookup_time += time.time() - start_time

        if not np.count_nonzero(distances):
            return False
        # with objmode(start_time = 'f8'):
        #     start_time = time.time() 
        for index, value in np.ndenumerate(pheremone_matrix[last_city]):
            if not_already_used[index[0]]:
                pheremones[index[0]] = value
        # with objmode(pheremone_lookup_time = 'f8'):
        #     pheremone_lookup_time += time.time() - start_time
        # with objmode(start_time = 'f8'):
        #     start_time = time.time() 
        transform_distances_to_probabilities(distances)
        # with objmode(transform_distances = 'f8'):
        #     transform_distances += time.time() - start_time
        if np.count_nonzero(pheremones):
            # with objmode(start_time = 'f8'):
            #     start_time = time.time() 
            transform_distances_to_probabilities(pheremones)
            # with objmode(transform_pheremones = 'f8'):
            #     transform_pheremones += time.time() - start_time
            # with objmode(start_time = 'f8'):
            #     start_time = time.time() 
            probabilities = weight_distances * distances + weight_pheremones * pheremones
            # with objmode(multi = 'f8'):
            #     multi += time.time() - start_time
        else:
            probabilities = distances

        # does not work in numba, no support for p= 
        #candidate_solution[last_city] = np.random.choice(all_cities, 1, p= probabilities)
        # work around:
        # with objmode(start_time = 'f8'):
        #     start_time = time.time()
        candidate_solution[last_city] = all_cities[np.searchsorted(np.cumsum(probabilities), np.random.random(), side= "right")]
        # with objmode(select = 'f8'):
        #     select += time.time() - start_time
        last_city = candidate_solution[last_city]
        not_already_used[last_city] = False
        counter +=1
    # with objmode():
    #     print(f"reset: {reset_time}\tdis_look: {distance_lookup_time}\tphere_look: {pheremone_lookup_time}\ttrans_dist: {transform_distances}\ttrans_phere: {transform_pheremones}\tmulti: {multi}\t select: {select}")
   

    #distance_matrix[last_city,begin_city] is not supported in numba
    if np.isinf(distance_matrix[last_city][begin_city]): return False
        
    candidate_solution[last_city] = begin_city

    # if DEBUG and np.count_nonzero(candidate_solution == -1) != 0:
    #     raise InitialisationError("There are still -1 in the path")
    # if DEBUG and np.unique(candidate_solution).size != candidate_solution.size:
    #     raise InitialisationError("There are duplicates in the path")
    
    return True







class Initialisation():


    _NUMBER_OF_CITIES: int = -1
    _distance_matrix: np.ndarray
    _all_cities:np.ndarray
    _ordered_arg_distance_matrix: np.ndarray
    _parameters: Parameters
    _pheromone_matrix: np.ndarray
    

    def __init__(self, parameters: Parameters, distanceMatrix: np.ndarray, ordered_arg_distance_matrix: np.ndarray, pheromone_matrix: np.ndarray) -> None:
        
        self._distance_matrix = distanceMatrix   
        self._ordered_arg_distance_matrix = ordered_arg_distance_matrix
        shape = distanceMatrix.shape
        if len(shape) != 2 or shape[0] != shape[1] : raise TravellingSalesMatrixException(f"The matrix is invalid is invalid, input shape: {shape}")

        self._NUMBER_OF_CITIES = distanceMatrix.shape[0]
        self._all_cities = np.arange(self._NUMBER_OF_CITIES, dtype= np.int16)

        if parameters.get_population_size() <= 0 : raise PopulationSizeException(f"The population size is invalid, input:{parameters.get_population_size()}")
    
        self._pheromone_matrix = pheromone_matrix
        self._parameters = parameters
      
    def get_pheremone_matrix_exposed(self) -> np.ndarray:
        return self._pheromone_matrix
    #to change parameters object after process switch
    def set_new_parameters(self, parameters: Parameters):
        self._parameters = parameters

    def _build_population(self, population: Population, number_of_random_init: int, number_of_nearest_neighbour_init: int, population_size : int) :
        self._extend_the_population_with_random_individuals(population, number_of_individuals_to_generate= number_of_random_init)
        
        if population.has_infinites():
            raise InitialisationError("Initial population has infinite solutions")
        self._extend_the_population_with_nearest_neighbours(population, number_of_individuals_to_generate= number_of_nearest_neighbour_init)
        
        if population.has_infinites():
            raise InitialisationError("Initial population has infinite solutions")
        self._extend_population_with_weighted_prob_individuals(population, 
                                                               number_of_individuals_to_generate= population_size - number_of_random_init - number_of_nearest_neighbour_init)
        
        
        if population.has_infinites():
            raise InitialisationError("Initial population has infinite solutions")
        
        EvolutionaryFunctions.delete_equals(population)

        EvolutionaryFunctions.delete_groupings_list(population.get_population_exposed(), 
                                               self._parameters.get_number_of_cities() - self._parameters.get_grouping_equality_for_elemintation())

        EvolutionaryFunctions.update_pheremone_matrix(self._parameters.get_pheromone_scaling(), self._pheromone_matrix, population.get_population_exposed())
        
        while population.get_current_population_size() != population_size:
            self._extend_population(population, intended_size= population_size)
            
            EvolutionaryFunctions.delete_equals(population)

            EvolutionaryFunctions.delete_groupings_list(population.get_population_exposed(), 
                                               self._parameters.get_number_of_cities() - self._parameters.get_grouping_equality_for_elemintation())
        
    
    

    def get_initial_population(self) -> Population:
        population = Population( self._parameters)
        number_of_random_init = self._parameters.get_number_of_random_initialisation()
        number_of_nearest_neighbour_init = self._parameters.get_number_of_nearest_neighbour_initialisation()

        self._build_population(population,number_of_random_init, number_of_nearest_neighbour_init, self._parameters.get_population_size())

        return population
        
        
    def get_second_core_population(self) -> Population:
        population = Population( self._parameters)
        tmp = self._parameters.get_composition_second_core()

        self._build_population(population,tmp[0], tmp[1], tmp[0] + tmp[1] + tmp[2])
        self._extend_population(population, tmp[3])

        return population
    
    def extend_population(self, population: Population, intended_size: int = 0):
        if not intended_size:
            intended_size = population.get_original_population_size()
        self._extend_population(population, intended_size)
    
    def _extend_population(self, population: Population, intended_size: int):
        
        if intended_size < population.get_current_population_size():
            raise InitialisationError(f"incorrect argument to extend:\n\t intended_size: {intended_size} \n\t current_size: {population.get_current_population_size()}")
        
        pheremone_extend = int((intended_size - population.get_current_population_size())/2)
        NN_extend = intended_size - population.get_current_population_size() - pheremone_extend
        self._extend_using_distances_and_pheremone(  population, number_of_individuals_to_generate = pheremone_extend)
        self._extend_the_population_with_nearest_neighbours(population, number_of_individuals_to_generate= NN_extend)
        EvolutionaryFunctions.delete_equals(population)

        EvolutionaryFunctions.delete_groupings_list(population.get_population_exposed(), 
                                               self._parameters.get_number_of_cities() - self._parameters.get_grouping_equality_for_elemintation())
        while population.get_current_population_size() != intended_size:
            pheremone_extend = (intended_size - population.get_current_population_size())
            
            self._extend_using_distances_and_pheremone(  population, number_of_individuals_to_generate = pheremone_extend)
            EvolutionaryFunctions.delete_equals(population)

            EvolutionaryFunctions.delete_groupings_list(population.get_population_exposed(), 
                                               self._parameters.get_number_of_cities() - self._parameters.get_grouping_equality_for_elemintation())
        
    def generate_pheremone_individuals(self,total_new_individuals: int ) -> np.ndarray :
        tmp = np.empty(total_new_individuals, dtype = object)  
        counter = 0
        while counter < tmp.size:
            individual = self._pheremone()
            if individual is None: 
                continue
            tmp[counter] = individual
            counter +=1
        return tmp
    
    def _pheremone(self) -> Individual:
        
        candidate_solution = np.full(self._NUMBER_OF_CITIES, -1, dtype=np.int16)
        not_already_used = np.full_like(candidate_solution, True, dtype=bool)
        distances = np.zeros(self._NUMBER_OF_CITIES, dtype= np.float64)
        pheremones = np.zeros(self._NUMBER_OF_CITIES, dtype=np.float64)
        
        # start_time = time.time()
        found_solution = pheremone(candidate_solution, not_already_used, self._distance_matrix, self._pheromone_matrix,self._all_cities, distances, pheremones,
                                     self._NUMBER_OF_CITIES, 
                                     weight_pheremones = self._parameters.get_weight_distance_probabilities_for_extention_with_pheremones(),
                                     weight_distances = self._parameters.get_weight_pheremone_probabilities_for_extention_with_pheremones() )
        
        # print(f"overall time in jit function: {time.time() - start_time}")
        if not found_solution:
            
            return None #type: ignore
        
        individual = Individual(self._NUMBER_OF_CITIES, self._distance_matrix)
        individual.use_cyclic_notation(candidate_solution, check_reverse=True)

        if DEBUG:
            try:
                for city in individual():
                    continue
            except CyclicNotationException as e:
                raise InitialisationError("The candidate solution is not a valid cycle")
        return individual
    
    def extend_using_distances_and_pheremone(self, out: np.ndarray):
        
        index = 0
        non_solution_counter = 0
  
        while index < out.size:
          
            individual = self._pheremone()
            if individual is None:
                non_solution_counter +=1
                
                continue
            out[index] = individual
            index +=1
        
    def _extend_using_distances_and_pheremone(self, population: Population, number_of_individuals_to_generate: int):
        
        counter = 0
        while counter < number_of_individuals_to_generate:
            individual = self._pheremone()
            if individual is None:
                continue
            population.add_individual(individual)
            counter +=1
    
    def _extend_the_population_with_nearest_neighbours(self, population: Population, number_of_individuals_to_generate: int):

        counter = 0
        while counter < number_of_individuals_to_generate:
            candidate_solution = np.full(self._NUMBER_OF_CITIES, -1, dtype=np.int16)
            not_already_used = np.full_like(candidate_solution, True, dtype=bool)
            begin_city = np.random.choice(self._all_cities) #rest of process is deterministic, to avoid same candidates we need to start in different position
            not_already_used[begin_city] = False
            invalid_solution = False
            last_city = begin_city
            for _ in range(self._NUMBER_OF_CITIES - 1):
                for candidate_city in self._ordered_arg_distance_matrix[last_city][1:]: #first element is back to itself as d(t1, t1) == 0
                    if not_already_used[candidate_city] and not np.isinf(self._distance_matrix[last_city, candidate_city]):
                        candidate_solution[last_city] = candidate_city 
                        break
                
                if candidate_solution[last_city] == -1:
                    invalid_solution = True
                    break
                else:
                    last_city = candidate_solution[last_city]
                    not_already_used[last_city] = False
                
            if invalid_solution or np.isinf(self._distance_matrix[last_city, begin_city]): continue

            candidate_solution[last_city] = begin_city

            if DEBUG and np.count_nonzero(candidate_solution == -1) != 0:
                raise InitialisationError("There are still -1 in the path")
            if DEBUG and np.unique(candidate_solution).size != candidate_solution.size:
                raise InitialisationError("There are duplicates in the path")
            
            individual = Individual(self._NUMBER_OF_CITIES, self._distance_matrix)
            individual.use_cyclic_notation(candidate_solution, check_reverse=True)

            if DEBUG:
                try:
                    for city in individual():
                        continue
                except CyclicNotationException as e:
                    raise InitialisationError("The candidate solution is not a valid cycle")
           
            population.add_individual(individual)
            counter +=1


    def _transform_distances_to_probabilities(self, distances: np.ndarray):

        transform_distances_to_probabilities(distances)


    
    def _extend_population_with_weighted_prob_individuals(self,  population: Population,  number_of_individuals_to_generate: int ):

        counter = 0
        while counter < number_of_individuals_to_generate:
            candidate_solution = np.full(self._NUMBER_OF_CITIES, -1, dtype=np.int16)
            not_already_used = np.full_like(candidate_solution, True, dtype=bool)
            not_already_used[0] = False
            last_city = 0
            invalid_solution = False

            for _ in range(self._NUMBER_OF_CITIES - 1):
            
                nearest_neighbours = self._ordered_arg_distance_matrix[last_city][:self._parameters.get_range_for_nearest_neighbours()]
                not_yet_allocated_nearest_neighbours = nearest_neighbours[not_already_used[nearest_neighbours]]

                if not_yet_allocated_nearest_neighbours.size:
                    distances = self._distance_matrix[last_city][not_yet_allocated_nearest_neighbours]
                    self._transform_distances_to_probabilities(distances)
        
                    candidate_solution[last_city] = np.random.choice(not_yet_allocated_nearest_neighbours,p= distances)
                else:
                    distances = np.zeros_like(self._distance_matrix[last_city])
                    np.multiply(self._distance_matrix[last_city], not_already_used, out = distances, where = ~np.isinf(self._distance_matrix[last_city]))
                    if not np.count_nonzero(distances):
                        invalid_solution = True
                        break
                    self._transform_distances_to_probabilities(distances)
                    candidate_solution[last_city] = np.random.choice(self._all_cities,p= distances)
                last_city = candidate_solution[last_city]
                not_already_used[last_city] = False
            
            if invalid_solution or np.isinf(self._distance_matrix[last_city, 0]): continue

            candidate_solution[last_city] =0

            if DEBUG and np.count_nonzero(candidate_solution == -1) != 0:
                raise InitialisationError("There are still -1 in the path")
            if DEBUG and np.unique(candidate_solution).size != candidate_solution.size:
                raise InitialisationError("There are duplicates in the path")
            
            individual = Individual(self._NUMBER_OF_CITIES, self._distance_matrix)
            individual.use_cyclic_notation(candidate_solution, check_reverse=True)

            if DEBUG:
                try:
                    for city in individual():
                        continue
                except CyclicNotationException as e:
                    raise InitialisationError("The candidate solution is not a valid cycle")
           
            population.add_individual(individual)
            counter +=1

    def _extend_the_population_with_random_individuals(self, population: Population,  number_of_individuals_to_generate: int) -> None:
        counter = 0
        while counter < number_of_individuals_to_generate:
            candidate_solution = np.random.permutation(self._all_cities)
            individual = Individual(self._NUMBER_OF_CITIES, self._distance_matrix)
            individual.use_path_notation(candidate_solution)

            _, has_inifite = individual.get_distance()
            if has_inifite:
                improve(individual.get_cyclic_representation_exposed(), self._distance_matrix)
                individual.reset_distance_calc()
                _, has_inifite = individual.get_distance()
                if has_inifite:
                    continue
                

            
            if DEBUG:
                try:
                    for city in individual():
                        continue
                except CyclicNotationException as e:
                    raise InitialisationError("The candidate solution is not a valid cycle")
            population.add_individual(individual)
            counter +=1