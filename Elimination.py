from itertools import zip_longest
import time
import numpy as np
from Debug import DEBUG
from Exceptions import ElimenationException
from Individual import Individual
from Initialisation import Initialisation
from Parameters import Parameters
from Population import Population
from EvolutionaryFunctions import EvolutionaryFunctions
from numba import jit, float64, int16, boolean

@jit(boolean[:](float64[:], int16, int16),nopython = True)
def k_tournament_elem(population_and_offspring_fitness: np.ndarray, k: int, original_size: int) -> np.ndarray:
    """
    Get k random individuals from the population, then select one according to a selection_fuction.
    examples selection_function: min(), max()
    """

    remove_indexes = np.full_like(population_and_offspring_fitness, True, dtype=np.bool)
    if np.count_nonzero(remove_indexes)  <= original_size:
        return remove_indexes

    if k > original_size:
        k = original_size
    
    while np.count_nonzero(remove_indexes) != original_size:

        sampled_individuals_index = np.random.choice( np.nonzero(remove_indexes)[0], k, replace=False)


        sampled_individuals = population_and_offspring_fitness[sampled_individuals_index]

        individual_index= np.argmax(sampled_individuals) # type: ignore 
        index_elem_population = sampled_individuals_index[individual_index]
        remove_indexes[index_elem_population] = False

    
    return remove_indexes

@jit((int16[:,:], float64[:], int16, int16, float64),nopython= True)
def calculate_scalings(rest_with_offspring_array: np.ndarray, scalings: np.ndarray, threshold: int, number_of_cities: int, penalty: float):
    for index, candidate in enumerate(rest_with_offspring_array):
        if index == rest_with_offspring_array.size -1: continue
        for other_index, other_candidate in enumerate(rest_with_offspring_array[index + 1:]):
            equal_edges = np.count_nonzero(candidate == other_candidate)
            if equal_edges <= threshold: continue
            scaling = (2 - ((number_of_cities - equal_edges)/threshold)**penalty)
            scalings[index] *= scaling
            scalings[other_index] *= scaling
    


class Elimination():

    _k: int
    _distance_matrix: np.ndarray
    _parameters: Parameters
    _initialisation: Initialisation

    def __init__(self,  distance_matrix: np.ndarray, parameters: Parameters, initialisation: Initialisation) -> None:
        self._parameters = parameters
        self._k = self._parameters.get_k_elimination()
        self._distance_matrix = distance_matrix
        self._parameters = parameters
        self._initialisation = initialisation

        
    def eliminate(self, population: Population, offspring: np.ndarray, index:int, extend: bool = True):

        
        #assumption: population._popopulation.size = population.get_original_population_size()
        if not offspring.size: return population
        

        EvolutionaryFunctions.delete_equals(population)
        self._delete_equals(population, offspring)
        
        EvolutionaryFunctions.update_pheremone_matrix(self._parameters.get_pheromone_scaling(), self._initialisation.get_pheremone_matrix_exposed(), offspring)

        if population.get_number_of_converged() > self._parameters.get_max_converged():
            immune, rest = population.get_population_for_elemination()
            immune = immune[:self._parameters.get_reset_converged()]
            immune.extend(rest)
            population.set_sorted_population_exposed(immune)
        
        # if extend and population.get_current_population_size() < population.get_original_population_size():
        #     self._initialisation.extend_population(population)

        if not extend and population.get_current_population_size() + offspring.size < population.get_original_population_size():
            self._merge_offspring_with_population(population, offspring)
            return


        self._merge_offspring_with_population(population, offspring)


        self._punish_groupings(population)


        self._reduce_population(population)
 

        if extend and population.get_current_population_size() < population.get_original_population_size():
            self._initialisation.extend_population(population)


        if DEBUG and len(population.get_population_exposed()) != population.get_original_population_size():
            raise ElimenationException("Population is not back to original size after elemination")
            
        
    def set_k(self, k) -> None:
        self._k = k

    def get_k(self) -> int:
        return self._k
    
    def _merge_offspring_with_population(self, population: Population, offspring: np.ndarray):
        population.add_individuals(offspring)

    def _delete_equals(self, population: Population, offspring: np.ndarray):
        offspring_remove_indexes = np.full_like(offspring, True, dtype=np.bool)
        for individual in population:
            for candidate_index, candidate in np.ndenumerate(offspring):
                if not offspring_remove_indexes[candidate_index[0]]: continue
                if individual == candidate:
                    offspring_remove_indexes[candidate_index[0]] = False
        return offspring[offspring_remove_indexes]

    def _reduce_population(self, population: Population):
        immune, rest_and_offspring = population.get_population_for_elemination()
        rest_and_offspring_array = np.array([indv.get_fitness() for indv in rest_and_offspring])
        indexes = k_tournament_elem(rest_and_offspring_array, self._k, population.get_original_population_size() - len(immune))
        immune.extend([indv for index, indv in enumerate(rest_and_offspring) if indexes[index]])
        population.set_sorted_population_exposed(immune)
    
    def _punish_groupings(self, population: Population):

        threshold = self._parameters.get_threshold_for_likeness_penalty()
        immune, rest_with_offspring = population.get_population_for_elemination()
        
        rest_with_offspring_array = np.empty((len(rest_with_offspring), self._parameters.get_number_of_cities()), dtype= np.int16)
        for index, individual in enumerate(rest_with_offspring):
            rest_with_offspring_array[index] = individual.get_cyclic_representation_exposed()
        
        scalings = np.full(len(rest_with_offspring), 1, dtype = np.float64)
       

        calculate_scalings(rest_with_offspring_array, scalings, threshold, self._parameters.get_number_of_cities(), self._parameters.get_scaling_likeness_penalty())
        
        for index, individual in enumerate(rest_with_offspring):
            individual.set_scaling(scalings[index])

    
    def delete_groupings_converged_population(self, population: Population):
        non_similar_cities = self._parameters.get_grouping_equality_for_elemintation()
        
        immune_population , rest = population.get_population_for_elemination()
        
        converged_population = np.array(immune_population)[[indv.is_converged() for indv in immune_population]]
        immune_offspring_remove_indexes = np.full_like(converged_population, True, dtype=np.bool)
        for immune_index, immune_candidate in np.ndenumerate(converged_population):
            if not immune_offspring_remove_indexes[immune_index[0]]: continue 
            for population_index, population_candidate in np.ndenumerate(converged_population[immune_index[0] + 1:]):
                if not immune_offspring_remove_indexes[population_index[0]]: continue 
                count = 0
                for immune_city, population_city in zip_longest(immune_candidate.get_cyclic_representation_exposed(), population_candidate.get_cyclic_representation_exposed()):
                    if immune_city != population_city: count +=1
                    if count > non_similar_cities: break
                if count <= non_similar_cities: 
                    if immune_candidate.get_distance() <= population_candidate.get_distance():
                        immune_offspring_remove_indexes[population_index[0] + immune_index[0] + 1] = False
                        continue
                    else:
                        immune_offspring_remove_indexes[immune_index[0]] = False #should not happen
                        break

        new_converged_population = converged_population[immune_offspring_remove_indexes]

        new_population = new_converged_population.tolist()
        new_population.extend(rest)
        population.set_sorted_population_exposed(new_population)

        if population.get_current_population_size() < population.get_original_population_size():
            self._initialisation.extend_population(population)

        