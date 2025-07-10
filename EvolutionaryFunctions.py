from itertools import zip_longest
import random as random
from threading import Thread
from typing import Iterable, List, Callable, Optional, Tuple
from Exceptions import OutOfBoundsException
from Individual import Individual
from Population import Population
import numpy as np


class EvolutionaryFunctions():

    @staticmethod
    def k_tournament(population: List[Individual], k: int, distance_matrix: np.ndarray, selection_function: Callable[[List[Individual]], Individual]) -> Individual:
        """
        Get k random individuals from the population, then select one according to a selection_fuction.
        examples selection_function: min(), max()
        """
        if k > len(population):
            raise Exception()
        
        sampled_individuals = random.sample(population, k)
        func: Callable[[Individual], float]  = lambda individual: individual.get_fitness()
        individual = selection_function(sampled_individuals, **{"key": func})
        return individual
    @staticmethod
    def find_correct_index( boolean_array: np.ndarray, length: int) -> Callable[[int],int]:

        def tmp(number: int) -> int:
            if number < length: return number
                 
            number = number - length                                                                                           
            if number > boolean_array.size: raise OutOfBoundsException("incorrect number to search")                                
            count = 0                                                                                                           
            for index, element in np.ndenumerate(boolean_array):                                                                
                if element:                                                                                                     
                    if count == number: return index[0]                                                                          
                    else: count +=1  
            return -1
        return tmp                                   
   

    """
    original_size = population_object.get_original_population_size() - immune.size
    _, population = population_object.get_population_for_elemination()
   
    """
    @staticmethod
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
    
    @staticmethod
    def non_duplicants_check(path :List[int], number_of_cities: int = -1) -> bool:
      number_of_cities = number_of_cities if number_of_cities != -1 else len(path) 

      sum_of_all_numbers = number_of_cities*(number_of_cities + 1) / 2
      sum_of_all_entries_in_path = sum(path)
      return sum_of_all_entries_in_path == sum_of_all_numbers

    @staticmethod
    def delete_groupings(offspring: np.ndarray, non_similar_cities: int) -> np.ndarray:
        
                
        offspring_remove_indexes = np.full_like(offspring, True, dtype=np.bool)
        for offspring_index, offspring_candidate in np.ndenumerate(offspring):
            if not offspring_remove_indexes[offspring_index[0]]: continue #should not happen
            if offspring_index[0] == offspring.size -1: continue
            for population_index, population_candidate in np.ndenumerate(offspring[offspring_index[0]+1:]):
                if not offspring_remove_indexes[population_index[0]]: continue
                count = 0
                for offspring_city, population_city in zip_longest(offspring_candidate.get_cyclic_representation_exposed(), population_candidate.get_cyclic_representation_exposed()):
                    if offspring_city != population_city: count +=1
                    if count > non_similar_cities: break
                if count <= non_similar_cities: 
                    if offspring_candidate.get_distance() <= population_candidate.get_distance():
                        offspring_remove_indexes[population_index[0]] = False
                        continue
                    else:
                        offspring_remove_indexes[offspring_index[0]] = False
                        break

        return offspring[offspring_remove_indexes]
  
    

    @staticmethod
    def delete_groupings_list(offspring: List[Individual], non_similar_cities: int):
        
                
        offspring_remove_indexes = np.full_like(offspring, True, dtype=np.bool)
        for offspring_index, offspring_candidate in enumerate(offspring):
            if not offspring_remove_indexes[offspring_index]: continue #should not happen
            if offspring_index == len(offspring) -1: continue
            for population_index, population_candidate in enumerate(offspring[offspring_index+1:]):
                if not offspring_remove_indexes[population_index]: continue
                count = 0
                for offspring_city, population_city in zip_longest(offspring_candidate.get_cyclic_representation_exposed(), population_candidate.get_cyclic_representation_exposed()):
                    if offspring_city != population_city: count +=1
                    if count > non_similar_cities: break
                if count <= non_similar_cities: 
                    if offspring_candidate.get_distance() <= population_candidate.get_distance():
                        offspring_remove_indexes[population_index] = False
                        continue
                    else:
                        offspring_remove_indexes[offspring_index] = False
                        break

        for index, can_stay in np.ndenumerate(offspring_remove_indexes):
            if not can_stay:
                offspring.pop(index[0])
    

    @staticmethod
    def update_pheremone_matrix(pheremone_weight:int, pheremone_matrix: np.ndarray, offspring: Iterable[Individual]):

 
        for candidated_offspring in offspring:
            if candidated_offspring is None: continue
            distance = candidated_offspring.get_distance()[0]
            for index,city in np.ndenumerate(candidated_offspring.get_cyclic_representation_exposed()):
                pheremone_matrix[index[0], city] =pheremone_matrix[index[0], city]*0.9 + pheremone_weight/distance
    
    @staticmethod
    def delete_equals(population: Population):
        offspring_remove_indexes = np.full(population.get_current_population_size(), True, dtype=np.bool)
        for index, individual in enumerate(population.get_population_exposed()):
            if index == population.get_current_population_size() - 1: continue
            if individual == population.get_population_exposed()[index + 1]:
                offspring_remove_indexes[index] = False

      
        population.remove_by_index([index[0] for index, can_stay in np.ndenumerate(offspring_remove_indexes) if not can_stay])

    @staticmethod
    def delete_equals_list(population: List[Individual]):
        offspring_remove_indexes = np.full(len(population), True, dtype=np.bool)
        for index, individual in enumerate(population):
            for candidate_index, candidate in enumerate(population[index + 1:]):
                if not offspring_remove_indexes[candidate_index]: continue
                if individual == candidate:
                    offspring_remove_indexes[candidate_index] = False

        return [ind for index, ind in enumerate(population) if offspring_remove_indexes[index]]
       
    @staticmethod
    def adjust_pheremone_matrix(original_matrix: np.ndarray, update_matrix: np.ndarray):
        np.add(original_matrix, update_matrix, out=original_matrix)

    @staticmethod
    def no_active_threads(thread_list: List[Thread]):
        for thread in thread_list:
            if thread.is_alive():
                return False
        return True