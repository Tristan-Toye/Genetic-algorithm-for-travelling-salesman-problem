from Debug import DEBUG
from EvolutionaryFunctions import EvolutionaryFunctions
from Exceptions import SelectionException
from Individual import Individual
from Population import Population
from typing import List, Tuple
import numpy as np

class Selection:
    def __init__(self, k: int) -> None:
        self._k = k
        self._selection_function = min

    # def select_amt(self, population: Population, num_pairs: int) -> List[Tuple[Individual, Individual]]:

    #     selected_pairs = []
    #     for _ in range(num_pairs):
    #         parent1 = EvolutionaryFunctions.k_tournament(population, self.k, self.selection_function)
    #         parent2 = EvolutionaryFunctions.k_tournament(population, self.k, self.selection_function)
    #         selected_pairs.append((parent1, parent2))
    #     return selected_pairs
    
    def select_pair(self,distance_matrix: np.ndarray, population: Population) -> Tuple[Individual, Individual]:
        parent1 = EvolutionaryFunctions.k_tournament(population.get_population_exposed(), self._k, distance_matrix, self._selection_function)
        parent2 = EvolutionaryFunctions.k_tournament(population.get_population_exposed(), self._k,distance_matrix, self._selection_function)
        while parent1 == parent2:
            parent2 = EvolutionaryFunctions.k_tournament(population.get_population_exposed(), self._k,distance_matrix, self._selection_function)

        return parent1, parent2
    
    def select_multiple_pair(self,array_parents1: np.ndarray, array_parents2: np.ndarray, distance_matrix: np.ndarray, population: Population):

        if DEBUG and array_parents1.size  != array_parents2.size:
            raise SelectionException("Length of parent lists must be equal")
            
        for index in range(2*array_parents1.size):
            new_parent = EvolutionaryFunctions.k_tournament(population.get_population_for_elemination()[1], self._k, distance_matrix, self._selection_function)
            
            while new_parent in array_parents1 or new_parent in array_parents2:
                new_parent = EvolutionaryFunctions.k_tournament(population.get_population_for_elemination()[1], self._k, distance_matrix, self._selection_function)

            if index < array_parents1.size:
                array_parents1[index] = new_parent.copy()
            else:
                array_parents2[index%array_parents1.size] = new_parent.copy()


