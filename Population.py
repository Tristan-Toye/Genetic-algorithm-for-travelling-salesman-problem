from copy import deepcopy
import random as random
import time
from Debug import DEBUG
from Exceptions import InitialisationError
from Individual import Individual
from typing import List, Iterable, Set, Tuple
import numpy as np

from MutationFactory import MutationFactory
from Parameters import Parameters
from local_search import optimise

class Population():


    _sorted_population: List[Individual]
    _parameters: Parameters
    def __init__(self,  parameters: Parameters) -> None:
        

        self._sorted_population = list()
        self._parameters = parameters

    def __iter__(self):
        for index, individual in enumerate(self._sorted_population):
            yield index, individual
    

    def __next__(self):
        pass

    def has_infinites(self) -> bool:
        return any([ind.get_distance()[1] for ind in self._sorted_population])

    @property
    def size(self) -> int:
        return self._parameters.get_population_size()

    def get_original_population_size(self) -> int:
        return self._parameters.get_population_size()
    
    def get_current_population_size(self) -> int:
        return len(self._sorted_population)

    def add_individual(self, individual: Individual) -> None:
 
        for index, pop_individual in enumerate(self._sorted_population):
            if individual.get_distance()[0]<= pop_individual.get_distance()[0]:
                self._sorted_population.insert(index , individual)
                return
        self._sorted_population.append(individual)
    
    def add_individuals(self, individuals: np.ndarray) -> None:
        for individual in individuals:
            self.add_individual(individual)
    
    def remove_individual(self, individual: Individual) -> None:
        self._sorted_population.remove(individual)

    def remove_by_index(self, indexes: Iterable[int]):
        self._sorted_population = [ind for index, ind in enumerate(self._sorted_population) if index not in indexes]
    
    def get_objective_values(self) -> List[float]:
        return [individual.get_distance()[0] for individual in self._sorted_population]
    
    def __getitem__(self, index: int|np.intp) -> Individual:
        return self._sorted_population[index]
    
    def get_population_exposed(self) -> List[Individual]:
        return self._sorted_population
    
    def get_population_for_elemination(self) -> Tuple[List[Individual], List[Individual]]:
        return self._sorted_population[:self._parameters.get_immunity_top_population()], self._sorted_population[self._parameters.get_immunity_top_population():]

    def set_sorted_population_exposed(self, sorted_population: List[Individual]):
        self._sorted_population = sorted_population

    def sort(self):
        self._sorted_population.sort(key = lambda ind: ind.get_distance())

    def get_rest_of_population(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray,np.ndarray]:
        best_population = np.empty(self._parameters.get_immunity_top_population(), dtype= object)
        index = 0
        for indv in self._sorted_population:
            if not (indv in parent1 or indv in parent2):
                best_population[index] = indv
                index +=1
                if index == self._parameters.get_immunity_top_population():
                    break
        bulk = np.array([indv for indv in self._sorted_population if not (indv in parent1 or indv in parent2 or indv in best_population)])
        return best_population, bulk
    
    def extend_with_inversed_converged_solutions(self, mutation: MutationFactory, converged_population: np.ndarray, inversed_converged_population: np.ndarray, number_of_mutations:int = 1 ):
        
        
        if converged_population.size != 0:
            vfunc = np.vectorize(lambda indv: indv.copy())
            copied_converged_population = vfunc(converged_population)
            vfunc_IVM = np.vectorize(lambda indv: mutation.IVM(indv, after_local_opt= False))
            vfunc_SIM = np.vectorize(lambda indv: mutation.SIM(indv, after_local_opt= False))
            
            inversed_converged_population[: converged_population.size], inversed_converged_population[converged_population.size:] = \
                vfunc_IVM(copied_converged_population), vfunc_SIM(copied_converged_population)
            
            p = np.random.choice(number_of_mutations)
            for _ in range(p):
                inversed_converged_population[: converged_population.size], inversed_converged_population[converged_population.size:] = \
                vfunc_IVM(inversed_converged_population[: converged_population.size]), vfunc_SIM(inversed_converged_population[converged_population.size:])
            
    def get_number_of_converged(self) -> int:
        return len([indv.is_converged() for indv in self._sorted_population if indv.is_converged()])
    
    def get_converged(self) -> List[Individual]:
        return [indv for indv in self._sorted_population if indv.is_converged()]