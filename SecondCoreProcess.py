



import queue
import threading
import time
from typing import List, cast

import numpy as np
from Debug import DEBUG

from EvolutionaryFunctions import EvolutionaryFunctions

from Individual import Individual
from Initialisation import Initialisation
from MutationFactory import MutationFactory
from Parameters import Parameters
from Population import Population
from Reporter import Reporter
from local_search import find_local_optima
from multiprocessing import Queue


class SecondCoreProcess:

    _population: Population
    _mutation: MutationFactory
    _distance_matrix: np.ndarray
    _ordered_arg_distance_matrix: np.ndarray
    _index: int
    _initialisation: Initialisation
    _no_change_counter: int
    _queue_second_core_to_tsp: Queue
    _new_solutions_found_counter: int
    _number_of_mutations:int 
    _queue_tsp_to_second_core: Queue
    _best_solution: float

    def __init__(self,reporter: Reporter, initialisation: Initialisation, init_population:List[Individual], distance_matrix: np.ndarray, ordered_arg_distance_matrix: np.ndarray, queue_second_core_to_tsp: Queue, queue_tsp_to_second_core: Queue) -> None:
        
        self._parameters = Parameters()
        self._parameters.set_number_of_cities(distance_matrix.shape[0])
        self._initialisation = initialisation
        self._initialisation.set_new_parameters(self._parameters)

        self._population = Population(self._parameters)
        self._population.set_sorted_population_exposed(init_population)
        self._mutation = MutationFactory(self._parameters)

        self._distance_matrix = distance_matrix
        self._ordered_arg_distance_matrix = ordered_arg_distance_matrix
        self._reporter = reporter
        self._index = 0
        self._no_change_counter = 0
        self._best_solution = self._population.get_population_exposed()[0].get_distance()[0]
        self._queue_second_core_to_tsp = queue_second_core_to_tsp
        self._number_of_mutations = 1
        self._queue_tsp_to_second_core = queue_tsp_to_second_core
        self._avg = 0

    def run(self):
        
        print("START SECOND CORE")
        while True:
            self._index +=1
            thread_list = list()
            
            for individual in self._population.get_population_exposed():
                
                if not individual.is_converged():
                    tmp = threading.Thread(target=find_local_optima, kwargs={"individual": individual, "distance_matrix": self._distance_matrix, "ordered_arg_distance_matrix": self._ordered_arg_distance_matrix})
                    thread_list.append(tmp)
                    
                    #find_local_optima(**{"individual": individual, "distance_matrix": self._distance_matrix, "ordered_arg_distance_matrix": self._ordered_arg_distance_matrix})
                    
                    # if self._best_solution <= individual.get_distance()[0]:
                    #     self._no_change_counter +=1
                    # else:
                    #     self._best_solution = individual.get_distance()[0]
                    #     print(f"send second core solution: {individual.get_distance()[0]}")
                    #     self._queue_second_core_to_tsp.put(individual)
      
           
            for thread in thread_list:
                thread.start()
        
            
            try:
                new_pheremone_matrix = cast(np.ndarray, self._queue_tsp_to_second_core.get_nowait())
                new_pheremone_matrix = new_pheremone_matrix.astype(np.float64)
                EvolutionaryFunctions.adjust_pheremone_matrix(self._initialisation.get_pheremone_matrix_exposed(), new_pheremone_matrix)
            except queue.Empty :
                pass
            
            for thread in thread_list:
                 thread.join()
            
            self._population.sort()

            # for individual in self._population.get_population_exposed():
            #     if self._best_solution <= individual.get_distance()[0]:
            #         self._no_change_counter +=1
            #     else:
            #         self._no_change_counter = 0
            #         self._queue_second_core_to_tsp.put(individual)
            #         print(f"put {individual.get_distance()[0]}")

            if self._best_solution >  self._population.get_population_exposed()[0].get_distance()[0]:
                self._best_solution = self._population.get_population_exposed()[0].get_distance()[0]
                self._queue_second_core_to_tsp.put(self._population.get_population_exposed()[0].get_cyclic_representation_exposed())
                
                self._no_change_counter = 0
            else: 
                self._no_change_counter +=1

            #EvolutionaryFunctions.delete_equals(self._population)
            self._population.set_sorted_population_exposed(self._population.get_population_exposed()[:self._parameters.get_population_size_second_core()])
           
            if self._no_change_counter == self._parameters.get_second_core_reset():
                
                self._no_change_counter = 0
                self._population.set_sorted_population_exposed(self._population.get_population_exposed()[:1])
                self._initialisation.extend_population(self._population, intended_size= self._parameters.get_population_size_second_core())
                
                self._number_of_mutations = min(self._number_of_mutations +1 , self._parameters.maximum_number_of_mutations())
                self._parameters.increase_restriction_on_length_of_inversion_offspring()
            else: #sometimes extention produces better solution then already converted, then the 
                self._parameters.set_immunity(self._population.get_number_of_converged())

                converged_population = np.array(self._population.get_converged())
                if converged_population.size:
                    inversed_converged_population = np.empty(2*converged_population.size, dtype=object)
                    self._population.extend_with_inversed_converged_solutions( self._mutation, converged_population,inversed_converged_population, number_of_mutations= self._number_of_mutations)
                    vfunc = np.vectorize(lambda indv: indv.get_distance()[0])
                    inversed_converged_population = inversed_converged_population[vfunc(inversed_converged_population) < self._parameters.restriction_on_length_of_inversion_offspring()*self._best_solution]
                    self._population.add_individuals(inversed_converged_population)
                else:
                    print("||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||")
                    print(f"no change counter: {self._no_change_counter}")
                    print(f"index: {self._index}")
                    print(f"population size: {self._population.get_current_population_size()}")
                    print(f"population: {[indv.is_converged() for indv in self._population.get_population_exposed()]}")
                    print("||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||")
            # EvolutionaryFunctions.update_pheremone_matrix(self._parameters.get_pheromone_scaling(), self._initialisation.get_pheremone_matrix_exposed(), self._population.get_population_exposed())
            if DEBUG and self._population[0] != self._population[np.argmin(np.array([indv.get_distance()[0] for indv in self._population.get_population_exposed()]))]:
                raise Exception("Population is not sorted")
            

            #rint(f"Second core index: {self._index} \t best: {self._best_solution}")

            # timeLeft = self._reporter.get_time()
            # if timeLeft < 240:
            #     print(f"second core iterations: {self._index}")
            #     break

        return 0
        

