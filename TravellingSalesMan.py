import queue
import threading
import time
import numpy as np
from typing import Iterable, Tuple, cast
from Debug import DEBUG, PRINT_STAGES, SEQUENTIAL
from Individual import Individual
from Initialisation import Initialisation
from MutationFactory import MutationFactory
from Parameters import Parameters
from Population import Population
from Variation import Variation
from Selection import Selection
from Elimination import Elimination
from EvolutionaryFunctions import EvolutionaryFunctions
from local_search import improve_population, optimise
from multiprocessing import Queue

class TravellingSalesMan():

    _parameters: Parameters
    _distance_matrix: np.ndarray
    _selection: Selection
    _variation: Variation
    _elimination: Elimination
    _initialisation: Initialisation
    _pheremone_matrix: np.ndarray
    _ordered_arg_distance_matrix: np.ndarray
    _best_objective: float 
    _mutation: MutationFactory
    _index_since_pheremone_matrix_second_core_update: int
    _queue_second_core_to_tsp: Queue
    _number_of_mutations:int
    _no_change_counter: int
    _queue_tsp_to_second_core: Queue


    def __init__(self, parameters: Parameters, distance_matrix: np.ndarray, ordered_arg_distance_matrix: np.ndarray, initialisation: Initialisation, pheremone_matrix: np.ndarray, queue_second_core_to_tsp: Queue, queue_tsp_to_second_core: Queue) -> None:
        self._parameters = parameters
        self._distance_matrix = distance_matrix

        self._selection = Selection(self._parameters.get_k_selection())
        self._mutation = MutationFactory(self._parameters)
        self._variation = Variation(self._parameters)
        self._elimination = Elimination(distance_matrix, parameters, initialisation)
        self._mutation = MutationFactory(self._parameters)
        self._initialisation = initialisation
        self._pheremone_matrix = pheremone_matrix
        self._ordered_arg_distance_matrix =ordered_arg_distance_matrix
        self._best_objective = -1
        self._mean_objective = -1
        self._index = 0
        self._index_since_pheremone_matrix_second_core_update = 0
        self._queue_second_core_to_tsp = queue_second_core_to_tsp
        self._number_of_mutations = 1
        self._no_change_counter = 0

        self._parent1_list = np.empty(self._parameters.get_number_of_parent_pairs(), dtype= object)
        self._parent2_list = np.empty(self._parameters.get_number_of_parent_pairs(), dtype= object)
        self._offspring_list = np.ndarray(shape = (self._parameters.get_number_of_parent_pairs(), self._parameters.get_offspring_size()), dtype= object)
        self._pheremone_list = np.empty(self._parameters.get_number_of_pheremone_offspring(), dtype= object)
        self._queue_tsp_to_second_core = queue_tsp_to_second_core

   
        
        

    def run(self, population: Population) -> Tuple[float, float, np.ndarray]:
        self._index +=1
        self._index_since_pheremone_matrix_second_core_update += 1
        
        if PRINT_STAGES: print("selection")
        self._selection.select_multiple_pair(self._parent1_list, self._parent2_list, self._distance_matrix, population)
        
        if PRINT_STAGES: print("variation & optimisation")
        # recombination threads
      
        offspring_threads = self._variation.produce_offspring(self._offspring_list, self._parent1_list, self._parent2_list, self._distance_matrix, self._ordered_arg_distance_matrix)
 
        
        #thread to enhance parents
        best_population , rest_of_population = population.get_population_for_elemination()
        # tmp = threading.Thread(target = improve_population, kwargs = {"population":rest_of_population, "distance_matrix": self._distance_matrix})
        # if SEQUENTIAL:
        #     improve_population(**{"population":rest_of_population, "distance_matrix": self._distance_matrix})
        # else: 
        #     offspring_threads.append(tmp)

        #thread to genereate mutations of converged solutions
        converged_population = np.array(population.get_converged())
        inversed_converged_population = np.empty(2*converged_population.size, dtype=object)
        if SEQUENTIAL:
            population.extend_with_inversed_converged_solutions(**{"mutation":self._mutation, "converged_population": converged_population ,"inversed_converged_population": inversed_converged_population, "number_of_mutations": self._number_of_mutations})
        else:
            tmp2 = threading.Thread(target = population.extend_with_inversed_converged_solutions, kwargs= {"mutation":self._mutation, "converged_population": converged_population ,"inversed_converged_population": inversed_converged_population, "number_of_mutations": self._number_of_mutations})
            offspring_threads.append(tmp2)

        #thread to genereate candidates from pheremone matrix
       
        self._initialisation.extend_using_distances_and_pheremone(**{"out":self._pheremone_list})
     
        #offspring_threads.append(tmp2)
       
        #starting all threads
        
        if not SEQUENTIAL:
            for thread in offspring_threads:
                thread.start()
        
        
        while True:
            try:
                new_solution = cast(np.ndarray ,self._queue_second_core_to_tsp.get_nowait())
                individual = Individual(self._parameters.get_number_of_cities(), self._distance_matrix)
                individual.use_cyclic_notation(new_solution)
                
                population.add_individual(individual)
                #recalculate immunity section and population size accordingly
                self._parameters.set_immunity(population.get_number_of_converged())
                self._parameters.set_population_size(population.get_number_of_converged())

                #self._queue_tsp_to_second_core.put_nowait(self._pheremone_matrix)
                #self._index_since_pheremone_matrix_second_core_update = 0
            except queue.Empty:
                break
    
        
        opt_count = 0
        ended_before_threads = False

        if self._parameters.index_to_sync_pheremone_matrix() != -1 and self._index != 0 and self._index % self._parameters.index_to_sync_pheremone_matrix():
            try:
                self._queue_tsp_to_second_core.put_nowait(self._pheremone_matrix)
            except queue.Full:
                pass
        
        if SEQUENTIAL:
            optimise(best_population, self._distance_matrix, self._ordered_arg_distance_matrix, self._parameters)

        else: 
        #main thread does local optimiser on immune population untill threads are done
       
            while True:
                start3 = time.time()
                did_work = optimise(best_population, self._distance_matrix, self._ordered_arg_distance_matrix, self._parameters)
                tmp = time.time() - start3
                self._avg_opt = ((self._index - 1) * self._avg_opt + tmp)/self._index
                opt_count +=1
                if not did_work: #don't waste time in infinite loop
                    ended_before_threads = True
                    for thread in offspring_threads:
                        thread.join()

                if EvolutionaryFunctions.no_active_threads(offspring_threads):
                    break

        
        if PRINT_STAGES: print(f"DONE: optimised {opt_count} times")
        offspring_list_flatten = self._offspring_list.flatten()
        
        if DEBUG and None in offspring_list_flatten:
            raise Exception(f"none in offspring: {offspring_list_flatten}")

        #restore state of recombination
        offspring_list_flatten[offspring_list_flatten is not None]
        population.sort()
        
        #merge relevant offspring
        #   non grouped results from recombination
        #   1 pheremone generated individual
        #   inverse mutation on converged solutions
        if inversed_converged_population.size:
            vfunc = np.vectorize(lambda indv: indv.get_distance()[0])   
            inversed_converged_population = inversed_converged_population[vfunc(inversed_converged_population) < self._parameters.restriction_on_length_of_inversion_offspring()*self._best_objective]
            offspring = EvolutionaryFunctions.delete_groupings(offspring_list_flatten, self._parameters.get_number_of_cities() - self._parameters.get_grouping_equality_for_elemintation())
            offspring = np.concatenate((offspring, self._pheremone_list, inversed_converged_population))
        else:
            offspring = EvolutionaryFunctions.delete_groupings(offspring_list_flatten, self._parameters.get_number_of_cities() - self._parameters.get_grouping_equality_for_elemintation())
            offspring = np.concatenate((offspring, self._pheremone_list))
        
        
        
        if PRINT_STAGES: print("elimination")
        
        self._elimination.eliminate(population, offspring, self._index)
        
        #recalculate immunity section and population size accordingly
        self._parameters.set_immunity(population.get_number_of_converged())
        self._parameters.set_population_size(population.get_number_of_converged())

        if PRINT_STAGES: print("evaluation")
        objective_values = population.get_objective_values()

        best_index = 0
        best_individual = population[best_index]
        best_objective = objective_values[best_index]
        if DEBUG and self._best_objective != -1 and best_objective > self._best_objective:
            raise Exception()
        
        if self._best_objective == best_individual.get_distance()[0]:
            self._no_change_counter +=1

        else:
            self._no_change_counter = 0

        if self._no_change_counter == self._parameters.tsp_increase_inversion_mutation_on_converged():
            self._no_change_counter = 0
            self._number_of_mutations = min(self._number_of_mutations + 1,  self._parameters.maximum_number_of_mutations())
            self._parameters.increase_restriction_on_length_of_inversion_offspring()



        
        mean_objective = float(np.mean(objective_values))
        self._best_objective = best_objective
        self._mean_objective = mean_objective
        if DEBUG and population[best_index] != population[np.argmin(objective_values)]:
            raise Exception("Population is not sorted")
        #best_individualif PRINT_STAGES:
        # print("#########################################################")
        # print(f"index: {self._index} \t best: {best_objective} \t number of opt: {opt_count} \t ended before_threads: {ended_before_threads} \t immune_size: {self._parameters.get_immunity_top_population()}\t converged: {population.get_number_of_converged()}\t population_size: {self._parameters.get_population_size()} ")
            

        return (
            mean_objective,            
            best_objective,                       
            best_individual.get_path_representation()
        )
    


    