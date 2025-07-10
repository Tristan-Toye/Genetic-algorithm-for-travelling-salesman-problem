from copy import deepcopy
import threading
import time
import numpy as np
from typing import Callable, List, Set, Tuple
from Exceptions import CyclicNotationException, HeuristicCrossoverError, OX1CrossoverError, OX1Exception, POSCrossoverError, POSException, ParentsException, edgeCrossoverError
from Individual import Individual
from Debug import DEBUG, PRINT_THREADING, PRINT_VARIATION, SEQUENTIAL
from itertools import zip_longest

from MutationFactory import MutationFactory
from Parameters import Parameters
from local_search import improve
from numba import jit, float64, int16, boolean


class Variation():

    _parameters: Parameters
    _mutation: MutationFactory

    def __init__(self, parameters: Parameters) -> None:
        self._parameters = parameters
        self._mutation = MutationFactory(parameters)


    def _deepcopy_list_individual(self, array: np.ndarray) -> np.ndarray:
        tmp = np.empty_like(array, dtype=object)
        for index, indvidual in np.ndenumerate(array):
            tmp[index[0]] = indvidual.copy()
        return tmp
    
    def _thread_heuristic(self, parent1_list: np.ndarray, parent2_list: np.ndarray, offspring_list: np.ndarray,
                         distanceMatrix: np.ndarray, ordered_arg_distance_matrix: np.ndarray, base_case_no_city: np.ndarray, all_cities: np.ndarray,
                         start_index_loop:int
                         ):
      
        parent1_list = self._deepcopy_list_individual(parent1_list)
        parent2_list = self._deepcopy_list_individual(parent2_list)
        for index, (parent1, parent2) in enumerate(zip_longest(parent1_list, parent2_list)):
                if parent1 == parent2: continue
                self._create_offspring_heuristic(offspring_list[index], 
                                            np.arange(start_index_loop, start_index_loop + self._parameters.get_number_of_heuristic_offspring())
                                            , parent1,parent2, distanceMatrix,ordered_arg_distance_matrix, base_case_no_city, all_cities)
     
        
        if PRINT_THREADING:
                print("thread Heuristic is done")
    def _thread_POS(self, parent1_list: np.ndarray, parent2_list: np.ndarray, offspring_list: np.ndarray,
                         distanceMatrix: np.ndarray, ordered_arg_distance_matrix: np.ndarray, base_case_no_city: np.ndarray, all_cities: np.ndarray,
                         start_index_loop:int
                         ):
       
        parent1_list = self._deepcopy_list_individual(parent1_list)
        parent2_list = self._deepcopy_list_individual(parent2_list)
        for index, (parent1, parent2) in enumerate(zip_longest(parent1_list, parent2_list)):
                    if parent1 == parent2: continue
                    self._create_offspring_POS(offspring_list[index], 
                                                np.arange(start_index_loop, start_index_loop + self._parameters.get_number_of_POS_offspring())
                                                , parent1,parent2, distanceMatrix, base_case_no_city, all_cities)
     
       
        if PRINT_THREADING:
                print("thread POS is done")
    def _thread_OX1(self, parent1_list: np.ndarray, parent2_list: np.ndarray, offspring_list: np.ndarray,
                         distanceMatrix: np.ndarray, ordered_arg_distance_matrix: np.ndarray, base_case_no_city: np.ndarray, all_cities: np.ndarray,
                         start_index_loop:int
                         ):
       
        parent1_list = self._deepcopy_list_individual(parent1_list)
        parent2_list = self._deepcopy_list_individual(parent2_list)
        for index, (parent1, parent2) in enumerate(zip_longest(parent1_list, parent2_list)):
            if parent1 == parent2: continue
            self._create_offspring_OX1(offspring_list[index], 
                                        np.arange(start_index_loop, start_index_loop + self._parameters.get_number_of_OX1_offspring())
                                        , parent1,parent2, distanceMatrix, base_case_no_city, all_cities)
        
       
        if PRINT_THREADING:
                print("thread OX1 is done")
    def _thread_111(self, parent1_list: np.ndarray, parent2_list: np.ndarray, offspring_list: np.ndarray,
                         distanceMatrix: np.ndarray, ordered_arg_distance_matrix: np.ndarray, base_case_no_city: np.ndarray, all_cities: np.ndarray,
                         start_index_loop:int
                         ):
        start_time = time.time()
        parent1_list = self._deepcopy_list_individual(parent1_list)
        parent2_list = self._deepcopy_list_individual(parent2_list)
        for index, (parent1, parent2) in enumerate(zip_longest(parent1_list, parent2_list)):
            if parent1 == parent2: continue
            self._create_offspring_1_1_1(offspring_list[index], 
                                        np.arange(start_index_loop, start_index_loop + self._parameters.get_number_of_1_1_1_offspring())
                                        , parent1,parent2, distanceMatrix, base_case_no_city, all_cities)
        print(f"111: {time.time() - start_time}")
        if PRINT_THREADING:
                print("thread 111 is done")
    def _thread_11P(self, parent1_list: np.ndarray, parent2_list: np.ndarray, offspring_list: np.ndarray,
                         distanceMatrix: np.ndarray, ordered_arg_distance_matrix: np.ndarray, base_case_no_city: np.ndarray, all_cities: np.ndarray,
                         start_index_loop:int
                         ):
        start_time = time.time()
        parent1_list = self._deepcopy_list_individual(parent1_list)
        parent2_list = self._deepcopy_list_individual(parent2_list)
        for index, (parent1, parent2) in enumerate(zip_longest(parent1_list, parent2_list)):
            if parent1 == parent2: continue
            self._create_offspring_1_1_para(offspring_list[index], 
                                            np.arange(start_index_loop, start_index_loop + self._parameters.get_number_of_1_1_para_offspring())
                                            , parent1,parent2, distanceMatrix, base_case_no_city, all_cities)
       
        print(f"11P: {time.time() - start_time}")
        if PRINT_THREADING:
                print("thread 11P is done")           
    def _thread_1P1(self, parent1_list: np.ndarray, parent2_list: np.ndarray, offspring_list: np.ndarray,
                         distanceMatrix: np.ndarray, ordered_arg_distance_matrix: np.ndarray, base_case_no_city: np.ndarray, all_cities: np.ndarray,
                         start_index_loop:int
                         ):
        start_time = time.time()
        parent1_list = self._deepcopy_list_individual(parent1_list)
        parent2_list = self._deepcopy_list_individual(parent2_list)
        for index, (parent1, parent2) in enumerate(zip_longest(parent1_list, parent2_list)):
            if parent1 == parent2: continue
            self._create_offspring_1_para_1(offspring_list[index], 
                                        np.arange(start_index_loop, start_index_loop + self._parameters.get_number_of_1_para_1_offspring())
                                                , parent1,parent2, distanceMatrix, base_case_no_city, all_cities)
        print(f"1P1: {time.time() - start_time}")
        if PRINT_THREADING:
                print("thread 1P1 is done")
    def _thread_1PP(self, parent1_list: np.ndarray, parent2_list: np.ndarray, offspring_list: np.ndarray,
                         distanceMatrix: np.ndarray, ordered_arg_distance_matrix: np.ndarray, base_case_no_city: np.ndarray, all_cities: np.ndarray,
                         start_index_loop:int
                         ):
        start_time = time.time()
        parent1_list = self._deepcopy_list_individual(parent1_list)
        parent2_list = self._deepcopy_list_individual(parent2_list)
        for index, (parent1, parent2) in enumerate(zip_longest(parent1_list, parent2_list)):
            if parent1 == parent2: continue
            self._create_offspring_1_para_para(offspring_list[index], 
                                        np.arange(start_index_loop, start_index_loop + self._parameters.get_number_of_1_para_para_offspring())
                                        , parent1,parent2, distanceMatrix, base_case_no_city, all_cities)
        print(f"1PP: {time.time() - start_time}")
        if PRINT_THREADING:
                print("thread 1PP is done")
    def _thread_PPP(self, parent1_list: np.ndarray, parent2_list: np.ndarray, offspring_list: np.ndarray,
                         distanceMatrix: np.ndarray, ordered_arg_distance_matrix: np.ndarray, base_case_no_city: np.ndarray, all_cities: np.ndarray,
                         start_index_loop:int
                         ):
        start_time = time.time()
        parent1_list = self._deepcopy_list_individual(parent1_list)
        parent2_list = self._deepcopy_list_individual(parent2_list)
        for index, (parent1, parent2) in enumerate(zip_longest(parent1_list, parent2_list)):
            if parent1 == parent2: continue
            self._create_offspring_para_para_para(offspring_list[index], 
                                        np.arange(start_index_loop, start_index_loop + self._parameters.get_number_of_para_para_para_offspring())
                                        , parent1,parent2, distanceMatrix, base_case_no_city, all_cities)
        print(f"PPP: {time.time() - start_time}")
        if PRINT_THREADING:
            print("thread PPP is done")
    
    def produce_offspring(self,offspring_list: np.ndarray, parent1_list: np.ndarray, parent2_list: np.ndarray, distanceMatrix: np.ndarray, ordered_arg_distance_matrix: np.ndarray) -> List[threading.Thread]:

        number_of_cities = self._parameters.get_number_of_cities()
        
        base_case_no_city = np.full(number_of_cities,-1, dtype= np.int16)
        all_cities = np.arange(number_of_cities, dtype=np.int16)
        start_index = 0

        
        thread_list = list()
        if self._parameters.get_number_of_heuristic_offspring():
       
            start_index_loop = start_index
            thread_kwargs = {
            "parent1_list": parent1_list,
            "parent2_list": parent2_list,
            "offspring_list": offspring_list,
            "distanceMatrix": distanceMatrix,
            "ordered_arg_distance_matrix": ordered_arg_distance_matrix,
            "base_case_no_city": base_case_no_city,
            "all_cities": all_cities,
            "start_index_loop": start_index_loop
            }
            tmp = threading.Thread(target=self._thread_heuristic, kwargs=thread_kwargs)
            thread_list.append(tmp)
            if SEQUENTIAL:
                self._thread_heuristic(**thread_kwargs)
            start_index += self._parameters.get_number_of_heuristic_offspring()

        if self._parameters.get_number_of_POS_offspring():
            start_index_loop = start_index
            thread_kwargs = {
            "parent1_list": parent1_list,
            "parent2_list": parent2_list,
            "offspring_list": offspring_list,
            "distanceMatrix": distanceMatrix,
            "ordered_arg_distance_matrix": ordered_arg_distance_matrix,
            "base_case_no_city": base_case_no_city,
            "all_cities": all_cities,
            "start_index_loop": start_index_loop
            }
            tmp = threading.Thread(target=self._thread_POS, kwargs=thread_kwargs)
            thread_list.append(tmp)
            if SEQUENTIAL:
                self._thread_POS(**thread_kwargs)
            start_index += self._parameters.get_number_of_POS_offspring()


        if self._parameters.get_number_of_OX1_offspring():
          
            start_index_loop = start_index
            thread_kwargs = {
            "parent1_list": parent1_list,
            "parent2_list": parent2_list,
            "offspring_list": offspring_list,
            "distanceMatrix": distanceMatrix,
            "ordered_arg_distance_matrix": ordered_arg_distance_matrix,
            "base_case_no_city": base_case_no_city,
            "all_cities": all_cities,
            "start_index_loop": start_index_loop
            }
            tmp = threading.Thread(target=self._thread_OX1, kwargs=thread_kwargs)
            thread_list.append(tmp)
            if SEQUENTIAL:
                self._thread_OX1(**thread_kwargs)
            start_index += self._parameters.get_number_of_OX1_offspring()



        #Modified ER: Genetic edge recombination crossover
        #   - preserve common subcycles
        #   - take edge from one of parents if possible
        #   - if not -> random
        #   - modification: lineair approach instead of prioritising elements with less possible edges available 
        if self._parameters.get_number_of_1_1_1_offspring():
            
            start_index_loop = start_index
            thread_kwargs = {
            "parent1_list": parent1_list,
            "parent2_list": parent2_list,
            "offspring_list": offspring_list,
            "distanceMatrix": distanceMatrix,
            "ordered_arg_distance_matrix": ordered_arg_distance_matrix,
            "base_case_no_city": base_case_no_city,
            "all_cities": all_cities,
            "start_index_loop": start_index_loop
            }
            tmp = threading.Thread(target=self._thread_111, kwargs=thread_kwargs)
            thread_list.append(tmp)
            if SEQUENTIAL:
                self._thread_111(**thread_kwargs)
            start_index += self._parameters.get_number_of_1_1_1_offspring()

        if self._parameters.get_number_of_1_1_para_offspring():
            
            
            start_index_loop = start_index
            thread_kwargs = {
            "parent1_list": parent1_list,
            "parent2_list": parent2_list,
            "offspring_list": offspring_list,
            "distanceMatrix": distanceMatrix,
            "ordered_arg_distance_matrix": ordered_arg_distance_matrix,
            "base_case_no_city": base_case_no_city,
            "all_cities": all_cities,
            "start_index_loop": start_index_loop
            }
            tmp = threading.Thread(target=self._thread_11P, kwargs=thread_kwargs)
            thread_list.append(tmp)
            if SEQUENTIAL:
                self._thread_11P(**thread_kwargs)
           
            start_index += self._parameters.get_number_of_1_1_para_offspring()

        if self._parameters.get_number_of_1_para_1_offspring():
           
 
            start_index_loop = start_index
            thread_kwargs = {
            "parent1_list": parent1_list,
            "parent2_list": parent2_list,
            "offspring_list": offspring_list,
            "distanceMatrix": distanceMatrix,
            "ordered_arg_distance_matrix": ordered_arg_distance_matrix,
            "base_case_no_city": base_case_no_city,
            "all_cities": all_cities,
            "start_index_loop": start_index_loop
            }
            tmp = threading.Thread(target=self._thread_1P1, kwargs=thread_kwargs)
            thread_list.append(tmp)
            if SEQUENTIAL:
                self._thread_1P1(**thread_kwargs)
            start_index += self._parameters.get_number_of_1_para_1_offspring()

        if self._parameters.get_number_of_1_para_para_offspring():
            start_index_loop = start_index
            thread_kwargs = {
            "parent1_list": parent1_list,
            "parent2_list": parent2_list,
            "offspring_list": offspring_list,
            "distanceMatrix": distanceMatrix,
            "ordered_arg_distance_matrix": ordered_arg_distance_matrix,
            "base_case_no_city": base_case_no_city,
            "all_cities": all_cities,
            "start_index_loop": start_index_loop
            }
            tmp = threading.Thread(target=self._thread_1PP, kwargs=thread_kwargs)
            thread_list.append(tmp)
            if SEQUENTIAL:
                self._thread_1PP(**thread_kwargs)
            
            start_index += self._parameters.get_number_of_1_para_para_offspring()

        if self._parameters.get_number_of_para_para_para_offspring():
            #parallel in future
            start_index_loop = start_index
            thread_kwargs = {
            "parent1_list": parent1_list,
            "parent2_list": parent2_list,
            "offspring_list": offspring_list,
            "distanceMatrix": distanceMatrix,
            "ordered_arg_distance_matrix": ordered_arg_distance_matrix,
            "base_case_no_city": base_case_no_city,
            "all_cities": all_cities,
            "start_index_loop": start_index_loop
            }
            tmp = threading.Thread(target=self._thread_PPP, kwargs=thread_kwargs)
            thread_list.append(tmp)
            if SEQUENTIAL:
                self._thread_PPP(**thread_kwargs)

        return thread_list
    
    def _create_offspring_heuristic(self,offspring_list : np.ndarray, indexes: np.ndarray, 
                               parent1: Individual, parent2: Individual, distanceMatrix: np.ndarray, ordered_arg_distance_matrix: np.ndarray, base_case_no_city: np.ndarray, all_cities: np.ndarray):
        
        #assumption: the length of indexes is even, this is enforced in parameters.py (starting position not guraranteed even index)
        for index in indexes:
            if offspring_list[index]: continue
            child1 = self._heuristic_crossover_with_common_subtour(parent1, parent2, distanceMatrix, ordered_arg_distance_matrix, base_case_no_city, all_cities)
            child2 = self._heuristic_crossover(parent1, parent2, distanceMatrix, ordered_arg_distance_matrix, base_case_no_city, all_cities)
            mutated_child1 = self._mutation.mutate(child1)
            mutated_child2 = self._mutation.mutate(child2)

            improve(mutated_child1.get_cyclic_representation_exposed(), distanceMatrix)
            improve(mutated_child2.get_cyclic_representation_exposed(), distanceMatrix)
            mutated_child1.reset_distance_calc()
            mutated_child2.reset_distance_calc()
            offspring_list[index] = mutated_child1
            offspring_list[index + 1] = mutated_child2
            if DEBUG:
                try:
                    for city in child1():
                        continue
                except CyclicNotationException as e:
                    raise POSCrossoverError("The candidate solution is not a valid cycle")
                try:
                    for city in child2():
                        continue
                except CyclicNotationException as e:
                    raise POSCrossoverError("The candidate solution is not a valid cycle")
        if PRINT_VARIATION:
            print("############################################################################")
            print(f"heuristic:\n\t parent1: {parent1.get_distance()}\n\t parent2: {parent2.get_distance()} \n\t \
                    average_child: {np.average([ind.get_distance()[0] for ind in offspring_list[indexes]])} \
                    min_child: {np.min([ind.get_distance()[0] for ind in offspring_list[indexes]])} \
                    max_child: {np.max([ind.get_distance()[0] for ind in offspring_list[indexes]])} \
                    \n\tchildren: {[ind.get_distance() for ind in offspring_list[indexes]]}")
            print("############################################################################")
    def _create_offspring_POS(self,offspring_list : np.ndarray, indexes: np.ndarray, 
                               parent1: Individual, parent2: Individual, distanceMatrix: np.ndarray, base_case_no_city: np.ndarray, all_cities: np.ndarray):
        
        #assumption: the length of indexes is even, this is enforced in parameters.py (starting position not guraranteed even index)
        
        for index in indexes:
            if offspring_list[index]: continue
            child1, child2 = self._POS(parent1, parent2, distanceMatrix, base_case_no_city, all_cities)
            mutated_child1 = self._mutation.mutate(child1)
            mutated_child2 = self._mutation.mutate(child2)
            improve(mutated_child1.get_cyclic_representation_exposed(), distanceMatrix)
            improve(mutated_child2.get_cyclic_representation_exposed(), distanceMatrix)
            mutated_child1.reset_distance_calc()
            mutated_child2.reset_distance_calc()
            offspring_list[index] = mutated_child1
            offspring_list[index + 1] = mutated_child2
            if DEBUG:
                try:
                    for city in child1():
                        continue
                except CyclicNotationException as e:
                    raise POSCrossoverError("The candidate solution is not a valid cycle")
                try:
                    for city in child2():
                        continue
                except CyclicNotationException as e:
                    raise POSCrossoverError("The candidate solution is not a valid cycle")
        if PRINT_VARIATION:
            print("############################################################################")
            print(f"POS:\n\t parent1: {parent1.get_distance()}\n\t parent2: {parent2.get_distance()} \n\t \
                    average_child: {np.average([ind.get_distance()[0] for ind in offspring_list[indexes]])} \
                    min_child: {np.min([ind.get_distance()[0] for ind in offspring_list[indexes]])} \
                    max_child: {np.max([ind.get_distance()[0] for ind in offspring_list[indexes]])} \
                    \n\tchildren: {[ind.get_distance() for ind in offspring_list[indexes]]}")
            print("############################################################################")
    def _create_offspring_OX1(self,offspring_list : np.ndarray, indexes: np.ndarray, 
                               parent1: Individual, parent2: Individual, distanceMatrix: np.ndarray, base_case_no_city: np.ndarray, all_cities: np.ndarray):
        
        #assumption: the length of indexes is even, this is enforced in parameters.py (starting position not guraranteed even index)
        
        for index in indexes:
            if offspring_list[index]: continue
            child1, child2 = self._OX1(parent1, parent2, distanceMatrix, base_case_no_city, all_cities)
            mutated_child1 = self._mutation.mutate(child1)
            mutated_child2 = self._mutation.mutate(child2)
            improve(mutated_child1.get_cyclic_representation_exposed(), distanceMatrix)
            improve(mutated_child2.get_cyclic_representation_exposed(), distanceMatrix)
            mutated_child1.reset_distance_calc()
            mutated_child2.reset_distance_calc()
            offspring_list[index] = mutated_child1
            offspring_list[index + 1] = mutated_child2
            if DEBUG:
                try:
                    for city in child1():
                        continue
                except CyclicNotationException as e:
                    raise OX1CrossoverError("The candidate solution is not a valid cycle")
                try:
                    for city in child2():
                        continue
                except CyclicNotationException as e:
                    raise OX1CrossoverError("The candidate solution is not a valid cycle")
        if PRINT_VARIATION:
            print("############################################################################")
            print(f"OX1:\n\t parent1: {parent1.get_distance()}\n\t parent2: {parent2.get_distance()} \n\t \
                    average_child: {np.average([ind.get_distance()[0] for ind in offspring_list[indexes]])} \
                    min_child: {np.min([ind.get_distance()[0] for ind in offspring_list[indexes]])} \
                    max_child: {np.max([ind.get_distance()[0] for ind in offspring_list[indexes]])} \
                    \n\tchildren: {[ind.get_distance() for ind in offspring_list[indexes]]}")
            print("############################################################################")
    def _create_offspring_1_1_1(self,offspring_list : np.ndarray, indexes: np.ndarray, 
                               parent1: Individual, parent2: Individual, distanceMatrix: np.ndarray, base_case_no_city: np.ndarray, all_cities: np.ndarray):
        for index in indexes:
            child = self._recombination(parent1, parent2, distanceMatrix, base_case_no_city, all_cities,
                                        p_inherent_common_path = 1,
                                        p_inherent_common_city_in_common_location = 1,
                                        p_inherent_difference_city_in_common_location = 1)
            mutated_child = self._mutation.mutate(child)
            improve(mutated_child.get_cyclic_representation_exposed(), distanceMatrix)
            mutated_child.reset_distance_calc()
            offspring_list[index] = mutated_child
        if PRINT_VARIATION:
            print("############################################################################")
            print(f"111:\n\t parent1: {parent1.get_distance()}\n\t parent2: {parent2.get_distance()} \n\t \
                    average_child: {np.average([ind.get_distance()[0] for ind in offspring_list[indexes]])} \
                    min_child: {np.min([ind.get_distance()[0] for ind in offspring_list[indexes]])} \
                    max_child: {np.max([ind.get_distance()[0] for ind in offspring_list[indexes]])} \
                    \n\tchildren: {[ind.get_distance() for ind in offspring_list[indexes]]}")
            print("############################################################################")
    def _create_offspring_1_1_para(self,offspring_list : np.ndarray, indexes: np.ndarray, 
                               parent1: Individual, parent2: Individual, distanceMatrix: np.ndarray, base_case_no_city: np.ndarray, all_cities: np.ndarray):
        for index in indexes:
            child = self._recombination(parent1, parent2, distanceMatrix, base_case_no_city, all_cities,
                                        p_inherent_common_path = 1,
                                        p_inherent_common_city_in_common_location = 1,
                                        p_inherent_difference_city_in_common_location = self._parameters.get_p_inherent_difference_city_in_common_location())
            mutated_child = self._mutation.mutate(child)
            improve(mutated_child.get_cyclic_representation_exposed(), distanceMatrix)
            mutated_child.reset_distance_calc()
            offspring_list[index] = mutated_child
        if PRINT_VARIATION:
            print("############################################################################")
            print(f"11P:\n\t parent1: {parent1.get_distance()}\n\t parent2: {parent2.get_distance()} \n\t \
                    average_child: {np.average([ind.get_distance()[0] for ind in offspring_list[indexes]])} \
                    min_child: {np.min([ind.get_distance()[0] for ind in offspring_list[indexes]])} \
                    max_child: {np.max([ind.get_distance()[0] for ind in offspring_list[indexes]])} \
                    \n\tchildren: {[ind.get_distance() for ind in offspring_list[indexes]]}")
            print("############################################################################")
    def _create_offspring_1_para_1(self,offspring_list : np.ndarray, indexes: np.ndarray, 
                               parent1: Individual, parent2: Individual, distanceMatrix: np.ndarray, base_case_no_city: np.ndarray, all_cities: np.ndarray):
        for index in indexes:
            child = self._recombination(parent1, parent2, distanceMatrix, base_case_no_city, all_cities,
                                        p_inherent_common_path = 1,
                                        p_inherent_common_city_in_common_location = self._parameters.get_p_inherent_common_city_in_common_location(),
                                        p_inherent_difference_city_in_common_location = 1)
            mutated_child = self._mutation.mutate(child)
            improve(mutated_child.get_cyclic_representation_exposed(), distanceMatrix)
            mutated_child.reset_distance_calc()
            offspring_list[index] = mutated_child
        if PRINT_VARIATION:
            print("############################################################################")
            print(f"1P1:\n\t parent1: {parent1.get_distance()}\n\t parent2: {parent2.get_distance()} \n\t \
                    average_child: {np.average([ind.get_distance()[0] for ind in offspring_list[indexes]])} \
                    min_child: {np.min([ind.get_distance()[0] for ind in offspring_list[indexes]])} \
                    max_child: {np.max([ind.get_distance()[0] for ind in offspring_list[indexes]])} \
                    \n\tchildren: {[ind.get_distance() for ind in offspring_list[indexes]]}")
            print("############################################################################")
    def _create_offspring_1_para_para(self,offspring_list : np.ndarray, indexes: np.ndarray, 
                               parent1: Individual, parent2: Individual, distanceMatrix: np.ndarray, base_case_no_city: np.ndarray, all_cities: np.ndarray):
        for index in indexes:
            child = self._recombination(parent1, parent2, distanceMatrix, base_case_no_city, all_cities,
                                        p_inherent_common_path = 1,
                                        p_inherent_common_city_in_common_location = self._parameters.get_p_inherent_common_city_in_common_location(),
                                        p_inherent_difference_city_in_common_location = self._parameters.get_p_inherent_difference_city_in_common_location())
            mutated_child = self._mutation.mutate(child)
            improve(mutated_child.get_cyclic_representation_exposed(), distanceMatrix)
            mutated_child.reset_distance_calc()
            offspring_list[index] = mutated_child
        if PRINT_VARIATION:
            print("############################################################################")
            print(f"1PP:\n\t parent1: {parent1.get_distance()}\n\t parent2: {parent2.get_distance()} \n\t \
                    average_child: {np.average([ind.get_distance()[0] for ind in offspring_list[indexes]])} \
                    min_child: {np.min([ind.get_distance()[0] for ind in offspring_list[indexes]])} \
                    max_child: {np.max([ind.get_distance()[0] for ind in offspring_list[indexes]])} \
                    \n\tchildren: {[ind.get_distance() for ind in offspring_list[indexes]]}")
            print("############################################################################")
    def _create_offspring_para_para_para(self,offspring_list : np.ndarray, indexes: np.ndarray, 
                               parent1: Individual, parent2: Individual, distanceMatrix: np.ndarray, base_case_no_city: np.ndarray, all_cities: np.ndarray):
        for index in indexes:
            child = self._recombination(parent1, parent2, distanceMatrix, base_case_no_city, all_cities,
                                        p_inherent_common_path = self._parameters.get_p_inherent_common_path(),
                                        p_inherent_common_city_in_common_location = self._parameters.get_p_inherent_common_city_in_common_location(),
                                        p_inherent_difference_city_in_common_location = self._parameters.get_p_inherent_difference_city_in_common_location())
            mutated_child = self._mutation.mutate(child)
            improve(mutated_child.get_cyclic_representation_exposed(), distanceMatrix)
            mutated_child.reset_distance_calc()
            offspring_list[index] = mutated_child
        if PRINT_VARIATION:
            print("############################################################################")
            print(f"PPP:\n\t parent1: {parent1.get_distance()}\n\t parent2: {parent2.get_distance()} \n\t \
                    average_child: {np.average([ind.get_distance()[0] for ind in offspring_list[indexes]])} \
                    min_child: {np.min([ind.get_distance()[0] for ind in offspring_list[indexes]])} \
                    max_child: {np.max([ind.get_distance()[0] for ind in offspring_list[indexes]])} \
                    \n\tchildren: {[ind.get_distance() for ind in offspring_list[indexes]]}")
            print("############################################################################")
    def _find_first_non_zero(self,array: np.ndarray, comparison_operator: Callable[[int], bool]) -> int:

        for value in array:
            if comparison_operator(value): return value
        return -1
        
    def _heuristic_crossover_with_common_subtour(self, parent1: Individual, parent2: Individual, distanceMatrix: np.ndarray, ordered_arg_distance_matrix: np.ndarray, base_case_no_city: np.ndarray, all_cities: np.ndarray,
                ) -> Individual:

        base_case_no_city = np.copy(base_case_no_city)
        all_cities = np.copy(all_cities)

        child_cyclic_representation = np.where(parent1.get_cyclic_representation_exposed() == parent2.get_cyclic_representation_exposed(), parent1.get_cyclic_representation_exposed(), base_case_no_city)
        child = Individual(parent1.get_number_of_cities(), distanceMatrix)
        child.use_cyclic_notation(child_cyclic_representation)


        cities_left_to_allocate = np.where(~np.isin(all_cities,child_cyclic_representation), all_cities, base_case_no_city )
        #get a random starting city that is not been allocated yet
        # needs to be random in case multiple offspring are generated, the rest of the process is deterministic
        start_city = self._find_first_non_zero(np.random.permutation(cities_left_to_allocate), lambda x: x != -1) 

        if start_city == -1:
            raise ParentsException("The parents were equal, this code should not have been executed")
        cities_left_to_allocate[start_city] = -1 

        self._build_heuristic_cycle_exposed(start_city, child, parent1, parent2, cities_left_to_allocate, distanceMatrix, ordered_arg_distance_matrix)

        if DEBUG:
            assert((cities_left_to_allocate == -1).all())
            
        return child
    
    def _heuristic_crossover(self, parent1: Individual, parent2: Individual, distanceMatrix: np.ndarray, ordered_arg_distance_matrix: np.ndarray, base_case_no_city: np.ndarray, all_cities: np.ndarray,
                ) -> Individual:


    
        base_case_no_city = np.copy(base_case_no_city)
        all_cities = np.copy(all_cities)

        child = Individual(parent1.get_number_of_cities(), distanceMatrix)
        
        cities_left_to_allocate = np.copy(all_cities)
        #get a random starting city that is not been allocated yet
        # needs to be random in case multiple offspring are generated, the rest of the process is deterministic
        start_city = np.random.choice(cities_left_to_allocate)
        cities_left_to_allocate[start_city] = -1 

        self._build_heuristic_cycle_exposed(start_city, child, parent1, parent2, cities_left_to_allocate, distanceMatrix, ordered_arg_distance_matrix)

    
        if DEBUG:
            assert((cities_left_to_allocate == -1).all())
            
        return child

    def _POS(self, parent1: Individual, parent2: Individual, distanceMatrix: np.ndarray, base_case_no_city: np.ndarray, all_cities: np.ndarray,
                ) -> Tuple[Individual, Individual]:
        
       
        base_case_no_city = np.copy(base_case_no_city)
        all_cities = np.copy(all_cities)
        #all_cities[1:] because 0 is implied constant and can't be assigned again
        random_indexes = np.sort(np.random.choice(all_cities[1:], self._parameters.get_number_of_random_fixed_positions_POS(), replace=False))
        fixed_values_parent1 = parent1.index_with_list(random_indexes)
        fixed_values_parent2 = parent2.index_with_list(random_indexes)

        child1 = Individual(parent1.get_number_of_cities(),distanceMatrix)
        child2 = Individual(parent2.get_number_of_cities(), distanceMatrix)
    
        #start from 1 as the the beginning city should only be allocated last (additional constrain on beginning city)
        cities_left_to_allocate_parent1 = np.where(~np.isin(all_cities,fixed_values_parent2), all_cities, base_case_no_city )
        cities_left_to_allocate_parent2 = np.where(~np.isin(all_cities,fixed_values_parent1), all_cities, base_case_no_city )             
    
        start_city1 = 0
        start_city2 = 0
        
        cities_left_to_allocate_parent1[start_city2] = -1
        cities_left_to_allocate_parent2[start_city1] = -1

        self._build_POS_cycle_exposed(start_city1,start_city2, child1, child2, parent1, parent2,
                                    cities_left_to_allocate_parent1, cities_left_to_allocate_parent2,
                                    random_indexes, fixed_values_parent1, fixed_values_parent2)
            
          

        if DEBUG:
            assert((cities_left_to_allocate_parent1 == -1).all())
            assert((cities_left_to_allocate_parent2 == -1).all())
            
        return child1, child2

    

    #you cannat use commonsubpath as the offspring will regenerate the parents
    def _OX1(self, parent1: Individual, parent2: Individual, distanceMatrix: np.ndarray, base_case_no_city: np.ndarray, all_cities: np.ndarray,
                ) -> Tuple[Individual, Individual]:
        
       
        base_case_no_city = np.copy(base_case_no_city)
        all_cities = np.copy(all_cities)

        index1 = np.random.choice(all_cities)
        index2 = index1 + self._parameters.get_OX1_constant_subpath_lenght()

        start_city_sequence1, child1_cyclic_representation, end_city_sequence1 = parent1.get_path_between_indexes_cyclic_format(index1, index2)
        start_city_sequence2, child2_cyclic_representation, end_city_sequence2 = parent2.get_path_between_indexes_cyclic_format(index1, index2)
        child1 = Individual(parent1.get_number_of_cities(), distanceMatrix)
        child1.use_cyclic_notation(child1_cyclic_representation)
        child2 = Individual(parent1.get_number_of_cities(), distanceMatrix)
        child2.use_cyclic_notation(child2_cyclic_representation)
        #start from 1 as the the beginning city should only be allocated last (additional constrain on beginning city)
        cities_left_to_allocate_parent2 = np.where(~np.isin(all_cities,child1_cyclic_representation), all_cities, base_case_no_city )
        cities_left_to_allocate_parent1 = np.where(~np.isin(all_cities,child2_cyclic_representation), all_cities, base_case_no_city )

        cities_left_to_allocate_parent1[start_city_sequence2] = -1
        cities_left_to_allocate_parent2[start_city_sequence1] = -1
        
        
        self._build_0X1_cycle_exposed(start_city_sequence1, start_city_sequence2 , end_city_sequence1, end_city_sequence2, child1, child2, parent1, parent2, 
                                                    cities_left_to_allocate_parent1, cities_left_to_allocate_parent2)                             

            

        if DEBUG:
            assert((cities_left_to_allocate_parent1 == -1).all())
            assert((cities_left_to_allocate_parent2 == -1).all())
            
        return child1, child2
    
    def _recombination(self, parent1: Individual, parent2: Individual, distanceMatrix: np.ndarray, base_case_no_city: np.ndarray, all_cities: np.ndarray,
                        p_inherent_common_path: float = 1, 
                        p_inherent_common_city_in_common_location: float = 0.6,
                        p_inherent_difference_city_in_common_location: float = 0.6                       
                        ) -> Individual:
       
        base_case_no_city = np.copy(base_case_no_city)
        all_cities = np.copy(all_cities)
        child_cyclic_representation = np.where(parent1.get_cyclic_representation_exposed() == parent2.get_cyclic_representation_exposed(), parent1.get_cyclic_representation_exposed(), base_case_no_city)
        child = Individual(parent1.get_number_of_cities(), distanceMatrix)
        child.use_cyclic_notation(child_cyclic_representation)
        #start from 1 as the the beginning city should only be allocated last (additional constrain on beginning city)
        cities_left_to_allocate = np.where(~np.isin(all_cities,child_cyclic_representation), all_cities, base_case_no_city )

        #check for -1 when parents are identical match
        start_city = self._find_first_non_zero(cities_left_to_allocate, lambda x: x != -1)
        if start_city == -1:
            raise ParentsException("The parents were equal, this code should not have been executed")
        cities_left_to_allocate[start_city] = -1 
    

        build_hamiltonian_cycle_exposed(start_city, child.get_cyclic_representation_exposed(), parent1.get_cyclic_representation_exposed(), parent2.get_cyclic_representation_exposed(), cities_left_to_allocate,
                                                self._parameters.get_number_of_cities(),
                                                p_inherent_common_path,
                                                p_inherent_common_city_in_common_location,
                                                p_inherent_difference_city_in_common_location)
       

        if DEBUG:
            assert((cities_left_to_allocate == -1).all())
            try:
                for city in child():
                    continue
            except CyclicNotationException as e:
                raise edgeCrossoverError("The candidate solution is not a valid cycle")
        return child

    def _get_city(self, start_city: int, child: Individual, city_parent1: int, city_parent2: int, cities_left_to_allocate: np.ndarray, 
                  p_inherent_common_city_in_common_location: float,
                  p_inherent_difference_city_in_common_location: float
                  ):
        # last city goes back to the start
        if (cities_left_to_allocate == -1).all():
            return start_city
        
        p_choice = np.random.random_sample()

        #same city on same place in both parents
        if p_choice <= p_inherent_common_city_in_common_location \
            and city_parent1 == city_parent2:
            if cities_left_to_allocate[city_parent1] != -1 : 
                return self._select_choosen_city(city_parent1, cities_left_to_allocate)
            else:
                return self._get_random_city(cities_left_to_allocate)
        #different city in both parents, but still select from parents
        elif p_choice <= p_inherent_difference_city_in_common_location \
            and city_parent1 != city_parent2:
            can_allocate_city_parent1 = cities_left_to_allocate[city_parent1] != -1
            can_allocate_city_parent2 = cities_left_to_allocate[city_parent2] != -1
            if can_allocate_city_parent1 and can_allocate_city_parent2:
                p_choice_between_parents = np.random.random_sample()
                if p_choice_between_parents <= 0.5:
                    return self._select_choosen_city(city_parent1, cities_left_to_allocate)
                else:
                    return self._select_choosen_city(city_parent2, cities_left_to_allocate)
            elif can_allocate_city_parent1:
                return self._select_choosen_city(city_parent1, cities_left_to_allocate)
            elif can_allocate_city_parent2:
                return self._select_choosen_city(city_parent2, cities_left_to_allocate)
            else:
                return self._get_random_city(cities_left_to_allocate)
        #just random sample
        else:
            return self._get_random_city(cities_left_to_allocate)
    
    def _get_random_city(self, cities_left_to_allocate: np.ndarray) -> int:
        return get_random_city(cities_left_to_allocate)
    
    def _select_choosen_city(self, city: int, cities_left_to_allocate: np.ndarray) -> int:
        return select_choosen_city(city, cities_left_to_allocate)
    
    def _build_hamiltonian_cycle_exposed(self,start_city: int, child: Individual, parent1: Individual, parent2: Individual, cities_left_to_allocate: np.ndarray,
                                         p_inherent_common_path: float,
                                         p_inherent_common_city_in_common_location: float,
                                         p_inherent_difference_city_in_common_location: float
                                         ) -> None:
        # assumption: start city is still in need of allocation
        p_choice = np.random.random_sample()

        for city_parent1, city_parent2, city_child in zip_longest(parent1(start_city), parent2(start_city), child(start_city)):
            if city_child != -1:
                if p_choice > p_inherent_common_path:
                    cities_left_to_allocate[city_child] = city_child
                    child.iterator_set(self._get_city(start_city, child, city_parent1, city_parent2, cities_left_to_allocate,
                                 p_inherent_common_city_in_common_location,
                                 p_inherent_difference_city_in_common_location))
                    continue
                else: 
                    continue
            else:
                p_choice = np.random.random_sample()
                child.iterator_set(self._get_city(start_city, child, city_parent1, city_parent2, cities_left_to_allocate,
                                 p_inherent_common_city_in_common_location,
                                 p_inherent_difference_city_in_common_location))
        child.check_reverse()

    def _build_0X1_cycle_exposed(self,start_city_sequence1: int, start_city_sequence2 : int, end_city_sequence1: int, end_city_sequence2: int, 
                                 child1: Individual, child2: Individual, parent1: Individual, parent2: Individual, 
                                cities_left_to_allocate_parent1: np.ndarray, cities_left_to_allocate_parent2: np.ndarray):                             
        
        # assumption: the constant subpath is on equal position of both children
        # assumption: start city is first city of constant subpath
        build_0X1_cycle_exposed(start_city_sequence1, start_city_sequence2, end_city_sequence1, end_city_sequence2, child1.get_cyclic_representation_exposed(),
                                child2.get_cyclic_representation_exposed(), parent1.get_cyclic_representation_exposed(), parent2.get_cyclic_representation_exposed(),
                                cities_left_to_allocate_parent1, cities_left_to_allocate_parent2, self._parameters.get_number_of_cities())
        child1.check_reverse()
        child2.check_reverse()
            
    def _build_POS_cycle_exposed(self,start_city1: int, start_city2:int, child1: Individual, child2: Individual, parent1: Individual, parent2: Individual, 
                                    cities_left_to_allocate_parent1: np.ndarray, cities_left_to_allocate_parent2: np.ndarray, 
                                    random_indexes: np.ndarray, fixed_values_parent1: np.ndarray, fixed_values_parent2: np.ndarray):                             
        # assumption: the constant subpath is on equal position of both children
        # assumption: start city is first city of constant subpath
        # choice: we start indexing from 0 -> ... (start_city == 0)
        build_POS_cycle_exposed(start_city1, start_city2, child1.get_cyclic_representation_exposed(), child2.get_cyclic_representation_exposed(), 
                                parent1.get_cyclic_representation_exposed(), parent2.get_cyclic_representation(),
                                cities_left_to_allocate_parent1, cities_left_to_allocate_parent2, random_indexes, fixed_values_parent1, fixed_values_parent2,
                                self._parameters.get_number_of_cities())
        child1.check_reverse()
        child2.check_reverse()

    def _build_heuristic_cycle_exposed(self, start_city: int, child: Individual, parent1: Individual, parent2: Individual, cities_left_to_allocate: np.ndarray,
                                        distance_matrix: np.ndarray, ordered_arg_distance_matrix: np.ndarray
    ):
   
        build_heuristic_cycle_exposed(start_city, child.get_cyclic_representation_exposed(), parent1.get_cyclic_representation_exposed(), 
                                      parent2.get_cyclic_representation_exposed(),cities_left_to_allocate,distance_matrix, ordered_arg_distance_matrix,
                                       self._parameters.get_number_of_cities() )

        child.check_reverse()

@jit(int16(int16, int16[:]), nopython = True)
def select_choosen_city(city: int, cities_left_to_allocate: np.ndarray) -> int:
    cities_left_to_allocate[city] = -1
    return city

@jit(int16(int16[:]),nopython = True)
def get_random_city(cities_left_to_allocate: np.ndarray) -> int:
        city = np.random.choice(cities_left_to_allocate[cities_left_to_allocate != -1], 1)[0]
        return select_choosen_city(city,cities_left_to_allocate)
    

"""
    previous_child_city = start_city
    representation = child.get_cyclic_representation()
    representation_parent1 = parent1.get_cyclic_representation_exposed()
    representation_parent2 = parent2.get_cyclic_representation_exposed()
    number_of_cities 
"""

@jit((int16, int16[:], int16[:], int16[:], int16[:], float64[:,:], int16[:,:], int16),nopython = True)
def build_heuristic_cycle_exposed( start_city: int, representation: np.ndarray, representation_parent1: np.ndarray, representation_parent2: np.ndarray, cities_left_to_allocate: np.ndarray,
                                        distance_matrix: np.ndarray, ordered_arg_distance_matrix: np.ndarray, number_of_cities: int
    ):
        
        previous_child_city = start_city
        counter = 0
        while counter < number_of_cities:
            city_child = representation[previous_child_city]
            if city_child !=-1:
                previous_child_city = city_child
                counter +=1
                continue
            
            if (cities_left_to_allocate == -1).all():
                representation[previous_child_city] = select_choosen_city(start_city, cities_left_to_allocate)
                break

            next_city_parent1 = representation_parent1[previous_child_city]
            next_city_parent2 = representation_parent2[previous_child_city]


            can_allocate_city_parent1 = cities_left_to_allocate[next_city_parent1] != -1
            can_allocate_city_parent2 = cities_left_to_allocate[next_city_parent2] != -1
            distance1 = distance_matrix[previous_child_city, next_city_parent1]
            distance2 = distance_matrix[previous_child_city, next_city_parent2]
            if can_allocate_city_parent1 and can_allocate_city_parent2:
                if distance1 <= distance2 and not np.isinf(distance1): #== should not happen if common edges are already defined
                    representation[previous_child_city] = select_choosen_city(next_city_parent1, cities_left_to_allocate)

                elif not np.isinf(distance2):
                    representation[previous_child_city] = select_choosen_city(next_city_parent2, cities_left_to_allocate)

            elif can_allocate_city_parent1 and not np.isinf(distance1):
                    representation[previous_child_city] = select_choosen_city(next_city_parent1, cities_left_to_allocate)

            elif can_allocate_city_parent2 and not np.isinf(distance2):
                    representation[previous_child_city] = select_choosen_city(next_city_parent2, cities_left_to_allocate)


            else:
                for candidate_city in ordered_arg_distance_matrix[previous_child_city][1:]: #first element is back to itself as d(t1, t1) == 0
                    if cities_left_to_allocate[candidate_city] != -1:
                        representation[previous_child_city] = select_choosen_city(candidate_city, cities_left_to_allocate)
                        break

            previous_child_city = representation[previous_child_city]
            counter +=1

        """
        for city_child in child(start_city):
            if city_child !=-1:
                previous_child_city = city_child
                continue
            if (cities_left_to_allocate == -1).all():
                child.iterator_set(self._select_choosen_city(start_city, cities_left_to_allocate))
                break

            next_city_parent1 = parent1.find_next_city(previous_child_city)
            next_city_parent2 = parent2.find_next_city(previous_child_city)

            
            can_allocate_city_parent1 = cities_left_to_allocate[next_city_parent1] != -1
            can_allocate_city_parent2 = cities_left_to_allocate[next_city_parent2] != -1
            distance1 = distance_matrix[previous_child_city, next_city_parent1]
            distance2 = distance_matrix[previous_child_city, next_city_parent2]
            if can_allocate_city_parent1 and can_allocate_city_parent2:
                if distance1 <= distance2 and not np.isinf(distance1): #== should not happen if common edges are already defined
                    child.iterator_set(self._select_choosen_city(next_city_parent1, cities_left_to_allocate))
                    previous_child_city = next_city_parent1
                elif not np.isinf(distance2):
                    child.iterator_set(self._select_choosen_city(next_city_parent2, cities_left_to_allocate))
                    previous_child_city = next_city_parent2
            elif can_allocate_city_parent1 and not np.isinf(distance1):
                    child.iterator_set(self._select_choosen_city(next_city_parent1, cities_left_to_allocate))
                    previous_child_city = next_city_parent1
            elif can_allocate_city_parent2 and not np.isinf(distance2):
                    child.iterator_set(self._select_choosen_city(next_city_parent2, cities_left_to_allocate))
                    previous_child_city = next_city_parent2

            else:
                for candidate_city in ordered_arg_distance_matrix[previous_child_city][1:]: #first element is back to itself as d(t1, t1) == 0
    
                    if cities_left_to_allocate[candidate_city] != -1:
                        child.iterator_set(self._select_choosen_city(candidate_city, cities_left_to_allocate))
                        previous_child_city = candidate_city
                        break

        child.check_reverse()
        """


"""
    representation_child1 = child1.get_cyclic_representation_exposed()
    representation_child2 = child2.get_cyclic_representation_exposed()
    representation_parent1 = parent1.get_cyclic_representation_exposed()
    representation_parent2 = parent2.get_cyclic_representation_exposed()
    number_of_cities = 50
"""

@jit((int16, int16, int16[:], int16[:], int16[:], int16[:], int16[:], int16[:], int16[:], int16[:], int16[:], int16),nopython = True)
def build_POS_cycle_exposed(start_city1: int, start_city2:int, representation_child1: np.ndarray, representation_child2: np.ndarray,
                                representation_parent1: np.ndarray, representation_parent2: np.ndarray, 
                                cities_left_to_allocate_parent1: np.ndarray, cities_left_to_allocate_parent2: np.ndarray, 
                                random_indexes: np.ndarray, fixed_values_parent1: np.ndarray, fixed_values_parent2: np.ndarray,
                                number_of_cities: int):                             
        # assumption: the constant subpath is on equal position of both children
        # assumption: start city is first city of constant subpath
        # choice: we start indexing from 0 -> ... (start_city == 0)

        previous_child_city1 = start_city1
        previous_child_city2 = start_city2
        previous_parent_city1 = start_city1
        previous_parent_city2 = start_city2

        counter = 1
        index_random_indexes = 0
        while counter < number_of_cities + 1: 
            city1_child = representation_child1[previous_child_city1]
            city2_child = representation_child2[previous_child_city2]


            if index_random_indexes < random_indexes.size and counter == random_indexes[index_random_indexes]:
                representation_child1[previous_child_city1] = select_choosen_city(fixed_values_parent1[index_random_indexes], cities_left_to_allocate_parent2)
                representation_child2[previous_child_city2] = select_choosen_city(fixed_values_parent2[index_random_indexes], cities_left_to_allocate_parent1)
                index_random_indexes +=1
            else:
                if (cities_left_to_allocate_parent1 == -1).all() and (cities_left_to_allocate_parent2 == -1).all():
                    representation_child1[previous_child_city1] = select_choosen_city(start_city1, cities_left_to_allocate_parent2)
                    representation_child2[previous_child_city2] = select_choosen_city(start_city2, cities_left_to_allocate_parent1)
                    break

                if DEBUG and ( city1_child != -1 or city2_child != -1 ): raise OX1Exception("Something went wrong when making POS child. We are not expecting cities to be allocated yet before loop arrives")

                city_parent1 = representation_parent1[previous_parent_city1]
                previous_parent_city1 = city_parent1
                while cities_left_to_allocate_parent1[city_parent1] == -1:
                    city_parent1 = representation_parent1[previous_parent_city1]
                    previous_parent_city1 = city_parent1

                city_parent2 = representation_parent2[previous_parent_city2]
                previous_parent_city2 = city_parent2
                while cities_left_to_allocate_parent2[city_parent2] == -1:
                    city_parent2 = representation_parent2[previous_parent_city2]
                    previous_parent_city2 = city_parent2

                representation_child1[previous_child_city1] = select_choosen_city(city_parent2, cities_left_to_allocate_parent2) #type: ignore
                representation_child2[previous_child_city2] =  select_choosen_city(city_parent1, cities_left_to_allocate_parent1) #type: ignore

            previous_child_city1 = representation_child1[previous_child_city1]
            previous_child_city2 = representation_child2[previous_child_city2]

            counter +=1

        """
        iterator_parent1 = iter(parent1(start_city1))
        iterator_parent2 = iter(parent2(start_city2))
        done = False
        counter = 1
        index_random_indexes = 0
        for city1_child, city2_child in zip_longest(child1(start_city1), child2(start_city2)):
            if index_random_indexes < random_indexes.size and counter == random_indexes[index_random_indexes]:
                child1.iterator_set(self._select_choosen_city(fixed_values_parent1[index_random_indexes], cities_left_to_allocate_parent2))
                child2.iterator_set(self._select_choosen_city(fixed_values_parent2[index_random_indexes], cities_left_to_allocate_parent1))
                index_random_indexes +=1
            else:
                if (cities_left_to_allocate_parent1 == -1).all() and (cities_left_to_allocate_parent2 == -1).all():
                    child1.iterator_set(self._select_choosen_city(start_city1, cities_left_to_allocate_parent2))
                    child2.iterator_set(self._select_choosen_city(start_city2, cities_left_to_allocate_parent1))
                    done = True
                    break

                if done: raise POSException("Something went wrong, process should have been terminated")

                if city1_child != -1 or city2_child != -1 : raise OX1Exception("Something went wrong when making POS child. We are not expecting cities to be allocated yet before loop arrives")
            
                city_parent1 = next(iterator_parent1)
                while cities_left_to_allocate_parent1[city_parent1] == -1:
                    city_parent1 = next(iterator_parent1)

                city_parent2 = next(iterator_parent2)
                while cities_left_to_allocate_parent2[city_parent2] == -1:
                    city_parent2 = next(iterator_parent2)

                child1.iterator_set(self._select_choosen_city(city_parent2, cities_left_to_allocate_parent2))
                child2.iterator_set(self._select_choosen_city(city_parent1, cities_left_to_allocate_parent1))
            counter +=1
        child1.check_reverse()
        child2.check_reverse()   
        
        """

        


"""
    representation_child1 = child1.get_cyclic_representation_exposed()
    representation_child2 = child2.get_cyclic_representation_exposed()
    representation_parent1 = parent1.get_cyclic_representation_exposed()
    representation_parent2 = parent2.get_cyclic_representation_exposed()
    number_of_cities = 50
"""
            
@jit((int16, int16, int16, int16, int16[:], int16[:], int16[:], int16[:], int16[:], int16[:],int16),nopython = True)
def build_0X1_cycle_exposed(start_city_sequence1: int, start_city_sequence2 : int, end_city_sequence1: int, end_city_sequence2: int, 
                                 representation_child1: np.ndarray, representation_child2: np.ndarray, representation_parent1: np.ndarray, 
                                 representation_parent2: np.ndarray, cities_left_to_allocate_parent1: np.ndarray, cities_left_to_allocate_parent2: np.ndarray,
                                 number_of_cities:int):                             
        
        # assumption: the constant subpath is on equal position of both children
        # assumption: start city is first city of constant subpath

        previous_child_city1 = end_city_sequence1
        previous_child_city2 = end_city_sequence2
        previous_parent_city1 = end_city_sequence1
        previous_parent_city2 = end_city_sequence2

        

        counter = 0
        while counter < number_of_cities: 
            city1_child = representation_child1[previous_child_city1]
            city2_child = representation_child2[previous_child_city2]
            if (cities_left_to_allocate_parent1 == -1).all() and (cities_left_to_allocate_parent2 == -1).all():
                representation_child1[previous_child_city1] = select_choosen_city(start_city_sequence1, cities_left_to_allocate_parent2)
                representation_child2[previous_child_city2] = select_choosen_city(start_city_sequence2, cities_left_to_allocate_parent1)
               
                break
            
            if city1_child != -1 and city2_child != -1: 
                previous_child_city1 = representation_child1[previous_child_city1]
                previous_child_city2 = representation_child2[previous_child_city2]
                counter +=1
                continue
            if DEBUG and (city1_child != -1 or city2_child != -1 ): raise OX1Exception("Something went wrong when making OX1 child. We expect equal length, equal position constant subpaths")
            
            city_parent1 = representation_parent1[previous_parent_city1]
            previous_parent_city1 = city_parent1
            while cities_left_to_allocate_parent1[city_parent1] == -1:
                city_parent1 = representation_parent1[previous_parent_city1]
                previous_parent_city1 = city_parent1

            city_parent2 = representation_parent2[previous_parent_city2]
            previous_parent_city2 = city_parent2
            while cities_left_to_allocate_parent2[city_parent2] == -1:
                city_parent2 = representation_parent2[previous_parent_city2]
                previous_parent_city2 = city_parent2
            
            representation_child1[previous_child_city1] = select_choosen_city(city_parent2, cities_left_to_allocate_parent2) #type: ignore
            representation_child2[previous_child_city2] = select_choosen_city(city_parent1, cities_left_to_allocate_parent1) #type: ignore
            
            previous_child_city1 = representation_child1[previous_child_city1]
            previous_child_city2 = representation_child2[previous_child_city2]
            
            counter +=1

        """
        for city1_child, city2_child in zip_longest(child1(end_city_sequence1), child2(end_city_sequence2)):
            if (cities_left_to_allocate_parent1 == -1).all() and (cities_left_to_allocate_parent2 == -1).all():
                child1.iterator_set(self._select_choosen_city(start_city_sequence1, cities_left_to_allocate_parent2))
                child2.iterator_set(self._select_choosen_city(start_city_sequence2, cities_left_to_allocate_parent1))
                done = True
                break

            if done: raise OX1Exception("Something went wrong, process should have been terminated")

            if city1_child != -1 and city2_child != -1: continue
            if city1_child != -1 or city2_child != -1 : raise OX1Exception("Something went wrong when making OX1 child. We expect equal length, equal position constant subpaths")
            
            city_parent1 = next(iterator_parent1)
            while cities_left_to_allocate_parent1[city_parent1] == -1:
                city_parent1 = next(iterator_parent1)
            city_parent2 = next(iterator_parent2)

            while cities_left_to_allocate_parent2[city_parent2] == -1:
                city_parent2 = next(iterator_parent2)

            child1.iterator_set(self._select_choosen_city(city_parent2, cities_left_to_allocate_parent2))
            child2.iterator_set(self._select_choosen_city(city_parent1, cities_left_to_allocate_parent1))
        child1.check_reverse()
        child2.check_reverse()
        """


@jit(int16(int16, int16, int16, int16[:], float64, float64),nopython = True)
def get_city(start_city: int, city_parent1: int, city_parent2: int, cities_left_to_allocate: np.ndarray, 
                  p_inherent_common_city_in_common_location: float,
                  p_inherent_difference_city_in_common_location: float
                  ) -> int:
        # last city goes back to the start
        if (cities_left_to_allocate == -1).all():
            return start_city
        
        p_choice = np.random.random_sample()

        #same city on same place in both parents
        if p_choice <= p_inherent_common_city_in_common_location \
            and city_parent1 == city_parent2:
            if cities_left_to_allocate[city_parent1] != -1 : 
                return city_parent1
            else:
                return get_random_city(cities_left_to_allocate)
        #different city in both parents, but still select from parents
        elif p_choice <= p_inherent_difference_city_in_common_location \
            and city_parent1 != city_parent2:
            can_allocate_city_parent1 = cities_left_to_allocate[city_parent1] != -1
            can_allocate_city_parent2 = cities_left_to_allocate[city_parent2] != -1
            if can_allocate_city_parent1 and can_allocate_city_parent2:
                p_choice_between_parents = np.random.random_sample()
                if p_choice_between_parents <= 0.5:
                    return city_parent1
                else:
                    return city_parent2
            elif can_allocate_city_parent1:
                return city_parent1
            elif can_allocate_city_parent2:
                return city_parent2
            else:
                return get_random_city(cities_left_to_allocate)
        #just random sample
        else:
            return get_random_city(cities_left_to_allocate)



        """
        # last city goes back to the start
        if (cities_left_to_allocate == -1).all():
            return start_city
        
        p_choice = np.random.random_sample()

        #same city on same place in both parents
        if p_choice <= p_inherent_common_city_in_common_location \
            and city_parent1 == city_parent2:
            if cities_left_to_allocate[city_parent1] != -1 : 
                return self._select_choosen_city(city_parent1, cities_left_to_allocate)
            else:
                return self._get_random_city(cities_left_to_allocate)
        #different city in both parents, but still select from parents
        elif p_choice <= p_inherent_difference_city_in_common_location \
            and city_parent1 != city_parent2:
            can_allocate_city_parent1 = cities_left_to_allocate[city_parent1] != -1
            can_allocate_city_parent2 = cities_left_to_allocate[city_parent2] != -1
            if can_allocate_city_parent1 and can_allocate_city_parent2:
                p_choice_between_parents = np.random.random_sample()
                if p_choice_between_parents <= 0.5:
                    return self._select_choosen_city(city_parent1, cities_left_to_allocate)
                else:
                    return self._select_choosen_city(city_parent2, cities_left_to_allocate)
            elif can_allocate_city_parent1:
                return self._select_choosen_city(city_parent1, cities_left_to_allocate)
            elif can_allocate_city_parent2:
                return self._select_choosen_city(city_parent2, cities_left_to_allocate)
            else:
                return self._get_random_city(cities_left_to_allocate)
        #just random sample
        else:
            return self._get_random_city(cities_left_to_allocate)
        """



"""
    representation_child = child.get_cyclic_representation_exposed()
    representation_parent1  = parent1.get_cyclic_representation_exposed()
    representation_parent2 = parent2.get_cyclic_representation_exposed()
"""

@jit((int16, int16[:], int16[:], int16[:], int16[:], int16, float64, float64, float64),nopython = True)
def build_hamiltonian_cycle_exposed(start_city: int, representation_child: np.ndarray, representation_parent1: np.ndarray, representation_parent2: np.ndarray, 
                                        cities_left_to_allocate: np.ndarray,
                                        number_of_cities: int,
                                        p_inherent_common_path: float,
                                        p_inherent_common_city_in_common_location: float,
                                        p_inherent_difference_city_in_common_location: float
                                         ) -> None:
        # assumption: start city is still in need of allocation
        p_choice = np.random.random_sample()

        previous_city_child = start_city
        previous_parent1_child = start_city
        previous_parent2_child = start_city
   

        counter = 0
        while counter < number_of_cities:
            city_child = representation_child[previous_city_child]
            city_parent1 = representation_parent1[previous_parent1_child]
            city_parent2 = representation_parent2[previous_parent2_child]
            if city_child != -1:
                if p_choice > p_inherent_common_path:
                    cities_left_to_allocate[city_child] = city_child
                    representation_child[previous_city_child] = select_choosen_city (get_city(start_city, city_parent1, city_parent2, cities_left_to_allocate,
                                 p_inherent_common_city_in_common_location,
                                 p_inherent_difference_city_in_common_location), cities_left_to_allocate)
            else:
                p_choice = np.random.random_sample()
                representation_child[previous_city_child] = select_choosen_city (get_city(start_city, city_parent1, city_parent2, cities_left_to_allocate,
                                 p_inherent_common_city_in_common_location,
                                 p_inherent_difference_city_in_common_location), cities_left_to_allocate)
            previous_city_child = representation_child[previous_city_child]   
            counter +=1

        """
        for city_parent1, city_parent2, city_child in zip_longest(parent1(start_city), parent2(start_city), child(start_city)):
            if city_child != -1:
                if p_choice > p_inherent_common_path:
                    cities_left_to_allocate[city_child] = city_child
                    child.iterator_set(self._get_city(start_city, child, city_parent1, city_parent2, cities_left_to_allocate,
                                 p_inherent_common_city_in_common_location,
                                 p_inherent_difference_city_in_common_location))
                    continue
                else: 
                    continue
            else:
                p_choice = np.random.random_sample()
                child.iterator_set(self._get_city(start_city, child, city_parent1, city_parent2, cities_left_to_allocate,
                                 p_inherent_common_city_in_common_location,
                                 p_inherent_difference_city_in_common_location))
        child.check_reverse()
        """



