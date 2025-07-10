from copy import deepcopy
import time
from typing import Iterable, Tuple
import numpy as np

from Debug import AGAIN_WITHOUT_DONT_LOOK_BITS, DEBUG, ENABLE_DOUBLE_BRIDGE, ENABLE_DOUBLE_BRIDGE_SECOND_CORE
from Exceptions import CyclicNotationException, LocalSearchException, OutOfBoundsException
from Individual import Individual

from numba import jit,int16, boolean, int64, float64

from Parameters import Parameters





def find_local_optima(individual: Individual, distance_matrix, ordered_arg_distance_matrix: np.ndarray):
    
    while not individual.is_converged():
        
        converged = opt_3(individual.get_cyclic_representation_exposed(), individual.get_dont_look_3opt_exposed(), distance_matrix, ordered_arg_distance_matrix)
        individual.reset_distance_calc()
        individual.set_converged(converged)

        if DEBUG:
            try:
                for city in individual(0):
                    pass
            except CyclicNotationException:
                raise LocalSearchException("cycle is broken")
        
       
            
    while ENABLE_DOUBLE_BRIDGE_SECOND_CORE and not individual.is_converged_double_bridge():
    
        converged = opt_4_double_bridge(individual.get_cyclic_representation_exposed(), individual.get_dont_look_4opt_exposed(), distance_matrix, ordered_arg_distance_matrix)
        individual.reset_distance_calc()
        individual.set_converged_double_bridge(converged)

        if DEBUG:
            try:
                for city in individual(0):
                    pass
            except CyclicNotationException:
                raise LocalSearchException("cycle is broken")




def optimise(individuals: Iterable[Individual], distance_matrix: np.ndarray, ordered_arg_distance_matrix: np.ndarray, parameters: Parameters):
    
    did_work = False
    for individual in individuals:
        if not individual.is_converged():
            did_work = True
            converged = opt_3(individual.get_cyclic_representation_exposed(), individual.get_dont_look_3opt_exposed(), distance_matrix, ordered_arg_distance_matrix)
            individual.reset_distance_calc()
            individual.set_converged(converged)

            if DEBUG:
                try:
                    for city in individual(0):
                        pass
                except CyclicNotationException:
                    raise LocalSearchException("cycle is broken")
        
        elif ENABLE_DOUBLE_BRIDGE and not individual.is_converged_double_bridge():
            did_work = True
            converged = opt_4_double_bridge(individual.get_cyclic_representation_exposed(), individual.get_dont_look_4opt_exposed(), distance_matrix, ordered_arg_distance_matrix)
            individual.reset_distance_calc()
            individual.set_converged_double_bridge(converged)

            if DEBUG:
                try:
                    for city in individual(0):
                        pass
                except CyclicNotationException:
                    raise LocalSearchException("cycle is broken")
        
    return did_work



#when parallell: not include parents!!!!
def improve_population(population: Iterable[Individual], distance_matrix: np.ndarray):
    
    for individual in population:
        if not individual.is_converged_on_basic():
            changed = improve(individual.get_cyclic_representation_exposed(), distance_matrix)
            individual.reset_distance_calc()
            individual.set_converged_on_basic(changed)
    

@jit(float64(float64[:,:], int16, int16, int16,int16),nopython = True)
def calculate_distance(distance_matrix: np.ndarray, city1, city2, city3, city4) -> float:
    tmp = distance_matrix[city1, city2] + distance_matrix[city2, city3] + distance_matrix[city3, city4]
    return tmp #type: ignore

@jit(boolean(int16[:],float64[:,:]),nopython = True)
def improve(individual: np.ndarray, distance_matrix: np.ndarray) -> bool:

    city_t1 = 0
    city_t2 = individual[city_t1]
    city_t3 = individual[city_t2]
    city_t4 = individual[city_t3]
    counter = 0
    changed = False
    while counter < individual.size:
        tmp = np.empty(6)
        tmp[0] = calculate_distance(distance_matrix, city_t1, city_t2, city_t3, city_t4)
        tmp[1] = calculate_distance(distance_matrix, city_t1, city_t2, city_t4, city_t3)
        tmp[2] = calculate_distance(distance_matrix, city_t1, city_t3, city_t2, city_t4)
        tmp[3] = calculate_distance(distance_matrix, city_t1, city_t3, city_t4, city_t2)
        tmp[4] = calculate_distance(distance_matrix, city_t1, city_t4, city_t2, city_t3)
        tmp[5] = calculate_distance(distance_matrix, city_t1, city_t4, city_t3, city_t2)

        index = np.argmin(tmp)

        if index == 0: #nothing changes
            city_t1, city_t2, city_t3, city_t4 = city_t2, city_t3, city_t4, individual[city_t4]
        elif index == 1:
            changed = True
            individual[city_t2],individual[city_t3],  individual[city_t4] = city_t4, individual[city_t4], city_t3, 
            city_t1, city_t2, city_t3, city_t4 = city_t2, city_t4, city_t3, individual[city_t3]
        elif index == 2:
            changed = True
            individual[city_t1], individual[city_t2],individual[city_t3] = \
                city_t3, city_t4, city_t2
            city_t1, city_t3, city_t4 = city_t3, city_t4, individual[city_t4]
        elif index == 3:
            changed = True
            individual[city_t1], individual[city_t2],individual[city_t4] = \
                city_t3, individual[city_t4], city_t2
            city_t1, city_t2, city_t3, city_t4 = city_t3, city_t4, city_t2, individual[city_t2]
        elif index == 4:
            changed = True
            individual[city_t1],  individual[city_t3], individual[city_t4] = \
                city_t4,  individual[city_t4], city_t2
            city_t1, city_t4 = city_t4, individual[city_t3]
        elif index == 5:
            changed = True
            individual[city_t1], individual[city_t2], individual[city_t3], individual[city_t4] = \
                city_t4,  individual[city_t4], city_t2, city_t3
            city_t1, city_t2, city_t3, city_t4 = city_t4, city_t3,city_t2, individual[city_t2]

        counter +=1
    return changed  

@jit( (int16[:], int16),nopython = True)
def find_index_and_previous(cyclic_representation: np.ndarray, city: int) -> Tuple[int, int]:
        if cyclic_representation.size == 0: return 0,0
        if city == 0: return 0, np.argwhere(cyclic_representation == 0)[0][0]
        if city >= cyclic_representation.size: raise OutOfBoundsException(f"City doesn't exists.\n\t city: {city} \n\t NUMBER_OF_CITIES: {cyclic_representation.size}")
        
        current_city = 0
        for counter in range(cyclic_representation.size - 1):
            tmp = cyclic_representation[current_city]
            if tmp == city: return counter + 1, current_city
            current_city = tmp
        raise CyclicNotationException(f"City not found. \n\t city: {city}")    
          


@jit((int16[:], boolean[:], int16, int16, int16, int16, int16, int16),nopython = True)
def make_swap(cyclic_path: np.ndarray,  dont_look:np.ndarray, city_t1, city_t2, city_t3, city_t4, city_t5, city_t6):
    if cyclic_path.size == 0: return 
    cyclic_path[city_t1],cyclic_path[city_t3], cyclic_path[city_t5] = city_t4, city_t6, city_t2
    dont_look[city_t1], dont_look[city_t3], dont_look[city_t5] = False, False, False


@jit(boolean(int16[:], boolean[:], float64[:,:], int16[:,:]), nopython = True)
def opt_3(cyclic_representation: np.ndarray, dont_look: np.ndarray, distance_matrix: np.ndarray, ordered_arg_distance_matrix: np.ndarray) -> bool:

    if cyclic_representation.size == 0: return False
    k = 20

    
    city_t1 = 0
    index_city_t1 = 0
    while index_city_t1 < cyclic_representation.size:
        city_t2 = cyclic_representation[city_t1]
        if not dont_look[city_t1]:
            for city_t4 in ordered_arg_distance_matrix[city_t1][1:k]:
                if city_t4 == city_t2: continue
                index_city_t4, city_t3 = find_index_and_previous(cyclic_representation, city_t4)

                if index_city_t1 + 1 < index_city_t4 -1 <= (index_city_t1 + cyclic_representation.size) - 3  :
                    for city_t6 in ordered_arg_distance_matrix[city_t3][1:k]:
                        index_city_t6, city_t5 = find_index_and_previous(cyclic_representation, city_t6)

                        if index_city_t6 < index_city_t1 or index_city_t4 < index_city_t6 - 1 :

                            current_distance = distance_matrix[city_t1, city_t2] + distance_matrix[city_t3, city_t4] + distance_matrix[city_t5, city_t6]

                            if np.isinf(current_distance):
                                make_swap(cyclic_representation, dont_look, city_t1, city_t2, city_t3, city_t4, city_t5, city_t6)
                                return False

                            new_distance = distance_matrix[city_t1, city_t4] + distance_matrix[city_t5, city_t2] + distance_matrix[city_t3, city_t6]
                            
                            if new_distance < current_distance:
                                make_swap(cyclic_representation, dont_look, city_t1, city_t2, city_t3, city_t4, city_t5, city_t6)
                                return False
            dont_look[city_t1] = True
        index_city_t1 +=1
        city_t1 = city_t2
    
    if AGAIN_WITHOUT_DONT_LOOK_BITS:
        #run again without dont look bits
        city_t1 = 0
        index_city_t1 = 0
        while index_city_t1 < cyclic_representation.size:
            city_t2 = cyclic_representation[city_t1]
            for city_t4 in ordered_arg_distance_matrix[city_t1][1:k]:
                if city_t4 == city_t2: continue 
                index_city_t4, city_t3 = find_index_and_previous(cyclic_representation, city_t4)

                if index_city_t1 + 1 < index_city_t4 -1 <= (index_city_t1 + cyclic_representation.size) - 3  :
                    for city_t6 in ordered_arg_distance_matrix[city_t3][1:k]:
                        index_city_t6, city_t5 = find_index_and_previous(cyclic_representation, city_t6)

                        if index_city_t6 < index_city_t1 or index_city_t4 < index_city_t6 - 1 :

                            current_distance = distance_matrix[city_t1, city_t2] + distance_matrix[city_t3, city_t4] + distance_matrix[city_t5, city_t6]

                            if np.isinf(current_distance):
                                make_swap(cyclic_representation, dont_look, city_t1, city_t2, city_t3, city_t4, city_t5, city_t6)
                                
                                return False

                            new_distance = distance_matrix[city_t1, city_t4] + distance_matrix[city_t5, city_t2] + distance_matrix[city_t3, city_t6]
                            
                            if new_distance < current_distance:
                                make_swap(cyclic_representation, dont_look, city_t1, city_t2, city_t3, city_t4, city_t5, city_t6)
                            
                                return False
            index_city_t1 +=1
            city_t1 = city_t2
    
    return True


@jit((int16[:], boolean[:], int16, int16, int16, int16, int16, int16, int16, int16),nopython = True)
def make_swap_double_bridge(cyclic_path: np.ndarray,  dont_look:np.ndarray, city_t1, city_t2, city_t3, city_t4, city_t5, city_t6, city_t7, city_t8):
    if cyclic_path.size == 0: return 
    cyclic_path[city_t1],cyclic_path[city_t3], cyclic_path[city_t5], cyclic_path[city_t7] = city_t6, city_t8, city_t2, city_t4
    dont_look[city_t1], dont_look[city_t3], dont_look[city_t5], dont_look[city_t7] = False, False, False, False

@jit(boolean(int16[:], boolean[:], float64[:,:], int16[:,:]), nopython = True)
def opt_4_double_bridge(cyclic_representation: np.ndarray, dont_look: np.ndarray, distance_matrix: np.ndarray, ordered_arg_distance_matrix: np.ndarray) -> bool:
    if cyclic_representation.size == 0: return False
    k = 20

    city_t1 = 0
    counter = 0
    while counter < cyclic_representation.size:
        city_t2 = cyclic_representation[city_t1]
        if not dont_look[city_t1]:
            for city_t6 in ordered_arg_distance_matrix[city_t1][1:k]:
                if city_t6 == city_t2: continue
                index_city_t6, city_t5 = find_index_and_previous(cyclic_representation, city_t6)

                if counter + 3 < index_city_t6 -1 <= (counter + cyclic_representation.size) - 3  :
                    city_t3 = cyclic_representation[city_t2]
                    while cyclic_representation[city_t3] != city_t5:
                        city_t4 = cyclic_representation[city_t3]
                        for city_t8 in ordered_arg_distance_matrix[city_t3][1:k]:
                            if city_t8 == city_t4: continue
                            index_city_t8, city_t7 = find_index_and_previous(cyclic_representation, city_t8)
                            if index_city_t8 < counter or index_city_t6 < index_city_t8 - 1 :

                                current_distance = distance_matrix[city_t1, city_t2] + distance_matrix[city_t3, city_t4] \
                                        + distance_matrix[city_t5, city_t6] + distance_matrix[city_t7, city_t8]
                                

                                if np.isinf(current_distance):
                                    make_swap_double_bridge(cyclic_representation, dont_look, city_t1, city_t2, city_t3, city_t4, city_t5, city_t6, city_t7, city_t8)
                                 
                                    return False

                                new_distance = distance_matrix[city_t1, city_t6] + distance_matrix[city_t7, city_t4]\
                                        + distance_matrix[city_t5, city_t2] + distance_matrix[city_t3, city_t8]
                            
                                if new_distance < current_distance:
                                    make_swap_double_bridge(cyclic_representation, dont_look, city_t1, city_t2, city_t3, city_t4, city_t5, city_t6, city_t7, city_t8)
                                    
                                    return False
                        city_t3 = city_t4       
                        
            dont_look[city_t1] = True
        counter +=1
        city_t1 = city_t2
    
    if AGAIN_WITHOUT_DONT_LOOK_BITS:
        city_t1 = 0
        counter = 0
        while counter < cyclic_representation.size:
            city_t2 = cyclic_representation[city_t1]
            if not dont_look[city_t1]:
                for city_t6 in ordered_arg_distance_matrix[city_t1][1:k]:
                    if city_t6 == city_t2: continue
                    index_city_t6, city_t5 = find_index_and_previous(cyclic_representation, city_t6)

                    if counter + 3 < index_city_t6 -1 <= (counter + cyclic_representation.size) - 3  :
                        city_t3 = cyclic_representation[city_t2]
                        while cyclic_representation[city_t3] != city_t5:
                            city_t4 = cyclic_representation[city_t3]
                            for city_t8 in ordered_arg_distance_matrix[city_t3][1:k]:
                                if city_t8 == city_t4: continue
                                index_city_t8, city_t7 = find_index_and_previous(cyclic_representation, city_t8)
                                if index_city_t8 < counter or index_city_t6 < index_city_t8 - 1 :

                                    current_distance = distance_matrix[city_t1, city_t2] + distance_matrix[city_t3, city_t4] \
                                            + distance_matrix[city_t5, city_t6] + distance_matrix[city_t7, city_t8]
                                    

                                    if np.isinf(current_distance):
                                        make_swap_double_bridge(cyclic_representation, dont_look, city_t1, city_t2, city_t3, city_t4, city_t5, city_t6, city_t7, city_t8)
                                        print("lets go double bridge no look")
                                        return False

                                    new_distance = distance_matrix[city_t1, city_t6] + distance_matrix[city_t7, city_t4]\
                                            + distance_matrix[city_t5, city_t2] + distance_matrix[city_t3, city_t8]
                                
                                    if new_distance < current_distance:
                                        make_swap_double_bridge(cyclic_representation, dont_look, city_t1, city_t2, city_t3, city_t4, city_t5, city_t6, city_t7, city_t8)
                                        print("lets go double bridge no look")
                                        return False
                            city_t3 = city_t4       
                            
                dont_look[city_t1] = True
            counter +=1
            city_t1 = city_t2
    
    return True
