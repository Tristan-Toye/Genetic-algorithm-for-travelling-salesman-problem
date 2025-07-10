import random
from tkinter import NO
from typing import List, Tuple
from copy import deepcopy
import numpy as np
from Exceptions import CyclicNotationException, InitialisationError, OutOfBoundsException, RepresentationException, InvalidCityInput, SubPathError
from Debug import DEBUG





class Individual():
    
    _representation : np.ndarray 
    _distance: float
    _NUMBER_OF_CITIES: int 

    _current_index: int
    _start_city: int

    _DISTANCE_MATRIX: np.ndarray

    _has_infinite: bool = False
    _scaling: float = 1.0

    _dont_look_opt3: np.ndarray
    _dont_look_opt4: np.ndarray
    _converged: bool

    _converged_on_basic: bool
    _converged_on_double_bridge: bool

    def __init__(self, number_of_cities:int, distanceMatrix: np.ndarray) -> None:
        self._representation = np.full(number_of_cities, -1, dtype=np.int16)
        self._distance = -1.0
        self._NUMBER_OF_CITIES = number_of_cities
        self._current_index = 0
        self._start_city = 0
        self._DISTANCE_MATRIX = distanceMatrix
        self._dont_look_opt3 = np.full_like(self._representation, False, dtype=np.bool)
        self._dont_look_opt4 = np.full_like(self._representation, False, dtype=np.bool)
        self._converged = False
        self._converged_on_basic = False
        self._converged_on_double_bridge = False

    def copy(self) -> 'Individual':
        tmp= Individual(self._NUMBER_OF_CITIES, self._DISTANCE_MATRIX)
        tmp.use_cyclic_notation(self._representation)
        if DEBUG:
                try:
                    for city in tmp():
                        continue
                except CyclicNotationException as e:
                    raise InitialisationError("The candidate solution is not a valid cycle")

        return tmp
    
    def set_converged(self, convergence: bool):
        self._converged = convergence

    def is_converged(self) -> bool:
        return self._converged
    
    def set_converged_on_basic(self, convergence: bool):
        self._converged_on_basic = convergence

    def is_converged_on_basic(self):
        return self._converged_on_basic
    
    
    def is_converged_double_bridge(self):
        return self._converged_on_double_bridge
    
    def set_converged_double_bridge(self, convergence: bool):
        self._converged_on_double_bridge = convergence
        self._dont_look_opt3 = np.full_like(self._representation, False, dtype=np.bool)
    
    def get_dont_look_3opt_exposed(self) -> np.ndarray:
        return self._dont_look_opt3
     
    def get_dont_look_4opt_exposed(self) -> np.ndarray:
        return self._dont_look_opt4
    
    def set_dont_look_False(self, cities: np.ndarray):
        self._dont_look_opt3 = np.full_like(self._dont_look_opt3, True)
        self._dont_look_opt3[cities] = False
        self._dont_look_opt4 = np.full_like(self._dont_look_opt4, True)
        self._dont_look_opt4[cities] = False
    
    def __repr__(self) -> str:
        return f"Individual(route={self._representation}, distance={self._distance})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Individual):
            return NotImplemented
        #TODO: risky business!!! -> distance is not unique
        if self._distance == -1:
            self.get_distance()
        if other._distance == -1:
            other.get_distance()
        return self._distance == other._distance
        

    def __gt__(self, other: object) -> bool:
        if not isinstance(other, Individual):
            return NotImplemented
        return self._distance > other._distance
    
    def __getitem__(self, index: int) -> int:
        return self._representation[index]
    
    def __hash__(self):

        return hash(str(self._representation))
    
    @property
    def size(self) -> int:
        return self._representation.size
    
    def __call__(self, start_city: int = 0):
        self._current_index = start_city
        self._start_city = start_city
        return self
    
    def __iter__(self):

        for counter in range(self._NUMBER_OF_CITIES):
            if DEBUG and counter != 0 and self._current_index == self._start_city:
                # if np.count_nonzero(self._representation == -1) != 1:
                #     debug = np.count_nonzero(self._representation == -1) != 1
                raise CyclicNotationException("This individual is not one cycle, we are back at the starting city before visiting all cities.")
            tmp = self._representation[self._current_index]
            yield tmp #city value can change during the iteration
            tmp = self._representation[self._current_index] # tmp is not guaranteed equal to self._representation[self._current_index] anymore
            # if tmp == 0 and np.count_nonzero(self._representation == -1) > 0:
            #     debug = True
            # if np.count_nonzero(self._representation == -1) == 1:
            #     debug = 0

            self._current_index = tmp

        if DEBUG and self._current_index != self._start_city:
                raise CyclicNotationException("This individual is not one cycle, after passing through the cycle we are not at the first city.")

        self._current_index = 0
        self._start_city = 0
    
    def find_next_city(self, city:int) -> int:
        if city >= self._NUMBER_OF_CITIES: raise OutOfBoundsException(f"City doesn't exists.\n\t city: {city} \n\t NUMBER_OF_CITIES: {self._NUMBER_OF_CITIES}")
        return self._representation[city]
    
    def find_index(self, city: int)-> int:
        if city == 0: return 0
        if city >= self._NUMBER_OF_CITIES: raise OutOfBoundsException(f"City doesn't exists.\n\t city: {city} \n\t NUMBER_OF_CITIES: {self._NUMBER_OF_CITIES}")
        
        current_city = 0
        for counter in range(self._NUMBER_OF_CITIES - 1):
            current_city = self._representation[current_city]
            if current_city == city: return counter + 1
        raise CyclicNotationException(f"City not found. \n\t city: {city}")
    
    def find_index_and_previous(self, city: int) -> Tuple[int, int]:
        if city == 0: return 0, np.argwhere(self._representation == 0)[0][0]
        if city >= self._NUMBER_OF_CITIES: raise OutOfBoundsException(f"City doesn't exists.\n\t city: {city} \n\t NUMBER_OF_CITIES: {self._NUMBER_OF_CITIES}")
        
        current_city = 0
        for counter in range(self._NUMBER_OF_CITIES - 1):
            tmp = self._representation[current_city]
            if tmp == city: return counter + 1, current_city
            current_city = tmp
        raise CyclicNotationException(f"City not found. \n\t city: {city}")
    
    def index(self, index: int) -> int:
        #assumption: indexes may be bigger then list size

        current_city = 0
        for _ in range(index):
            current_city = self._representation[current_city]
        return current_city
    
    

    def index_with_list(self, indexes: np.ndarray) -> np.ndarray:
        #assumption: indexes are sorted
        #assumption: indexes may be bigger then list size
        tmp = np.empty_like(indexes)
        current_city = 0
        tmp_index = 0
        for current_index in range(indexes[indexes.size -1] +1):
            if current_index == indexes[tmp_index]: 
                tmp[tmp_index] = current_city
                tmp_index +=1
            current_city = self._representation[current_city]

        if DEBUG and tmp_index != tmp.size:
            raise IndexError(f"Index list was probably not ordered: \n\tindexes:\t{indexes}")
        return tmp
    
    def get_path_between_indexes_cyclic_format(self, index1: int, index2: int) -> Tuple[int, np.ndarray, int]:
        
        tmp = np.full(self._NUMBER_OF_CITIES,-1)
        current_city = 0
        start_city = -1
        if index1 >= index2: return -1, tmp, -1
        if index1 >= self._NUMBER_OF_CITIES:
            index1 = index1%self._NUMBER_OF_CITIES
            index2 = index2%self._NUMBER_OF_CITIES
            if index2 < index1: index2 += self._NUMBER_OF_CITIES
        if index2 - index1 >= self._NUMBER_OF_CITIES: raise OutOfBoundsException(f"Indexes out of range of cities.\n\t NUMBER_OF_CITIES: {self._NUMBER_OF_CITIES} \n\t index1: {index1} \n\t index2: {index2}")
        counter = 0

        while True:
            tmp_city = self._representation[current_city]
            if counter == index1:
                start_city = current_city
            if counter >= index1:
                tmp[current_city] = tmp_city
                if counter == index2 - 1: return start_city, tmp, tmp_city
            current_city = tmp_city
            counter +=1
        
    
    def get_end_of_first_assigned_subpath(self) -> int:
        current_city = 0
        in_path = False
        for _ in range(self._NUMBER_OF_CITIES - 1):
            tmp = self._representation[current_city]
            if not in_path and tmp != -1: in_path = True
            if in_path and tmp == -1: return current_city
            current_city = tmp
        if in_path:
            raise CyclicNotationException(f"No unassigned cities left . \n\t individual: {self}")
        else:
            return -1
        
    


    def iterator_set(self, city:int) -> None:
        if DEBUG and city < 0:
            raise InvalidCityInput(f"Cities cannot be represented by a negative number. \n\t input: {city}")
        # if city == 0 and np.count_nonzero(self._representation == -1) != 1:
        #     debug = True
        self._representation[self._current_index] = city

    def check_reverse(self):
        
        path_notation = np.full_like(self._representation, -1, dtype=np.int16)
        current_city = 0

        for index in range(self._NUMBER_OF_CITIES):
            path_notation[index] = current_city
            current_city = self._representation[current_city]
        
        if DEBUG and -1 in self._representation: raise CyclicNotationException(f"Error in copying cyclic notation.\n\t {self._representation}")
        if DEBUG and -1 in path_notation: raise CyclicNotationException("Error in building path notation")
        if DEBUG and np.unique(path_notation).size != path_notation.size: raise CyclicNotationException("path notation is not a accepted cycle")


        self.use_path_notation(path_notation, self._representation)
    
    def use_cyclic_notation(self, cyclic_notation: np.ndarray, check_reverse = False) -> None:
        if (cyclic_notation.size != self._NUMBER_OF_CITIES): 
            raise RepresentationException(f"The representaion has the wrong size.\n\t NUMBER_OF_CITIES: {self._NUMBER_OF_CITIES}\n\t length representation: {len(cyclic_notation)}")
        
        self._representation = deepcopy(cyclic_notation)
        self._representation = self._representation.astype(np.int16)

        if DEBUG and not -1 in self._representation:
            try:
                for city in self():
                    continue
            except CyclicNotationException as e:
                raise RepresentationException("The representation is not cyclic")

        if check_reverse:
            self.check_reverse()
        
            

    def use_path_notation(self, path_notation: np.ndarray, cyclic_notation_copy: np.ndarray = np.empty(0)) -> None:
        if (path_notation.size != self._NUMBER_OF_CITIES): 
            raise RepresentationException(f"The representaion has the wrong size.\n\t NUMBER_OF_CITIES: {self._NUMBER_OF_CITIES}\n\t length representation: {len(path_notation)}")
        
        if not cyclic_notation_copy.size:
            cyclic_notation_copy = self.get_cyclic_notation_from_path(path_notation)
      
        
        reverse_cyclic = self.get_cyclic_notation_from_path(path_notation[::-1])
        distance_forward, has_inf_forward = self._calculate_distance(cyclic_notation_copy)
        distance_reversed, has_inf_reversed = self._calculate_distance(reverse_cyclic)
        if distance_forward < distance_reversed:
            self._representation = cyclic_notation_copy
            self._distance = distance_forward
            self._has_infinite = has_inf_forward
        else:
            self._representation = reverse_cyclic
            self._distance = distance_reversed
            self._has_infinite = has_inf_reversed
        
        try:
            for city in self():
                continue
        except CyclicNotationException as e:
            raise RepresentationException("The representation is not cyclic")
        

    def get_cyclic_notation_from_path(self, path_notation: np.ndarray) -> np.ndarray:

        cyclic_notation: np.ndarray = np.full(self._NUMBER_OF_CITIES, -1, dtype=  np.int16)

        for index_path in np.arange(1, self._NUMBER_OF_CITIES + 1, dtype= np.int16):
            cyclic_notation[path_notation[index_path-1]] = path_notation[index_path % self._NUMBER_OF_CITIES]

        if DEBUG and -1 in cyclic_notation: raise CyclicNotationException("Error in building cyclic notation")
       
        return cyclic_notation

    def reset_distance_calc(self):
        self._distance = -1     

    def _calculate_distance(self, cyclic_notation) -> Tuple[float, bool]:
        total_distance = 0.0
        penalty = 1e6  # High penalty for unreachable paths
        local_inf = False
        for city in range(self._NUMBER_OF_CITIES):
            if np.isinf(dis := self._DISTANCE_MATRIX[city, cyclic_notation[city]]):
                local_inf = True
                total_distance+= penalty
            else:
                total_distance += dis

        return total_distance, local_inf

    def get_path_representation(self) -> np.ndarray:
        path_notation = np.full_like(self._representation, -1, dtype=np.int16)
        current_city = 0
        for index in range(self._NUMBER_OF_CITIES):
            path_notation[index] = current_city
            current_city = self._representation[current_city]

        if DEBUG and -1 in path_notation: raise CyclicNotationException("Error in building path notation")
 
        return path_notation
    
    def get_cyclic_representation(self) -> np.ndarray:
        return deepcopy(self._representation)
    
    def get_cyclic_representation_exposed(self) -> np.ndarray:
        return self._representation
    
    def get_distance(self) -> Tuple[float, bool]:
        if self._distance != -1:
            return self._distance, self._has_infinite
        else:
            self._distance, self._has_infinite = self._calculate_distance(self._representation)
            return self._distance, self._has_infinite
    
    def get_number_of_cities(self)-> int:
        return self._NUMBER_OF_CITIES
    
    def get_distance_matrix(self) -> np.ndarray:
        return self._DISTANCE_MATRIX
    
    def reset_scaling(self):
        self._scaling = 1.0

    def adjust_scaling(self, adjustment: float):
        self._scaling *= adjustment

    def set_scaling(self, value: float):
        self._scaling = value
    
    def get_fitness(self) -> float:
        return self.get_distance()[0]*self._scaling
    
    def mutate(self, positions_to_swap: int = 1) -> None:

        start_mutate_point, first_swap, second_swap = self._swap_two_elements()
        for _ in range(positions_to_swap - 1):
            start_mutate_point, first_swap, second_swap = self._swap_two_elements(start_mutate_point, first_swap, second_swap)
        


    def _swap_two_elements(self, 
                           start_mutate_point: np.int16 = np.int16(-1), 
                           first_swap: np.int16 = np.int16(-1),
                           second_swap: np.int16 = np.int16(-1) ) -> tuple[np.int16, np.int16, np.int16]:

        if start_mutate_point == -1:
            start_mutate_point = np.random.choice(self._representation)
            first_swap = self._representation[start_mutate_point]
            second_swap = self._representation[first_swap]

        final_mutate_point = self._representation[second_swap]

        self._representation[start_mutate_point] = second_swap
        self._representation[first_swap] = final_mutate_point
        self._representation[second_swap] = first_swap
        return second_swap, first_swap, final_mutate_point