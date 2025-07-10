

from Debug import DEBUG
from Exceptions import CyclicNotationException, InitialisationError
from Individual import Individual
from Parameters import Parameters
import numpy as np

class MutationFactory():

    _parameters: Parameters

    def __init__(self, parameters: Parameters):
        self._parameters = parameters


    def mutate(self, individual: Individual) -> Individual:
        if np.random.random_sample() > self._parameters.get_mutation_rate():
            p = np.random.random_sample()
            if p <= self._parameters.get_chance_on_DM_mutation():
                return self.DM(individual)

            elif p <= self._parameters.get_chance_on_DM_mutation() + self._parameters.get_chance_on_SIM_mutation():
                return self.SIM(individual)
            else:
                return self.IVM(individual)

        return individual
    

    def DM(self, individual: Individual) -> Individual:
        #displacement mutation
        try:
            for city in individual():
                continue
        except CyclicNotationException:
            raise InitialisationError("Mutation individual is not cyclic")
        
        cyclic_notation = individual.get_cyclic_representation()
        index1 = np.random.choice(cyclic_notation)
        index2 = index1 + self._parameters.get_length_DM_subtour()
        tmp = individual.index_with_list(np.array([index1, index2]))
       

        city_before_subpath = tmp[0]
        city_end_subpath = tmp[1]

        tmp = None #free memory

        new_start_index = np.random.choice(cyclic_notation)
       
        if index1 <= new_start_index <= index2:
            new_start_index = new_start_index + self._parameters.get_length_DM_subtour() + 1

        if index2 != index2%individual.get_number_of_cities() and new_start_index <= index2%individual.get_number_of_cities():
            new_start_index = new_start_index + self._parameters.get_length_DM_subtour() + 1

        new_start_city = individual.index(new_start_index)
        
        cyclic_notation[city_before_subpath], cyclic_notation[city_end_subpath], cyclic_notation[new_start_city] = \
            cyclic_notation[city_end_subpath], cyclic_notation[new_start_city] , cyclic_notation[city_before_subpath]
        
        
        mutated_individual = Individual(individual.get_number_of_cities(), individual.get_distance_matrix())
        mutated_individual.use_cyclic_notation(cyclic_notation, check_reverse= True)

        if DEBUG:
                try:
                    for city in individual():
                        continue
                except CyclicNotationException as e:
                    raise InitialisationError("The candidate solution is not a valid cycle")
        
        return mutated_individual
    
    
    
    def SIM(self, individual: Individual, after_local_opt: bool = False)-> Individual:
        # Simple inversion mutation
        path_notation = individual.get_path_representation()
        index1 = np.random.choice(individual.get_number_of_cities() - self._parameters.get_length_SIM_subtour())
        
        if index1 == 0: index1 +=1

        index2 = index1 + self._parameters.get_length_SIM_subtour()

        path_notation[:index1], path_notation[index1: index2], path_notation[index2:] = \
            path_notation[:index1], path_notation[index2 - 1: index1-1: -1], path_notation[index2:]
        mutated_individual = Individual(individual.get_number_of_cities(), individual.get_distance_matrix())
        mutated_individual.use_path_notation(path_notation)
        changed_cities = np.empty(self._parameters.get_length_IVM_subtour() + 1, dtype=np.int16)
        changed_cities[:self._parameters.get_length_IVM_subtour()], changed_cities[self._parameters.get_length_IVM_subtour()] = path_notation[index1: index2], path_notation[index1 -1]
        if after_local_opt:
            mutated_individual.set_dont_look_False(changed_cities)
        if DEBUG:
                try:
                    for city in individual():
                        continue
                except CyclicNotationException as e:
                    raise InitialisationError("The candidate solution is not a valid cycle")
        return mutated_individual
    
    
    
    def IVM(self, individual: Individual, after_local_opt: bool = False) -> Individual:
        # Inversion mutation
        path_notation = individual.get_path_representation()
        index1 = np.random.choice(individual.get_number_of_cities() - self._parameters.get_length_IVM_subtour() - 1)
        if index1 == 0: index1 +=1
        index2 = index1 + self._parameters.get_length_IVM_subtour()
        path_notation[:index1], path_notation[index1: index2], path_notation[index2:] = \
            path_notation[:index1], path_notation[index2 - 1: index1-1: -1], path_notation[index2:]
        
        index1 = index1 -1 #city before the inversion is "city_before_subpath"
        city_before_subpath = path_notation[index1]
        city_end_subpath = path_notation[index2]

        new_start_index = np.random.choice(path_notation)
      
        if index1 <= new_start_index <= index2:
            # + 2 since we did index1-1
            new_start_index = (new_start_index + self._parameters.get_length_IVM_subtour() + 2)%individual.get_number_of_cities()

        new_start_city = path_notation[new_start_index]

        cyclic_notation = individual.get_cyclic_notation_from_path(path_notation) #number of cities remains constant
        cyclic_notation[city_before_subpath], cyclic_notation[city_end_subpath], cyclic_notation[new_start_city] = \
            cyclic_notation[city_end_subpath], cyclic_notation[new_start_city] , cyclic_notation[city_before_subpath]
        

        mutated_individual = Individual(individual.get_number_of_cities(), individual.get_distance_matrix())
        mutated_individual.use_cyclic_notation(cyclic_notation, check_reverse=True)
        changed_cities = np.empty(self._parameters.get_length_IVM_subtour() + 3,dtype=np.int16)
        changed_cities[:self._parameters.get_length_IVM_subtour() + 2], changed_cities[self._parameters.get_length_IVM_subtour() + 2] = path_notation[index1: index2 + 1], new_start_city
        if after_local_opt:
            mutated_individual.set_dont_look_False(changed_cities)
        if DEBUG:
                try:
                    for city in individual():
                        continue
                except CyclicNotationException as e:
                    raise InitialisationError("The candidate solution is not a valid cycle")
        return mutated_individual