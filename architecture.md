# Architecture Overview: Genetic Algorithm for Traveling Salesman Problem

## Executive Summary

This document provides a comprehensive overview of the architecture and optimization techniques implemented in the Genetic Algorithm for Traveling Salesman Problem (TSP) solver. The system employs a sophisticated multi-layered architecture combining genetic algorithms, local search optimization, parallel processing, and hybrid metaheuristics to achieve high-performance TSP solutions.

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Core Components](#core-components)
3. [Data Flow Architecture](#data-flow-architecture)
4. [Optimization Techniques](#optimization-techniques)
5. [Parallel Processing Architecture](#parallel-processing-architecture)
6. [Performance Optimization](#performance-optimization)
7. [Integration Patterns](#integration-patterns)
8. [Scalability Considerations](#scalability-considerations)

## System Architecture

### High-Level Architecture

The system follows a **modular, event-driven architecture** with the following key characteristics:

```
┌─────────────────────────────────────────────────────────────────┐
│                        Main Controller                          │
│                         (main.py)                              │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                    GA Core Engine                               │
│                (TravellingSalesMan.py)                         │
└─────────────────────┬───────────────────────────────────────────┘
                      │
        ┌─────────────┼─────────────┐
        ▼             ▼             ▼
┌─────────────┐ ┌─────────────┐ ┌─────────────┐
│  Selection  │ │  Variation  │ │ Elimination │
│ (Selection.py)│ │(Variation.py)│ │(Elimination.py)│
└─────────────┘ └─────────────┘ └─────────────┘
        │             │             │
        └─────────────┼─────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Parallel Processing Layer                      │
│              (SecondCoreProcess.py)                             │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Local Search Layer                           │
│                  (local_search.py)                             │
└─────────────────────────────────────────────────────────────────┘
```

### Architectural Principles

1. **Separation of Concerns**: Each component has a single, well-defined responsibility
2. **Modularity**: Components are loosely coupled and highly cohesive
3. **Parallelism**: Multi-threading and multi-processing for performance
4. **Extensibility**: Easy to add new optimization techniques
5. **Performance**: JIT compilation and optimized data structures

## Core Components

### 1. Main Controller (`main.py`)

**Purpose**: Orchestrates the entire optimization process and manages experiment execution.

**Key Responsibilities**:
- Initialize system components
- Manage experiment parameters
- Coordinate between main GA and second core process
- Handle Weights & Biases integration
- Manage execution flow and termination

**Architecture Pattern**: **Facade Pattern** - Provides simplified interface to complex subsystem

```python
class r0849537:
    def __init__(self, use_wandb, run_name=None):
        # Initialize all components
        self._parameters = Parameters()
        self.reporter = Reporter()
        
    def optimize(self, filename, index=0):
        # Orchestrate optimization process
        # Coordinate between components
        # Manage parallel processes
```

### 2. GA Core Engine (`TravellingSalesMan.py`)

**Purpose**: Implements the main genetic algorithm loop and coordinates all GA operations.

**Key Responsibilities**:
- Execute genetic algorithm generations
- Coordinate selection, variation, and elimination phases
- Manage population evolution
- Integrate with local search optimization
- Handle convergence detection

**Architecture Pattern**: **Strategy Pattern** - Different optimization strategies can be plugged in

```python
class TravellingSalesMan:
    def __init__(self, parameters, distance_matrix, ...):
        self._selection = Selection(parameters.get_k_selection())
        self._variation = Variation(parameters)
        self._elimination = Elimination(distance_matrix, parameters, initialisation)
        
    def run(self, population):
        # Execute one generation of GA
        # Coordinate all phases
        # Return results
```

### 3. Population Management (`Population.py`)

**Purpose**: Manages TSP solution populations and provides population-level operations.

**Key Responsibilities**:
- Maintain solution collections
- Handle population sorting and selection
- Manage convergence tracking
- Provide population statistics
- Handle population evolution

**Data Structure**: Array-based population with efficient sorting and selection

```python
class Population:
    def __init__(self, parameters):
        self._population = np.empty(parameters.get_population_size(), dtype=object)
        self._size = parameters.get_population_size()
        
    def sort(self):
        # Sort by fitness (distance)
        
    def get_converged(self):
        # Return converged individuals
```

### 4. Individual Representation (`Individual.py`)

**Purpose**: Represents individual TSP solutions with efficient operations.

**Key Responsibilities**:
- Maintain tour representation (cyclic notation)
- Calculate tour distances
- Handle tour operations (mutation, crossover)
- Track convergence status
- Provide multiple representation formats

**Data Structure**: Cyclic representation for efficient tour operations

```python
class Individual:
    def __init__(self, number_of_cities, distance_matrix):
        self._cyclic_representation = np.empty(number_of_cities, dtype=np.int16)
        self._distance = -1
        self._converged = False
        
    def get_distance(self):
        # Calculate/return tour distance
        
    def get_cyclic_representation(self):
        # Return tour as successor list
```

## Data Flow Architecture

### Main Optimization Loop

```
1. Initialization
   ├── Load distance matrix
   ├── Create initial population
   ├── Initialize pheromone matrix
   └── Start second core process

2. Main GA Loop (per generation)
   ├── Selection Phase
   │   ├── Tournament selection
   │   └── Parent pair selection
   │
   ├── Variation Phase (Parallel)
   │   ├── Crossover operations (OX1, POS, Heuristic)
   │   ├── Mutation operations (SIM, DM, IVM)
   │   └── Pheromone-based generation
   │
   ├── Local Search Phase (Parallel)
   │   ├── 3-opt optimization
   │   ├── 4-opt (double bridge)
   │   └── Convergence detection
   │
   ├── Elimination Phase
   │   ├── Population evaluation
   │   ├── Survivor selection
   │   └── Population update
   │
   └── Convergence Check
       ├── Objective evaluation
       ├── Termination criteria
       └── Parameter adaptation
```

### Data Flow Between Components

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ Population  │───▶│ Selection   │───▶│ Variation   │
│             │    │             │    │             │
└─────────────┘    └─────────────┘    └─────────────┘
       ▲                   │                   │
       │                   ▼                   ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ Elimination │◀───│ Local Search│◀───│ Offspring   │
│             │    │             │    │ Generation  │
└─────────────┘    └─────────────┘    └─────────────┘
       │
       ▼
┌─────────────┐
│ Population  │
│ Update      │
└─────────────┘
```

## Optimization Techniques

### 1. Genetic Algorithm Components

#### Selection Strategy
- **Tournament Selection**: k=5 tournament size
- **Parent Pair Selection**: 3 parent pairs per generation
- **Elitism**: Preserve best individuals

#### Crossover Operators
- **OX1 (Order Crossover)**: Preserves order and position
- **POS (Position-based Crossover)**: Preserves absolute positions
- **Heuristic Crossover**: Distance-based edge recombination

#### Mutation Operators
- **SIM (Single Insertion Mutation)**: 40% probability
- **DM (Displacement Mutation)**: 20% probability
- **IVM (Inversion Mutation)**: 40% probability

### 2. Local Search Optimization

#### 3-Opt Algorithm
- **Purpose**: Remove three edges, test all reconnections
- **Implementation**: JIT-compiled for performance
- **Complexity**: O(n³) per iteration

#### 4-Opt (Double Bridge)
- **Purpose**: Additional optimization beyond 3-opt
- **Implementation**: Cannot be found by 3-opt
- **Complexity**: O(n⁴) per iteration

#### Convergence Detection
- **Don't-look bits**: Skip already optimized cities
- **Convergence flags**: Track optimization status
- **Early termination**: Stop when no improvement

### 3. Hybrid Metaheuristics

#### Pheromone Matrix
- **Purpose**: Ant Colony Optimization inspired solution generation
- **Implementation**: Matrix-based pheromone tracking
- **Integration**: Synchronized between main and second core

#### Adaptive Parameters
- **Dynamic population sizing**: Based on convergence
- **Mutation rate adaptation**: Increase when stuck
- **Immunity mechanisms**: Protect elite solutions

## Parallel Processing Architecture

### Multi-Threading Strategy

#### Offspring Generation Threads
```python
def produce_offspring(self, parent1_list, parent2_list):
    threads = []
    
    # OX1 crossover thread
    if self._parameters.get_number_of_OX1_offspring():
        thread = threading.Thread(target=self._thread_OX1, kwargs=kwargs)
        threads.append(thread)
    
    # POS crossover thread
    if self._parameters.get_number_of_POS_offspring():
        thread = threading.Thread(target=self._thread_POS, kwargs=kwargs)
        threads.append(thread)
    
    # Heuristic crossover thread
    if self._parameters.get_number_of_heuristic_offspring():
        thread = threading.Thread(target=self._thread_heuristic, kwargs=kwargs)
        threads.append(thread)
    
    # Start all threads
    for thread in threads:
        thread.start()
    
    # Wait for completion
    for thread in threads:
        thread.join()
```

#### Local Search Threads
```python
def optimise(individuals, distance_matrix, ordered_arg_distance_matrix, parameters):
    threads = []
    
    for individual in individuals:
        if not individual.is_converged():
            thread = threading.Thread(target=find_local_optima, 
                                    kwargs={"individual": individual, 
                                           "distance_matrix": distance_matrix,
                                           "ordered_arg_distance_matrix": ordered_arg_distance_matrix})
            threads.append(thread)
    
    # Execute in parallel
    for thread in threads:
        thread.start()
    
    for thread in threads:
        thread.join()
```

### Second Core Process Architecture

#### Purpose
- Dedicated parallel optimization process
- Continuous local search optimization
- Independent solution improvement

#### Communication Pattern
```python
class SecondCoreProcess:
    def __init__(self, queue_second_core_to_tsp, queue_tsp_to_second_core):
        self._queue_second_core_to_tsp = queue_second_core_to_tsp
        self._queue_tsp_to_second_core = queue_tsp_to_second_core
    
    def run(self):
        while True:
            # Receive pheromone matrix updates
            try:
                new_pheremone_matrix = self._queue_tsp_to_second_core.get_nowait()
                self._update_pheromone_matrix(new_pheremone_matrix)
            except queue.Empty:
                pass
            
            # Perform local search optimization
            self._optimize_population()
            
            # Send improved solutions back
            if self._found_better_solution():
                self._queue_second_core_to_tsp.put(self._best_solution)
```

## Performance Optimization

### 1. JIT Compilation (Numba)

#### Optimized Functions
```python
@jit(boolean(int16[:], boolean[:], float64[:,:], int16[:,:]), nopython=True)
def opt_3(cyclic_representation, dont_look, distance_matrix, ordered_arg_distance_matrix):
    # Optimized 3-opt implementation
    
@jit((int16, int16[:], int16[:], int16[:], int16[:], float64[:,:], int16[:,:], int16), nopython=True)
def build_heuristic_cycle_exposed(start_city, representation, representation_parent1, 
                                 representation_parent2, cities_left_to_allocate,
                                 distance_matrix, ordered_arg_distance_matrix, number_of_cities):
    # Optimized heuristic crossover
```

#### Performance Benefits
- **10-100x speedup** for critical algorithms
- **Type safety** with compile-time optimization
- **Memory efficiency** with optimized data structures

### 2. Memory Management

#### Efficient Data Structures
- **NumPy arrays**: Fast numerical operations
- **Pre-allocated arrays**: Minimize memory allocation
- **Lazy evaluation**: Calculate distances only when needed

#### Memory Optimization Strategies
```python
# Pre-allocate arrays for performance
self._parent1_list = np.empty(self._parameters.get_number_of_parent_pairs(), dtype=object)
self._parent2_list = np.empty(self._parameters.get_number_of_parent_pairs(), dtype=object)
self._offspring_list = np.ndarray(shape=(self._parameters.get_number_of_parent_pairs(), 
                                        self._parameters.get_offspring_size()), dtype=object)
```

### 3. Algorithmic Optimizations

#### Ordered Distance Matrix
- **Pre-computed**: Nearest neighbor lists for each city
- **Efficient lookup**: O(1) access to nearest neighbors
- **Local search acceleration**: Quick neighbor selection

#### Convergence Tracking
- **Don't-look bits**: Skip already optimized cities
- **Convergence flags**: Track optimization status
- **Early termination**: Stop when no improvement possible

## Integration Patterns

### 1. Component Integration

#### Dependency Injection
```python
class TravellingSalesMan:
    def __init__(self, parameters, distance_matrix, initialisation, 
                 pheremone_matrix, queue_second_core_to_tsp, queue_tsp_to_second_core):
        self._parameters = parameters
        self._selection = Selection(parameters.get_k_selection())
        self._variation = Variation(parameters)
        self._elimination = Elimination(distance_matrix, parameters, initialisation)
```

#### Event-Driven Communication
```python
# Queue-based communication between processes
self._queue_second_core_to_tsp = queue_second_core_to_tsp
self._queue_tsp_to_second_core = queue_tsp_to_second_core

# Non-blocking communication
try:
    new_solution = self._queue_second_core_to_tsp.get_nowait()
    self._process_new_solution(new_solution)
except queue.Empty:
    pass
```

### 2. External Integration

#### Weights & Biases Integration
```python
if self.use_wandb:
    wandb.init(
        project="Genetic-Algorithms-Project",
        entity="dag-malstaf-ku-leuven",
        name=run_name,
        config={
            "population_size": self._parameters.get_population_size(),
            "offspring_size": self._parameters.get_offspring_size(),
            "mutation_rate": self._parameters.get_mutation_rate()
        }
    )
    
    wandb.log({
        "mean_distance": meanObjective,
        "best_distance": bestObjective
    })
```

#### Experiment Management
```python
# Multiple experiment execution
for i in range(500):
    student = r0849537(False, "")
    student.optimize("tour50.csv", i)
```

## Scalability Considerations

### 1. Problem Size Scalability

#### Dynamic Parameter Adjustment
```python
def set_number_of_cities(self, number_of_cities):
    if number_of_cities <= 50:
        self._expexted_number_of_iterations_per_minute = 1400
        self._population_size = 80
        self._immunity = 5
    elif number_of_cities <= 100:
        self._expexted_number_of_iterations_per_minute = 600
        self._population_size = 80
        self._immunity = 5
    elif number_of_cities <= 200:
        self._expexted_number_of_iterations_per_minute = 120
        self._population_size = 80
        self._immunity = 5
    # ... continues for larger problems
```

#### Memory Scalability
- **Efficient data structures**: NumPy arrays for large problems
- **Lazy evaluation**: Calculate distances only when needed
- **Garbage collection**: Minimal object creation

### 2. Performance Scalability

#### Multi-Core Utilization
- **Thread-based parallelism**: Utilize all CPU cores
- **Process-based parallelism**: Second core process
- **Load balancing**: Dynamic thread distribution

#### Algorithmic Scalability
- **O(n²) complexity**: Efficient for problems up to 1000 cities
- **Early termination**: Stop optimization when converged
- **Adaptive parameters**: Adjust based on problem characteristics

### 3. Extensibility

#### Plugin Architecture
- **Modular components**: Easy to add new crossover/mutation operators
- **Strategy pattern**: Different optimization strategies
- **Configuration-driven**: Parameter-based algorithm selection

#### Research Integration
- **Paper-based techniques**: Easy to implement new research findings
- **Benchmarking**: Compare with existing algorithms
- **Experiment tracking**: Weights & Biases integration

## Conclusion

The architecture demonstrates a sophisticated approach to TSP optimization with:

1. **Modular Design**: Clear separation of concerns and easy maintenance
2. **High Performance**: JIT compilation, multi-threading, and optimized algorithms
3. **Scalability**: Handles problems from 50 to 1000+ cities efficiently
4. **Extensibility**: Easy to add new optimization techniques
5. **Research Integration**: Incorporates latest academic findings
6. **Parallel Processing**: Multi-threading and multi-processing for performance
7. **Hybrid Approach**: Combines genetic algorithms with local search and metaheuristics

The system successfully balances performance, scalability, and maintainability while providing a robust foundation for TSP optimization research and applications.

---

*This architecture overview is based on analysis of the implemented codebase and research literature on TSP optimization techniques.* 