# Genetic Algorithm for Traveling Salesman Problem - Technical Documentation

## Overview

This document provides comprehensive technical documentation for the Genetic Algorithm implementation for solving the Traveling Salesman Problem (TSP). The implementation features advanced optimization techniques including local search, multi-threading, and parallel processing.

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Core Components](#core-components)
3. [Algorithm Implementation](#algorithm-implementation)
4. [Data Structures](#data-structures)
5. [Performance Optimization](#performance-optimization)
6. [API Reference](#api-reference)
7. [Configuration](#configuration)
8. [Troubleshooting](#troubleshooting)

## System Architecture

### High-Level Design

The system follows a modular architecture with the following key components:

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Main Entry    │───▶│  GA Core Engine  │───▶│ Local Search    │
│   (main.py)     │    │(TravellingSalesMan)│   │ (local_search.py)│
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Parameters     │    │   Population     │    │ Second Core     │
│ (Parameters.py) │    │ (Population.py)  │    │(SecondCoreProcess)│
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Variation     │    │   Individual     │    │   Selection     │
│ (Variation.py)  │    │ (Individual.py)  │    │ (Selection.py)  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### Component Responsibilities

- **Main Entry**: Orchestrates experiments and manages execution flow
- **GA Core Engine**: Implements the main genetic algorithm loop
- **Parameters**: Manages dynamic parameter configuration
- **Variation**: Handles crossover and mutation operations
- **Population**: Manages solution populations and selection
- **Individual**: Represents TSP solutions and their operations
- **Local Search**: Implements 3-opt and 4-opt optimization
- **Second Core**: Provides parallel optimization capabilities

## Core Components

### Main Entry Point (`main.py`)

**Class**: `r0849537`

**Purpose**: Main entry point for the genetic algorithm solver.

**Key Methods**:
- `__init__(use_wandb, run_name)`: Initialize solver with optional WandB tracking
- `optimize(filename, index)`: Solve TSP instance from CSV file

**Parameters**:
- `use_wandb` (bool): Enable Weights & Biases tracking
- `run_name` (str): Name for the experiment run
- `filename` (str): Path to TSP instance CSV file
- `index` (int): Experiment index for result tracking

### GA Core Engine (`TravellingSalesMan.py`)

**Class**: `TravellingSalesMan`

**Purpose**: Implements the core genetic algorithm logic.

**Key Methods**:
- `run(population)`: Execute one generation of the genetic algorithm
- `_selection_phase()`: Perform parent selection
- `_variation_phase()`: Generate offspring through crossover
- `_elimination_phase()`: Select next generation

**Dependencies**:
- `Parameters`: For algorithm configuration
- `Variation`: For crossover operations
- `Selection`: For parent selection
- `Elimination`: For population management

### Parameters Management (`Parameters.py`)

**Class**: `Parameters`

**Purpose**: Manages dynamic parameter configuration based on problem size.

**Key Methods**:
- `set_number_of_cities(n)`: Configure parameters for n-city problem
- `get_population_size()`: Get current population size
- `get_offspring_size()`: Get offspring generation count
- `get_mutation_rate()`: Get mutation probability

**Parameter Categories**:
- **Population Parameters**: Size, immunity, convergence thresholds
- **Genetic Operators**: Crossover and mutation rates
- **Local Search**: Optimization parameters
- **Performance**: Threading and timing configurations

## Algorithm Implementation

### Genetic Algorithm Flow

1. **Initialization**
   ```python
   # Create initial population
   population = initialisation.get_initial_population()
   ```

2. **Main Loop**
   ```python
   while not converged:
       # Selection
       parents = selection.select_multiple_pair(population)
       
       # Variation
       offspring = variation.produce_offspring(parents)
       
       # Local Search
       local_search.optimise(offspring)
       
       # Elimination
       population = elimination.eliminate(population, offspring)
   ```

3. **Convergence Detection**
   - Time-based termination
   - Objective value stagnation
   - Population diversity metrics

### Crossover Operators

#### OX1 (Order Crossover)
**Purpose**: Preserves order and position information from parents.

**Implementation**:
```python
def _OX1(self, parent1, parent2):
    # Select random segment
    start, end = random_segment()
    
    # Copy segment from parent1
    child[start:end] = parent1[start:end]
    
    # Fill remaining positions from parent2
    remaining = [city for city in parent2 if city not in child[start:end]]
    child[end:] = remaining[:len(child)-end]
    child[:start] = remaining[len(child)-end:]
```

#### POS (Position-based Crossover)
**Purpose**: Preserves absolute positions from parents.

**Implementation**:
```python
def _POS(self, parent1, parent2):
    # Select random positions
    positions = random_positions()
    
    # Copy selected positions from parent1
    for pos in positions:
        child[pos] = parent1[pos]
    
    # Fill remaining with parent2 sequence
    remaining = [city for city in parent2 if city not in child]
    j = 0
    for i in range(len(child)):
        if child[i] == -1:  # Unassigned position
            child[i] = remaining[j]
            j += 1
```

#### Heuristic Crossover
**Purpose**: Distance-based edge recombination.

**Implementation**:
```python
def _heuristic_crossover(self, parent1, parent2):
    # Build edge map
    edge_map = build_edge_map(parent1, parent2)
    
    # Start from random city
    current_city = random_city()
    
    # Build tour using edge preferences
    while len(tour) < n_cities:
        next_city = select_best_edge(current_city, edge_map)
        tour.append(next_city)
        current_city = next_city
```

### Local Search Algorithms

#### 3-opt Optimization
**Purpose**: Remove three edges and reconnect to find better tours.

**Implementation**:
```python
@jit(boolean(int16[:], boolean[:], float64[:,:], int16[:,:]), nopython=True)
def opt_3(cyclic_representation, dont_look, distance_matrix, ordered_arg_distance_matrix):
    for city_t1 in range(n_cities):
        if dont_look[city_t1]:
            continue
            
        city_t2 = cyclic_representation[city_t1]
        
        for city_t4 in nearest_neighbors[city_t1]:
            # Find city_t3 and city_t5
            # Calculate current and new distances
            # Perform swap if improvement found
```

#### 4-opt (Double Bridge)
**Purpose**: Remove four edges and reconnect for additional optimization.

**Implementation**:
```python
@jit(boolean(int16[:], boolean[:], float64[:,:], int16[:,:]), nopython=True)
def opt_4_double_bridge(cyclic_representation, dont_look, distance_matrix, ordered_arg_distance_matrix):
    # Similar to 3-opt but removes 4 edges
    # Implements double bridge move
```

## Data Structures

### Individual Representation

**Class**: `Individual`

**Purpose**: Represents a TSP solution.

**Key Attributes**:
- `_cyclic_representation`: Array representing tour as successor list
- `_distance`: Cached tour distance
- `_converged`: Convergence status flags

**Key Methods**:
- `get_distance()`: Calculate/return tour distance
- `get_cyclic_representation()`: Return tour representation
- `is_converged()`: Check convergence status
- `copy()`: Create deep copy of individual

### Population Management

**Class**: `Population`

**Purpose**: Manages collection of TSP solutions.

**Key Attributes**:
- `_population`: Array of Individual objects
- `_size`: Current population size
- `_converged_count`: Number of converged individuals

**Key Methods**:
- `get_population_exposed()`: Return population array
- `sort()`: Sort by fitness
- `add_individual(individual)`: Add new individual
- `get_number_of_converged()`: Count converged individuals

### Distance Matrix

**Type**: `numpy.ndarray`

**Purpose**: Stores pairwise distances between cities.

**Structure**:
```python
distance_matrix[i][j] = distance from city i to city j
```

**Optimizations**:
- Pre-computed for efficiency
- Symmetric matrix (distance[i][j] = distance[j][i])
- Used with ordered neighbor lists for local search

## Performance Optimization

### JIT Compilation

**Technology**: Numba

**Purpose**: Accelerate critical computational paths.

**Optimized Functions**:
- Local search algorithms (3-opt, 4-opt)
- Distance calculations
- Crossover operations
- Selection algorithms

**Example**:
```python
@jit(boolean(int16[:], boolean[:], float64[:,:], int16[:,:]), nopython=True)
def opt_3(cyclic_representation, dont_look, distance_matrix, ordered_arg_distance_matrix):
    # Optimized 3-opt implementation
```

### Multi-threading

**Purpose**: Parallelize computationally intensive operations.

**Threaded Operations**:
- Offspring generation
- Local search optimization
- Population improvement

**Implementation**:
```python
def produce_offspring(self, parent1_list, parent2_list):
    threads = []
    
    # Create threads for different crossover types
    if self._parameters.get_number_of_OX1_offspring():
        thread = threading.Thread(target=self._thread_OX1, kwargs=kwargs)
        threads.append(thread)
    
    # Start all threads
    for thread in threads:
        thread.start()
    
    # Wait for completion
    for thread in threads:
        thread.join()
```

### Second Core Process

**Purpose**: Dedicated parallel optimization process.

**Implementation**:
```python
class SecondCoreProcess:
    def __init__(self, reporter, initialisation, population, distance_matrix, 
                 queue_second_core_to_tsp, queue_tsp_to_second_core):
        # Initialize second core process
        
    def run(self):
        # Continuous optimization loop
        while True:
            # Receive solutions from main process
            # Apply local search
            # Send improved solutions back
```

### Memory Management

**Strategies**:
- Efficient data structures (NumPy arrays)
- Lazy evaluation of distances
- Garbage collection optimization
- Memory pooling for frequently allocated objects

## API Reference

### Main Solver API

```python
class r0849537:
    def __init__(self, use_wandb: bool = False, run_name: str = None):
        """
        Initialize the GA solver.
        
        Args:
            use_wandb: Enable Weights & Biases tracking
            run_name: Name for the experiment run
        """
        
    def optimize(self, filename: str, index: int = 0) -> int:
        """
        Solve TSP instance.
        
        Args:
            filename: Path to TSP instance CSV file
            index: Experiment index for result tracking
            
        Returns:
            0 on successful completion
        """
```

### Parameters API

```python
class Parameters:
    def set_number_of_cities(self, number_of_cities: int) -> None:
        """Configure parameters for given problem size."""
        
    def get_population_size(self) -> int:
        """Get current population size."""
        
    def get_offspring_size(self) -> int:
        """Get offspring generation count."""
        
    def get_mutation_rate(self) -> float:
        """Get mutation probability."""
```

### Individual API

```python
class Individual:
    def __init__(self, number_of_cities: int, distance_matrix: np.ndarray):
        """Initialize individual with given number of cities."""
        
    def get_distance(self) -> Tuple[float, bool]:
        """Get tour distance and convergence status."""
        
    def get_cyclic_representation(self) -> np.ndarray:
        """Get tour as cyclic representation."""
        
    def is_converged(self) -> bool:
        """Check if individual has converged."""
```

## Configuration

### Parameter Tuning

The algorithm automatically adjusts parameters based on problem size:

**Small Problems (≤50 cities)**:
- Population size: 80
- Expected iterations/minute: 1400
- Immunity: 5
- Pheromone offspring: 15

**Medium Problems (51-200 cities)**:
- Population size: 80
- Expected iterations/minute: 600-120
- Immunity: 5
- Pheromone offspring: 15

**Large Problems (201-500 cities)**:
- Population size: 60
- Expected iterations/minute: 100
- Immunity: 3
- Pheromone offspring: 10

**Very Large Problems (501+ cities)**:
- Population size: 60
- Expected iterations/minute: 50
- Immunity: 3
- Pheromone offspring: 10

### Debug Configuration

**Debug Flags** (in `Debug.py`):
- `DEBUG`: Enable debug assertions
- `PRINT_STAGES`: Print algorithm stage information
- `PRINT_THREADING`: Print threading information
- `SEQUENTIAL`: Disable multi-threading
- `ENABLE_DOUBLE_BRIDGE`: Enable 4-opt optimization

### Performance Configuration

**Threading Options**:
- `SEQUENTIAL = False`: Enable multi-threading
- Thread count: Automatic based on CPU cores
- Thread synchronization: Queue-based communication

**Memory Configuration**:
- Array data types: Optimized for memory usage
- Cache management: Automatic distance caching
- Garbage collection: Minimal object creation

## Troubleshooting

### Common Issues

**1. Memory Errors**
- **Symptom**: OutOfMemoryError or excessive memory usage
- **Cause**: Large problem sizes or inefficient data structures
- **Solution**: Reduce population size or enable garbage collection

**2. Convergence Issues**
- **Symptom**: Algorithm doesn't converge or converges too quickly
- **Cause**: Improper parameter configuration
- **Solution**: Adjust mutation rates or population diversity

**3. Performance Problems**
- **Symptom**: Slow execution or poor results
- **Cause**: Inefficient local search or threading issues
- **Solution**: Enable JIT compilation or adjust thread count

**4. Threading Issues**
- **Symptom**: Deadlocks or race conditions
- **Cause**: Improper thread synchronization
- **Solution**: Use sequential mode or fix queue handling

### Debugging Tools

**1. Debug Mode**
```python
# Enable debug mode
from Debug import DEBUG
DEBUG = True
```

**2. Performance Profiling**
```python
# Profile execution
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()
# Run algorithm
profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats()
```

**3. Memory Profiling**
```python
# Monitor memory usage
import psutil
import os

process = psutil.Process(os.getpid())
memory_info = process.memory_info()
print(f"Memory usage: {memory_info.rss / 1024 / 1024} MB")
```

### Performance Monitoring

**Weights & Biases Integration**:
```python
# Enable tracking
solver = r0849537(use_wandb=True, run_name="debug_run")

# Monitor metrics
wandb.log({
    "mean_distance": mean_objective,
    "best_distance": best_objective,
    "population_size": population_size,
    "convergence_rate": convergence_rate
})
```

**Custom Metrics**:
```python
# Track custom performance metrics
def log_performance_metrics(iteration, population, time_elapsed):
    metrics = {
        "iteration": iteration,
        "best_distance": population[0].get_distance()[0],
        "mean_distance": np.mean([ind.get_distance()[0] for ind in population]),
        "time_elapsed": time_elapsed,
        "converged_count": sum(1 for ind in population if ind.is_converged())
    }
    return metrics
```

---

*This documentation follows Google's technical documentation standards and provides comprehensive coverage of the Genetic Algorithm implementation for the Traveling Salesman Problem.* 