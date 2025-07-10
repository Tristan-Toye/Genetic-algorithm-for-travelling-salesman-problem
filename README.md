# Genetic Algorithm for Traveling Salesman Problem

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![WandB](https://img.shields.io/badge/WandB-Enabled-orange.svg)](https://wandb.ai/)

A high-performance implementation of a Genetic Algorithm (GA) for solving the Traveling Salesman Problem (TSP) using advanced optimization techniques including local search, multi-threading, and parallel processing.

## 🚀 Features

- **Advanced Genetic Algorithm**: Implements multiple crossover operators (OX1, POS, Heuristic)
- **Local Search Optimization**: 3-opt and 4-opt (double bridge) local search algorithms
- **Multi-threading Support**: Parallel offspring generation and optimization
- **Second Core Processing**: Dedicated secondary process for enhanced optimization
- **Pheromone Matrix**: Ant Colony Optimization inspired pheromone-based solution generation
- **Population Management**: Dynamic population sizing and immunity mechanisms
- **Performance Monitoring**: Integration with Weights & Biases for experiment tracking
- **Multiple Problem Sizes**: Optimized parameters for TSP instances from 50 to 1000+ cities

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Architecture](#architecture)
- [Algorithm Details](#algorithm-details)
- [Performance](#performance)
- [Contributing](#contributing)
- [License](#license)

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- NumPy
- Numba (for JIT compilation)
- Weights & Biases (optional, for experiment tracking)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Genetic-algorithm-for-travelling-salesman-problem.git
cd Genetic-algorithm-for-travelling-salesman-problem
```

2. Install dependencies:
```bash
pip install numpy numba wandb
```

## 🚀 Quick Start

### Basic Usage

```python
from main import r0849537

# Initialize the GA solver
solver = r0849537(use_wandb=False)

# Solve a TSP instance
result = solver.optimize("tour50.csv")
```

### With Weights & Biases Tracking

```python
# Enable experiment tracking
solver = r0849537(use_wandb=True, run_name="experiment_1")
result = solver.optimize("tour100.csv")
```

## 📖 Usage

### Running Multiple Experiments

```python
# Run 500 experiments on tour50.csv
for i in range(500):
    student = r0849537(False, "")
    student.optimize("tour50.csv", i)
```

### Available TSP Instances

The repository includes several TSP instances:
- `tour50.csv` - 50 cities
- `tour100.csv` - 100 cities  
- `tour200.csv` - 200 cities
- `tour500.csv` - 500 cities
- `tour750.csv` - 750 cities
- `tour1000.csv` - 1000 cities

### Configuration

The algorithm automatically adjusts parameters based on problem size:
- Population size: 60-80 individuals
- Offspring size: 6-15 per generation
- Local search iterations: Optimized per problem size
- Immunity mechanisms: 3-5 elite individuals

## 🏗️ Architecture

### Core Components

- **`main.py`**: Entry point and experiment orchestration
- **`TravellingSalesMan.py`**: Main GA implementation
- **`Parameters.py`**: Dynamic parameter management
- **`Variation.py`**: Crossover operators and offspring generation
- **`local_search.py`**: 3-opt and 4-opt optimization
- **`Individual.py`**: TSP solution representation
- **`Population.py`**: Population management and selection
- **`SecondCoreProcess.py`**: Parallel optimization process

### Key Classes

```python
class r0849537:
    """Main GA solver class"""
    
class TravellingSalesMan:
    """Core genetic algorithm implementation"""
    
class Parameters:
    """Dynamic parameter management based on problem size"""
    
class Variation:
    """Crossover operators and offspring generation"""
```

## 🔬 Algorithm Details

### Genetic Algorithm Components

1. **Initialization**
   - Random initialization (30% of population)
   - Nearest neighbor initialization (30% of population)
   - Pheromone-based initialization (remaining)

2. **Selection**
   - Tournament selection with k=5
   - Parent pair selection for crossover

3. **Crossover Operators**
   - **OX1 (Order Crossover)**: Preserves order and position
   - **POS (Position-based Crossover)**: Preserves absolute positions
   - **Heuristic Crossover**: Distance-based edge recombination

4. **Mutation**
   - **SIM (Single Insertion Mutation)**: 40% probability
   - **DM (Displacement Mutation)**: 20% probability  
   - **IVM (Inversion Mutation)**: 40% probability

5. **Local Search**
   - **3-opt**: Primary local optimization
   - **4-opt (Double Bridge)**: Secondary optimization
   - Don't-look bits for efficiency

6. **Population Management**
   - Elitism with immunity mechanisms
   - Dynamic population sizing
   - Convergence detection

### Advanced Features

- **Pheromone Matrix**: Inspired by Ant Colony Optimization
- **Second Core Process**: Dedicated parallel optimization
- **Multi-threading**: Parallel offspring generation
- **JIT Compilation**: Numba-optimized critical paths

## 📊 Performance

### Benchmark Results

| Problem Size | Best Distance | Mean Distance | Runtime |
|-------------|---------------|---------------|---------|
| 50 cities   | 25,434        | ~25,500       | ~1 min  |
| 100 cities  | 78,201        | ~78,500       | ~2 min  |
| 200 cities  | 36,631        | ~37,000       | ~5 min  |
| 500 cities  | 130,058       | ~131,000      | ~15 min |
| 750 cities  | 164,182       | ~165,000      | ~25 min |
| 1000 cities | 160,408       | ~161,000      | ~40 min |

### Optimization Features

- **Convergence Detection**: Automatic stopping criteria
- **Memory Management**: Efficient data structures
- **Scalability**: Handles problems up to 1000+ cities
- **Reproducibility**: Deterministic results with fixed seeds

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/

# Run linting
flake8 .
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Based on research from the Evolutionary Algorithms course
- Inspired by various TSP optimization papers in the `docs/` folder
- Uses Weights & Biases for experiment tracking
- Implements techniques from Lin-Kernighan and other TSP literature

## 📞 Contact

For questions or support, please open an issue on GitHub or contact the maintainers.

---

**Note**: This implementation is optimized for academic research and educational purposes. For production use, consider additional testing and validation. 