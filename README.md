# 🧩 ARC Solver – Basic GP Engine

This project is a **from-scratch Genetic Programming (GP) solver** designed to tackle tasks from the [ARC (Abstraction and Reasoning Corpus)](https://github.com/fchollet/ARC).  
It includes a minimal GP framework, an ARC-specific DSL, and simple visualization tools.

---

# 📂 Repository Structure

- **apps/** # Simple web interface for the application
  - **css/** # Stylesheets for the web interface
  - **img/** # Images used in the web interface
  - **js/** # JavaScript files for the web interface
  - **testing_interface.html** # HTML file for the testing interface
- **data/** # ARC-like tasks and datasets
  - **evaluation/** # Evaluation set for tasks
  - **training/** # Training set for tasks
  - **evaluation.txt** # Evaluation dataset description
  - **training.txt** # Training dataset description
  - **viewer.html** # HTML viewer for dataset visualization
- **model/** # Core Genetic Programming (GP) engine
  - **GP.py** # Main GP loop implementation
  - **Individual.py** # Representation of individual programs in GP
  - **Testing.py** # Test harness for running experiments
  - **GP_Old.py** # Legacy version of the GP implementation
  - **test_init.py** # Experiments for population initialization
  - **results/** # Stored outputs from experiments
  - **tree.pdf** # Reference for GP tree structure
- **object_representation/** # Domain-specific language (DSL) and task representation
  - **Dsl.py** # Domain-specific language primitives
  - **graph.py** # Graph-based representations for tasks
  - **object.py** # ARC object representation
  - **solver.py** # Task solver using hand written programs
  - **viewer.py** # Visualization tools for task representations
## 🚀 Features

- 🧬 **Basic GP engine** – crossover, mutation, and multiple selection strategies (tournament, proportional, elitism).  
- 🏗 **DSL for ARC** – primitives to manipulate objects and grids.  
- 📊 **Training & evaluation pipeline** – runs ARC tasks in `data/`.  
- 👀 **Visualization** – GP tree reference (`tree.pdf`) and simple browser viewer (`apps/`).  

---

## ▶️ Quick Start

Clone the repo:

```bash
git clone https://github.com/yourusername/arc-gp-solver.git
cd arc-gp-solver
``` 
Run a training example:

```bash

python model/Testting.py 
```

📝 Notes
  - ✅ This is a basic GP implementation for research & learning.
  - 📜 GP_Old.py – legacy version kept for comparison.
  - 🌐 apps/ contains a browser-based testing interface.
