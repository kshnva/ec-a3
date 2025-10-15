# Evolutionary Computing: Assignment 3

This project explores the **co-evolution of robot morphology and neural control** using the [ARIEL](https://github.com/ci-group/ariel) simulation environment and **CMA-ES** optimization.  
Robots evolve both their **body structure** (via a Neural Developmental Encoder) and **controller** (a neural network or CPG) to achieve efficient locomotion across **flat, rugged, and sloped terrains**.

After the best morphology is found, the **controller is further evolved** independently to test whether generative encodings promote **modularity, sparsity, and regularity** in neural structure.

### Highlights
- Hierarchical CMA-ES for joint body–brain evolution  
- Multi-terrain evaluation for robust gait adaptation  
- Quantitative analysis of network structure (efficiency, modularity, symmetry, sparsity)  
- Visualization of fitness progress and structural trends  

### Tech Stack
**Python**, **NumPy**, **SciPy**, **Matplotlib**, **ARIEL (MuJoCo)**

### Results
Evolved robots developed stable, modular morphologies and rhythmic gaits.  
Fixed-body controller evolution showed structural organization but limited fitness gains, emphasizing the need for **co-adaptive body–brain evolution**.
