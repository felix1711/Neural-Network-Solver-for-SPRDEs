# Neural Network Solver for Singularly Perturbed Reaction-Diffusion Equations

A neural-network-based numerical approach for solving **Singularly Perturbed One-Dimensional Reaction-Diffusion Equations (SPRDEs)**, with a focus on neural network architecture selection, optimisation strategies and numerical accuracy.

## Overview

Singular perturbation problems arise in several areas of applied mathematics and engineering, including fluid mechanics, chemical reactor theory, elasticity, hydrodynamics and reaction-diffusion processes.

These problems often involve solutions with sharp boundary or interior layers, making them challenging for conventional numerical methods.

This project investigates the use of **feedforward neural networks** to approximate solutions to singularly perturbed 1D reaction-diffusion equations. Two benchmark test problems are implemented and evaluated against their known analytical solutions.

The work focuses on improving training performance through systematic hyperparameter selection and a hybrid optimisation strategy.

## Objectives

- Develop a neural-network-based numerical solver for SPRDEs.
- Investigate the effect of network depth on training loss and model complexity.
- Compare commonly used optimisation algorithms: SGD, RMSprop, Adam and L-BFGS.
- Implement a hybrid **Adam–L-BFGS optimisation strategy** to improve convergence.
- Evaluate numerical solutions against analytical solutions using error metrics and visual comparisons.

## Methodology

### 1. Mathematical Formulation

The problem considers a self-adjoint, second-order singularly perturbed reaction-diffusion equation of the general form:

\[
-\varepsilon v''(u)+\phi(u)v(u)=\omega(u),
\quad 0\leq u\leq 1
\]

subject to Dirichlet boundary conditions:

\[
v(0)=\gamma_1,\qquad v(1)=\gamma_2
\]

where:

- \(\varepsilon\) is a small positive perturbation parameter.
- \(\phi(u)\) is a positive coefficient function.
- \(\omega(u)\) is a prescribed source function.
- \(\gamma_1,\gamma_2\) are boundary values.

### 2. Neural Network Architecture

A feedforward neural network is constructed using **PyTorch**. The input is the independent spatial variable \(u\), and the network predicts the approximate solution \(v(u)\).

The network architecture and training parameters are investigated experimentally.

Key hyperparameters include:

- Number of hidden layers.
- Number of neurons per layer.
- Activation function.
- Optimisation algorithm.
- Initial weights and biases.

### 3. Hyperparameter Selection

**Network depth:** Different configurations with 1–5 hidden layers are evaluated while keeping the number of neurons per layer fixed at 5.

The experiments indicate that 3 and 5 hidden layers achieve comparable training losses. A 3-layer architecture is selected to reduce model complexity while maintaining solution quality.

**Optimiser comparison:** SGD, RMSprop, Adam and L-BFGS are evaluated individually.

A hybrid optimisation strategy is subsequently implemented, alternating 100 Adam iterations with one L-BFGS iteration. This combines Adam's adaptive gradient updates with L-BFGS's quasi-Newton optimisation.

### 4. Training Procedure

The network is trained by minimising a composite objective incorporating the differential-equation residual and boundary-condition errors.

The implementation uses PyTorch for model construction, parameter optimisation and gradient-based training.

The spatial domain is discretised into 1,000 equally spaced training points over \([0,1]\). The network iteratively updates its parameters to minimise the objective function.

## Numerical Experiments

Two benchmark problems from the literature are implemented to evaluate the proposed approach.

### Test Problem 1

- Implements the first benchmark singularly perturbed reaction-diffusion equation.
- Evaluates the predicted solution against its known analytical solution.
- Studies the effect of the perturbation parameter \(\varepsilon\).
- Compares maximum absolute errors across different mesh configurations.

**Results:**

![Test Problem 1 solution comparison](results/test_problem_1/solution_comparison.png)

![Test Problem 1 absolute error](results/test_problem_1/absolute_error.png)

### Test Problem 2

- Implements a second benchmark reaction-diffusion problem.
- Evaluates the neural network's ability to approximate the analytical solution.
- Computes numerical errors for different perturbation parameters and mesh configurations.

**Results:**

![Test Problem 2 solution comparison](results/test_problem_2/solution_comparison.png)

![Test Problem 2 absolute error](results/test_problem_2/absolute_error.png)

## Hyperparameter Analysis

### Hidden Layer Comparison

Training loss is evaluated for networks with different numbers of hidden layers.

![Hidden layer comparison](results/hyperparameter_analysis/layer_comparison.png)

### Optimiser Comparison

The hybrid Adam–L-BFGS strategy is compared with individual optimisation algorithms.

| Optimiser | Training Loss |
|---|---:|
| SGD | 0.659543 |
| RMSprop | 0.375030 |
| Adam | 0.365092 |
| L-BFGS | 0.659543 |
| **Hybrid Adam–L-BFGS** | **0.000133** |

The hybrid strategy achieves a substantially lower training loss in the reported experiment.

## Technologies Used

- **Programming:** Python
- **Deep Learning Framework:** PyTorch
- **Numerical Computing:** NumPy
- **Visualisation:** Matplotlib
- **Optimisation:** SGD, RMSprop, Adam and L-BFGS

## Repository Structure

```text
src/              Shared model and training utilities
experiments/      Test Problem 1, Test Problem 2 and hyperparameter studies
results/          Numerical results and visualisations
docs/             Mathematical formulation and methodology
```

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/neural-network-solver-sprde.git

cd neural-network-solver-sprde
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

## Running the Experiments

Run the benchmark problems independently:

```bash
python experiments/test_problem_1.py
```

```bash
python experiments/test_problem_2.py
```

Run the hyperparameter experiments:

```bash
python experiments/hyperparameter_analysis.py
```

## Key Findings

- A 3-hidden-layer architecture provides a favourable balance between training loss and network complexity in the tested configurations.
- Individual optimisers exhibit different convergence behaviour.
- The hybrid Adam–L-BFGS approach substantially reduces training loss in the reported experiment.
- The two benchmark problems provide a basis for evaluating the accuracy of neural-network approximations against analytical solutions.

**Note:** Training loss and numerical solution error are different measures. The reported optimiser losses should not be interpreted as maximum absolute solution errors.
