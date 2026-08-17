# A Diffusion-based Algorithm for Workspace Generation of Highly Articulated Manipulators

Python implementation of the mathematical and numerical framework presented in:

**Y. Wang and G. S. Chirikjian, "A Diffusion-based Algorithm for Workspace Generation of Highly Articulated Manipulators," Proceedings of the 2002 IEEE International Conference on Robotics & Automation (ICRA), Washington, DC, 2002, pp. 1525–1530.**

## Overview

This repository contains a Python implementation of the diffusion-based algorithm proposed by Wang and Chirikjian for generating the workspace density of highly articulated manipulators.

The paper formulates workspace generation as a diffusion process on the motion group `SE(2)`. The corresponding partial differential equation is transformed using the motion-group Fourier transform, resulting in a system of linear differential equations in Fourier space. The solution is then obtained using a matrix exponential and transformed back to recover the workspace density.

This implementation was developed to understand the mathematical derivation in the paper and reproduce its numerical results and figures as closely as possible from the information provided in the paper.

## Method

The workspace density is modeled using the diffusion equation

```text
∂f/∂L =
[ α X̃₁ᴿ + β(X̃₁ᴿ)² + X̃₃ᴿ + ε(X̃₃ᴿ)² ] f
```

where:

* `L` is the manipulator length,
* `α` describes asymmetry in bending,
* `β` describes flexibility,
* `ε` describes extensibility,
* `X̃₁ᴿ` and `X̃₃ᴿ` are differential operators defined on `SE(2)`.

After applying the motion-group Fourier transform, the equation becomes

```text
d f̂ / dL = B f̂
```

with

```text
B =
α η(X̃₁,p)
+ β η(X̃₁,p)²
+ η(X̃₃,p)
+ ε η(X̃₃,p)²
```

The solution used in the implementation is

```text
f̂(p; L) = exp(B(p)L)
```

The inverse motion-group Fourier transform is then used to reconstruct the workspace density in Cartesian coordinates.

The implementation uses the truncated Fourier representation described in the numerical section of the paper.

## What is Implemented

The Python implementation includes:

* Construction of the Fourier-space operators `η(X̃₁,p)` and `η(X̃₃,p)`.
* Construction of the Fourier-space matrix `B(p)`.
* Solution of the diffusion equation using the matrix exponential.
* Reconstruction of positional workspace density.
* Bessel-function based radial reconstruction.
* Parameter studies corresponding to the cases shown in Figure 3 of the paper.
* Workspace and workspace-density comparisons corresponding to Figures 5 and 6.
* Numerical sanity checks for the implemented operators and matrix-exponential formulation.
* The parameter ranges and reported optimum described in the paper.

## Generated Figures

Running the main Python script generates four main outputs.

### Figure 2 — Workspace for Different Numbers of Modules

A Figure-2-style workspace is generated for:

* 1 module
* 2 modules
* 3 modules
* 4 modules

The paper states that each module can reach 16 different states. However, the paper does not provide the numerical local transformations used for the original brute-force enumeration.

Therefore, the implementation uses a **symmetric 16-state surrogate model** for this part.

This means that the generated Figure-2-style plot is intended to reproduce the qualitative growth of the workspace with increasing numbers of modules, rather than claim an exact reconstruction of the original numerical figure.

### Figure 3 — Effects of the Parameters

The implementation reproduces the parameter combinations used in the paper to demonstrate the effects of:

* Manipulator length `L`
* Flexibility parameter `β`
* Extensibility parameter `ε`
* Asymmetry parameter `α`

The generated density plots are intended to reproduce the qualitative behavior described in the paper.

### Figure 5 — Four-module Manipulator

For the four-module example, the paper reports the matched parameters:

```text
L       = 6
α       = 0
β       = 0.12
ε       = 0.08
```

The implementation generates:

1. A brute-force-style workspace representation.
2. A diffusion-based workspace-density representation.

These are displayed side by side for comparison.

### Figure 6 — Five-module Manipulator

For the five-module example, the paper uses:

```text
L       = 7.5
α       = 0
β       = 0.12
ε       = 0.08
```

The implementation again generates the workspace and corresponding diffusion-based workspace density.

## Numerical Settings

The implementation follows the numerical truncation described in the paper:

```text
l_B  = 7
l_B' = 4
p_max = 100
```

The parameter search reported in the paper uses:

```text
β       : 0.02 to 0.74   (step = 0.02)
ε       : 0.00 to 0.80   (step = 0.04)
α       : fixed at 0
```

For the four-module example, the paper reports:

```text
L       = 6
α       = 0
β       = 0.12
ε       = 0.08
```

with a reported minimum cost of approximately:

```text
C = 0.5748
```

These values are also recorded in the implementation.

## Validation

Two numerical checks are included in the code.

### 1. Fourier Operator Check

The implementation verifies the expected structure of `η(X̃₃,p)` from the analytical expression given in the paper.

### 2. Matrix Exponential Check

For the simplified case

```text
α = β = ε = 0
```

the numerical matrix-exponential result is compared against the corresponding Bessel-function expression.

These checks are intended to verify the main mathematical components before performing the workspace-density reconstruction.

## Repository Structure

```text
.
├── highly_articulated_manipulator.py
├── README.md
├── requirements.txt
├── figure2_workspace_modules.png
├── figure3_parameter_effects.png
├── figure5_four_module_comparison.png
└── figure6_five_module_comparison.png
```

## Requirements

The implementation uses:

* Python
* NumPy
* SciPy
* Matplotlib

Install the required packages with:

```bash
pip install -r requirements.txt
```

or:

```bash
pip install numpy scipy matplotlib
```

## Running the Code

Run:

```bash
python highly_articulated_manipulator.py
```

The script will:

1. Run the numerical sanity checks.
2. Print the parameter-search specification.
3. Generate the Figure-2-style workspace.
4. Generate the Figure-3-style parameter study.
5. Generate the four-module comparison.
6. Generate the five-module comparison.

The generated plots are saved as PNG files.

## Notes on Reproduction

The purpose of this repository is to translate the mathematical formulation in the paper into a working numerical implementation and reproduce the reported workspace-density behavior.

The diffusion-based calculations follow the equations and numerical truncation described in the paper.

For the brute-force workspace used in the Figure-2-style visualization, the paper specifies 16 states per module but does not provide the numerical local transformations required to reproduce the original enumeration exactly. A symmetric surrogate model is therefore used for that portion of the implementation.

Accordingly, the repository distinguishes between:

* **Paper-based implementation:** the diffusion equation, Fourier formulation, matrix exponential solution, inverse transform, and parameter values.
* **Surrogate component:** the local 16-state transformations used for the brute-force Figure-2-style workspace.

## Reference

Wang, Y. and Chirikjian, G. S.

**A Diffusion-based Algorithm for Workspace Generation of Highly Articulated Manipulators**

Proceedings of the 2002 IEEE International Conference on Robotics & Automation (ICRA), Washington, DC, 2002, pp. 1525–1530.

## Acknowledgement

This repository was developed as a study and implementation of the mathematical framework presented in the above paper, with the goal of understanding the derivation and translating it into a numerical Python implementation.
