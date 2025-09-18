# Augmented Methods for Binary Classification

This repository provides implementations of augmented optimization methods for binary classification problems, focusing on **Support Vector Machines (SVM)** and **Logistic Regression**.

## Motivation

Binary classification is central to many machine learning applications, but standard solvers often struggle with efficiency and scalability on large or constrained problems.  

This project explores **augmented methods**—such as the Augmented Lagrangian framework, Accelerated Gradient Descent, and L-BFGS—to improve robustness, convergence speed, and classification accuracy. By testing across synthetic and real-world datasets, the goal was to evaluate how these methods perform compared to traditional approaches and highlight practical trade-offs in solver design.

## Methods

- **Augmented Lagrangian Method for Soft-Margin SVM (ALM-SVM)**  
  - Constrained optimization handled via dual variables + quadratic penalties  
  - Projected gradient descent with backtracking line search  
  - Adaptive penalty parameter for stable convergence  

- **Binary Logistic Regression Optimization**
  - **Accelerated Gradient Descent (AGD)** with momentum and Lipschitz-based stepsize  
  - **Limited-Memory BFGS (L-BFGS)** quasi-Newton solver with Wolfe line search  

## Datasets

- **SVM Experiments:** `randData_rho02`, `randData_rho08`
- **Logistic Regression Experiments:** `gisette`, `rcv1`, `realsim`  

## Results
- **SVM**
  - **ALM:**
    - Achieved **97.5% accuracy** on `rho02`  
    - Achieved **88% accuracy** on `rho08`  
  
- **Logistic Regression**
  - **AGD:**
    - Achieved **93.10% accuracy** on `gisette`
    - Achieved **95.55% accuracy** on `rcv1`
    - Achieved **83.64% accuracy** on `realsim`
  - **L-BFGS:**
    - Achieved **97.8% accuracy** on `gisette`  
    - Achieved **95.55% accuracy** on `rcv1`
    - Achieved **83.64% accuracy** on `realsim`
## Usage

Clone the repo and run experiments:

```bash
git clone https://github.com/yourname/augmented-ml-methods
cd augmented-ml-methods

# Example: run SVM on rho02 dataset
matlab codeForSVM/test_rho02.m

# Example: run Logistic Regression with L-BFGS on test_gisette dataset
matlab codeForLogReg/test_gisette.m
