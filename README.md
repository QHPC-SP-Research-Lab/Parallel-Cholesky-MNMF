# Parallel-Cholesky-MNMF

A parallel kernel based on Cholesky decomposition to accelerate Multichannel Non-Negative Matrix Factorization (MNMF).

## Overview
Multichannel Source Separation has been a popular topic, and recently proposed methods based on the local Gaussian model (LGM) have provided promising results despite its high computational cost when several sensors are used. The main reason being due to inversion of a spatial covariance matrix, with a complexity of $O(I^3)$, being $I$ the number of sensors. This drawback limits the practical application of this approach for tasks such as sound field reconstruction or virtual reality, among others.

In this repository, we present a numerical approach to reduce the complexity of the Multichannel NMF to address the task of audio source separation for scenarios with a high number of sensors such as High Order Ambisonics (HOA) encoding. In particular, we propose a parallel driver to compute the multiplicative update rules in MNMF approaches. The proposed driver has been designed to work on both sequential and multi-core computers, as well as Graphics Processing Units (GPUs). The proposed software was written in C/CUDA languages and can be called from numerical computing environments.

## Implementation
Our proposed solution tries to reduce the computational cost of the multiplicative update rules by using the Cholesky decomposition and by solving several triangular equation systems. The proposal has been evaluated for different scenarios with promising results in terms of execution times for both CPU and GPU.

## Getting Started
* Clone the repository
* Follow the instructions in the "Installation" section to setup dependencies
* Run the **Lanza*.sh** files provided to understand the usage of the library

## Installation
The library has the following dependencies:
* CMake
* C99 compiler
* OpenMP (for multi-threading support)
* CUDA/cuBLAS (for GPU support)
* Intel Math Kernel Library (MKL) (for CPU support)

## Contributing
We welcome contributions to this project. Please fork the repository and submit a pull request to the master branch.

## Reference
If you use this library in your research, please cite the following paper:

> Antonio J. Muñoz-Montoro, Julio J. Carabias-Orti, Daniele Salvati et al. An efficient parallel kernel based on Cholesky decomposition to accelerate Multichannel Non-Negative Matrix Factorization, 17 October 2022, PREPRINT (Version 1) available at Research Square [https://doi.org/10.21203/rs.3.rs-2152303/v1]

## License
This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).
