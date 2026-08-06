# CutE-FSI
Short abstract

## Summary

## Installation instructions and executing the code

### Requirements
#### deal.II
Install deal.II 9.7.0 or newer via https://dealii.org/ \
Download: https://dealii.org/current_release/download/ \
Installation instructions: https://dealii.org/current/readme.html 

Alternatively, you can use the candi install script: https://github.com/dealii/candi 

#### Other dependencies:
* PETSc (https://petsc.org/)
* MUMPS (https://mumps-solver.org/)
* MPI (https://www.mpi-forum.org/)

### Install and run
```
$ git clone https://github.com/akwenske/CutE-FSI.git
$ cd CutE-FSI
$ mkdir build && cd build
$ cmake -DDEAL_II_DIR=<path-to-deal-ii> -DCMAKE_BUILD_TYPE=Release ..
$ make -j <N> && mpirun -np <N> ./cute-fsi
```

## First steps

[# Automated testing ????]:

[Configuration: Sphere]

[Configuration: Horizontal Interface]

## Documentation

[We could briefly explain in one sentence the main functions of the code.]

To generate the documentation run

```
$ doxygen Doxyfile
```
in the CutE-FSI directory.

## Contributing

We are happy for further contributions from others. In case of questions, please contact us. 

## License
The license is GNU LESSER GENERAL PUBLIC LICENSE (LGPL) Version 2.1. Detailed information can be found [here](LICENSE).

## References

[[1]](https://doi.org/10.48550/arXiv.2603.25279) S. Frei and T. Knoke and M. C. Steinbach and A.-K. Wenske and T. Wick; Numerical Analysis of a Cut Finite Element Approach for Fully Eulerian Fluid-Structure Interaction with Fixed Interface, arXiv, 2026, https://arxiv.org/abs/2603.25279
      
[[2]](https://doi.org/10.3934/acse.2025005) S. Frei, T. Knoke, M. C. Steinbach, A.-K. Wenske, T. Wick; Numerical Simulations of Fully Eulerian Fluid-Structure Contact Interaction using a Ghost-Penalty Cut Finite Element Approach, Advances in Computational Science and Engineering (ACSE), Vol. 3, 2025, pp. 74-94

[[3]](https://doi.org/10.1007/978-3-031-86173-4_32) S. Frei, T. Knoke, M.C. Steinbach, A.-K. Wenske, T. Wick; Modeling And Numerical Simulation Of Fully Eulerian Fluid-Structure Interaction Using Cut Finite Elements. In: Numerical Mathematics and Advanced Applications (ENUMATH 2023), Volume 1, pp. 313-323, Apr 2025
