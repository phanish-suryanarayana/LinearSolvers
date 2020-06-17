# LinearSolvers

Collection of solvers for linear systems of equations for both serial and parallel computations. 

Currently, you can find the Alternating Anderson Richardson (AAR) method \[1,2\] implemented in Matlab, PETSc, and standalone C w/ MPI code. Additional details can be found in the README file with each subfolder. Users of the code are expected to cite one of the references below, preferably \[1\].  

**References**

\[1\] Suryanarayana, P., Pratapa, P.P. and Pask, J.E., 2019. Alternating Anderson–Richardson method: An efficient alternative to preconditioned Krylov methods for large, sparse linear systems. Computer Physics Communications, 234, pp.278-285.

\[2\] Pratapa, P.P., Suryanarayana, P. and Pask, J.E., 2016. Anderson acceleration of the Jacobi iterative method: An efficient alternative to Krylov methods for large, sparse linear systems. Journal of Computational Physics, 306, pp.43-54.
