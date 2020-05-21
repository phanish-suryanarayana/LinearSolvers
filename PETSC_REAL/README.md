#Instructions to run the Petsc_Real code:

This code is for solving **real-valued** systems of equations using the AAR method, with the Poisson equation provided as an example.  

1. Modules required to compile the code:intel/19.0.3, mvapich2/2.3.1, mkl/11.2, petsc/3.11.3 *(must without Complex configuration)*

2. Compile the code by typing "make" in the root directory $PETSC_REAL.
    
3. Output file : "test.out" - prints number of processors used, input parameters, time taken by the solver.

4. Parameters in the command line
    + pc         - 0 or 1 indicates use of Jacobi or Block-Jacobi preconditioning respectively
    + solver_tol - convergence tolerance for the solver
    + m          - solver parameter (history)
    + p          - solver parameter (frequency of Anderson extrapolation, if p_aar=1, then it is AR)
    + omega      - solver parameter (Richardson relaxation)
    + beta       - solver parameter (Anderson relaxation) (no beta for PGR and PL2R, any number in input is fine)
    
5. Running the code: (e.g. for 8 processors)

    mpirun -np np ./petsc_real pc tol  m  p  omega beta 

    `mpirun -np 64 ./petsc_real 0 1e-6 13 11 1.0 0.9 -log_summary>test.out`
	   
(6) The executable (petsc_real) is created in the root directory and the source code is in main.c system.c(.h) tools.c(h) AAR_Real.c(.h)


