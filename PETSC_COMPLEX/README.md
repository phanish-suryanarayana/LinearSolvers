# Instructions to run the Petsc_Complex code:

This code is for solving **complex-valued** systems of equations using the AAR method, with the Helmholtz equation provided as an example.  

1. Modules required to compile the code:intel, mvapich2, petsc **(must with Complex configuration)**

2. Compile the code by typing "make" in the root directory $PETSC_COMPLEX.
    
3. Output file : "test.out" - prints number of processors used, input parameters, time taken by the solver.

4. Parameters in the command line
    + pc - 0 or 1 indicates use of Jacobi or Block-Jacobi preconditioning respectively
    + solver_tol - convergence tolerance for the solver
    + m - solver parameter (history)
    + p - solver parameter (frequency of Anderson extrapolation, if p_aar=1, then it is AR)
    + omega - solver parameter (Richardson relaxation)
    + beta - solver parameter (Anderson relaxation)
    
5. Running the code: (e.g. for 8 processors)

    mpirun -np np ./petsc_real pc tol  m  p  omega beta 
    
    `mpirun -n 8  ./petsc_complex 0 1e-6 13 11 1.0 0.9 -log_summary>test.out`
	   
6. The executable (petsc_complex) is created in the root directory and the source code is in main.c system.c(.h) tools.c(h) AAR_Real.c(.h)

7. If you want to use the DDEBUG mode to print out more information, please add -DDEBUG in the CFLAGS. 

8. If you are using PETSC/3.5 (or lower version), please change the 2 *include* option in makefile. FYI, the developer used intel/19.0.5, mvapich2/2.3.2 and petsc/3.12.2-mva2 modules. Please use the complexed version of petsc!