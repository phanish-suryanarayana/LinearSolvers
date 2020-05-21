#Instructions to run the MPI_Real code:

This code is for solving **real-valued** systems of equations using the AAR method, with the Poisson equation provided as an example. 

1. Code tested using mvapich2/2.1 and intel/19.0 compilers for MPI and C/C++. 

2. Compile the code by typing "make" in the root directory $MPI_REAL.
    
3. Output file : "test.out" - prints number of processors used, input parameters, time taken by the solver.
    If no output files needed, delete the -log_summary in the command line.

4. Parameters needs to be entered in the command line
    + solver_tol - convergence tolerance for the solver
    + m          - solver parameter (history)
    + p          - solver parameter (frequency of Anderson extrapolation, if p_aar=1, then it is AR)
    + omega      - solver parameter (Richardson relaxation)
    + beta       - solver parameter (Anderson relaxation)
    
5. Running the code: (e.g. for 8 processors)
        
    mpirun -n 8 ./mpi_real tol m p omega beta
            
    `mpirun -n 8 ./mpi_real 1e-6 13 11 1.0 0.9 -log_summary>test.out`
	   
6. The executable (mpi_real) is created in the root directory and the source code is in main.c system.c(.h) tools.c(h) AAR.c(.h)

7. Users should define their own Lap_Vec_mult and Precondition function.