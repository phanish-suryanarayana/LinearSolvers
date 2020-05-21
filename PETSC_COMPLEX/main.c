
/*=============================================================================================
| Alternating Anderson Richardson (AAR) code and tests in complex-valued systems.
| Copyright (C) 2020 Material Physics & Mechanics Group at Georgia Tech.
| 
| Authors: Xin Jing, Phanish Suryanarayana
|
| Last Modified: 3 March 2020
|-------------------------------------------------------------------------------------------*/

#include "system.h"
#include "AAR_Complex.h"

int main( int argc, char **argv ) {
    int ierr, i; 
    petsc_complex system;
    PetscReal t0,t1, t2, t3;

    PetscInitialize(&argc,&argv,(char*)0,help);

    t0 = MPI_Wtime();

    Setup_and_Initialize(&system, argc, argv);

    // Compute RHS and Matrix for the helmholtz equation
    /* Matrix, A = system->helmholtzOpr
       NOTE: For a different problem, other than helmholtz equation, provide the matrix through the variable "system->helmholtzOpr"
       and right hand side through "system->RHS". */
    ComputeMatrixA(&system);

    t1 = MPI_Wtime();
    PetscPrintf(PETSC_COMM_WORLD,"\nTime spent in initialization = %.4f seconds.\n",t1-t0);

    PetscPrintf(PETSC_COMM_WORLD,"*************************************************************************** \n \n");
    
    // -------------- AAR solver --------------------------
    VecSet(system.AAR, 1.0);
    t2 = MPI_Wtime();
    AAR_Complex(system.helmholtzOpr, system.AAR, system.RHS, system.omega, 
        system.beta, system.m, system.p, system.solver_tol, 2000, system.pc, system.da);
    t3 = MPI_Wtime();
    PetscPrintf(PETSC_COMM_WORLD,"*************************************************************************** \n \n");   

    t1 = MPI_Wtime();
    PetscPrintf(PETSC_COMM_WORLD,"Total wall time = %.4f seconds.\n\n",t1-t0);
    
    Objects_Destroy(&system); 
    ierr = PetscFinalize();
    CHKERRQ(ierr);

    return 0;
}