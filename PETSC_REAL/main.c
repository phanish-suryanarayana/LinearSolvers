
/*=============================================================================================
| Alternating Anderson Richardson (AAR) code and tests in real-valued systems.
| Copyright (C) 2020 Material Physics & Mechanics Group at Georgia Tech.
| 
| Authors: Xin Jing, Phanish Suryanarayana
|
| Last Modified: 3 March 2020
|-------------------------------------------------------------------------------------------*/

#include "system.h"
#include "AAR_Real.h"

int main( int argc, char **argv ) {
    int ierr, i; 
    petsc_real system;
    PetscReal t0,t1, t2, t3;

    PetscInitialize(&argc,&argv,(char*)0,help);

    t0 = MPI_Wtime();

    Setup_and_Initialize(&system, argc, argv);

    // Compute RHS and Matrix for the Poisson equation
    /* Matrix, A = psystem->poissonOpr
       NOTE: For a different problem, other than Poisson equation, provide the matrix through the variable "psystem->poissonOpr"
       and right hand side through "psystem->RHS". */
    ComputeMatrixA(&system);     

    t1 = MPI_Wtime();
    PetscPrintf(PETSC_COMM_WORLD,"\nTime spent in initialization = %.4f seconds.\n",t1-t0);

    PetscPrintf(PETSC_COMM_WORLD,"*************************************************************************** \n \n");
    
    t2 = MPI_Wtime();
    // -------------- AAR solver --------------------------
    AAR(system.poissonOpr, system.AAR, system.RHS, system.omega, 
        system.beta, system.m, system.p, system.solver_tol, 2000, system.pc, system.da);
    t3 = MPI_Wtime();
    PetscPrintf(PETSC_COMM_WORLD,"*************************************************************************** \n \n");   

    Objects_Destroy(&system); 
    ierr = PetscFinalize();
    CHKERRQ(ierr);

    return 0;
}