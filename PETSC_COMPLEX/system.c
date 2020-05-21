/**
 * @file    system.c
 * @brief   This file contains functions for constructing matrix, RHS, initial guess
 *
 * @author  Xin Jing <xjing30@gatech.edu>
 *          Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
 * 
 * Copyright (c) 2020 Material Physics & Mechanics Group at Georgia Tech.
 */

#include "system.h"

/**
 * @brief   Read_parameters
 *
 *          Read and set parameters from command line
 */

void Read_parameters(petsc_complex* system, int argc, char **argv) {    
    int rank; 
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); 

    PetscInt p, i; 
    PetscReal Nr, Dr, val; 

    system->order = 6;                                  // store half order
    system->numPoints_x = 36;                           // system size in x direction
    system->numPoints_y = 36;                           // system size in y direction
    system->numPoints_z = 36;                           // system size in z direction

    if (argc < 7) {
        PetscPrintf(PETSC_COMM_WORLD, "Wrong inputs\n"); 
        exit(-1); 
    } else {
        system->pc = atoi(argv[1]); 
        system->solver_tol = atof(argv[2]); 
        system->m = atoi(argv[3]); 
        system->p = atoi(argv[4]); 
        system->omega = atof(argv[5]); 
        system->beta = atof(argv[6]); 
    }

    if (system->pc >1 || system->pc <0){
        PetscPrintf(PETSC_COMM_WORLD, "Nonexistent precondition\n"); 
        exit(-1); 
    }

    //coefficients of the laplacian
    system->coeffs[0] = 0; 
    for (p=1; p<=system->order; p++)
        system->coeffs[0]+= ((PetscReal)1.0/(p*p)); 

    system->coeffs[0]*=((PetscReal)3.0); 

    for (p=1; p<=system->order; p++) {
        Nr=1; 
        Dr=1; 
        for(i=system->order-p+1; i<=system->order; i++)
            Nr*=i; 
        for(i=system->order+1; i<=system->order+p; i++)
            Dr*=i; 
        val = Nr/Dr;  
        system->coeffs[p] = (PetscReal)(-1*pow(-1, p+1)*val/(p*p*(1))); 
    }

    for (p=0; p<=system->order; p++) {
        system->coeffs[p] = system->coeffs[p]/(2*M_PI); 
        // so total (-1/4*pi) factor on fd coeffs
    }  

    PetscPrintf(PETSC_COMM_WORLD, "***************************************************************************\n"); 
    PetscPrintf(PETSC_COMM_WORLD, "                           INPUT PARAMETERS                                \n"); 
    PetscPrintf(PETSC_COMM_WORLD, "***************************************************************************\n"); 
    PetscPrintf(PETSC_COMM_WORLD, "FD Order    : %d\n", system->order * 2);
    PetscPrintf(PETSC_COMM_WORLD, "Solver      : AAR\n"); 
    if (system->pc == 1)
        PetscPrintf(PETSC_COMM_WORLD, "system_pc   : Block-Jacobi using ICC(0)\n"); 
    else
        PetscPrintf(PETSC_COMM_WORLD, "system_pc   : Jacobi \n"); 
    PetscPrintf(PETSC_COMM_WORLD, "solver_tol  : %e \n", system->solver_tol); 
    PetscPrintf(PETSC_COMM_WORLD, "m           : %d\n", system->m); 
    PetscPrintf(PETSC_COMM_WORLD, "p           : %d\n", system->p); 
    PetscPrintf(PETSC_COMM_WORLD, "omega       : %lf\n", system->omega); 
    PetscPrintf(PETSC_COMM_WORLD, "beta        : %lf\n", system->beta); 
    PetscPrintf(PETSC_COMM_WORLD, "***************************************************************************\n"); 

    return; 
}

void Setup_and_Initialize(petsc_complex* system, int argc, char **argv) {
    Read_parameters(system, argc, argv); 
    Objects_Create(system);      
}

/**
 * @brief   Objects_Create
 *
 *          Create all required objects in system
 */

void Objects_Create(petsc_complex* system) {
    PetscInt n_x = system->numPoints_x; 
    PetscInt n_y = system->numPoints_y; 
    PetscInt n_z = system->numPoints_z; 
    PetscInt o = system->order; 

    PetscInt gxdim, gydim, gzdim, xcor, ycor, zcor, lxdim, lydim, lzdim, nprocx, nprocy, nprocz, gidx; 
    Mat A; 
    PetscMPIInt comm_size; 
    PetscScalar ***r, val;
    MPI_Comm_size(PETSC_COMM_WORLD, &comm_size); 

    DMDACreate3d(PETSC_COMM_WORLD, DM_BOUNDARY_PERIODIC, DM_BOUNDARY_PERIODIC, DM_BOUNDARY_PERIODIC, DMDA_STENCIL_STAR, n_x, n_y, n_z, 
    PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE, 1, o, 0, 0, 0, &system->da);   // create a pattern and communication layout
    DMSetUp(system->da);
    DMDAGetCorners(system->da, &xcor, &ycor, &zcor, &lxdim, &lydim, &lzdim); 

    DMCreateGlobalVector(system->da, &system->RHS);                          // using the layour of da to create vectors RHS
    VecDuplicate(system->RHS, &system->AAR); 

    PetscRandom rnd; 
    unsigned long seed; 
    int rank, i, j, k, t = 0; 
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); 

    // RHS vector
    PetscRandomCreate(PETSC_COMM_WORLD, &rnd); 
    PetscRandomSetFromOptions(rnd); 
    
    // Random RHS vector
    // seed=0;  
    // PetscRandomSetSeed(rnd,seed);
    // PetscRandomSeed(rnd);

    // VecSetRandom(system->RHS,rnd);  
    // PetscRandomDestroy(&rnd);

    // generating RHS independent of number of processors
    VecAssemblyBegin(system->RHS);
    DMDAVecGetArray(system->da, system->RHS, &r); 
    for (k = zcor; k < lzdim+zcor; k++)
        for (j = ycor; j < lydim+ycor; j++)
            for (i = xcor; i < lxdim+xcor; i++){
                gidx = (k)*n_x*n_y + (j)*n_x + (i);
                PetscRandomSetSeed(rnd, gidx + 1);
                PetscRandomSeed(rnd);
                PetscRandomGetValue(rnd, &val);
                r[k][j][i] = val;
            }
    DMDAVecRestoreArray(system->da, system->RHS, &r);
    VecAssemblyEnd(system->RHS);
    PetscRandomDestroy(&rnd); 

    // Initial all ones guess
    VecSet(system->AAR, 1.0); 

    if (comm_size == 1 ) {
        DMCreateMatrix(system->da, &system->helmholtzOpr); 
        DMSetMatType(system->da, MATSEQSBAIJ); // sequential symmetric block sparse matrices
    } else  {
        DMCreateMatrix(system->da, &system->helmholtzOpr); 
        DMSetMatType(system->da, MATMPISBAIJ); // distributed symmetric sparse block matrices
    }

}


/**
 * @brief   Objects_Destroy
 *
 *          Destroy all objects inside system
 */

void Objects_Destroy(petsc_complex *system) {
    DMDestroy(&system->da); 
    VecDestroy(&system->RHS); 
    VecDestroy(&system->AAR); 
    MatDestroy(&system->helmholtzOpr); 
    return; 
}


/**
 * @brief   ComputeMatrixA
 *
 *          Compute Matrix A
 */

void ComputeMatrixA(petsc_complex* system) {
    PetscInt i, j, k, l, colidx, gxdim, gydim, gzdim, xcor, ycor, zcor, lxdim, lydim, lzdim, nprocx, nprocy, nprocz; 
    MatStencil row; 
    MatStencil* col; 
    PetscScalar* val; 
    PetscInt o = system->order;  
    int rank; 
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); 
    double Qr = -0.134992, Qi = -0.070225; // real and imag for complex diagonal matrix

    PetscScalar Dinv_factor; 
    Dinv_factor = (system->coeffs[0]) + Qr + PETSC_i * Qi; // (-1/4pi)(Lap)+Q = Diag term
    system->Dinv_factor = 1/Dinv_factor; 

    DMDAGetInfo(system->da, 0, &gxdim, &gydim, &gzdim, &nprocx, &nprocy, &nprocz, 0, 0, 0, 0, 0, 0);  // only get xyz dimension and number of processes in each direction
    PetscPrintf(PETSC_COMM_WORLD, "nprocx: %d, nprocy: %d, nprocz: %d\n", nprocx, nprocy, nprocz); 

    DMDAGetCorners(system->da, &xcor, &ycor, &zcor, &lxdim, &lydim, &lzdim); 
    // printf("rank: %d, xcor: %d, ycor: %d, zcor: %d, lxdim: %d, lydim: %d, lzdim: %d\n", rank, xcor, ycor, zcor, lxdim, lydim, lzdim); 
    // Returns the global (x, y, z) indices of the lower left corner and size of the local region, excluding ghost points.

    MatScale(system->helmholtzOpr, 0.0); 
    // MatView(system->helmholtzOpr, PETSC_VIEWER_STDOUT_WORLD); 

    PetscMalloc(sizeof(MatStencil)*(o*6+1), &col); 
    PetscMalloc(sizeof(PetscScalar)*(o*6+1), &val); 

    // within each local subblock e.g. from xcor to xcor+lxdim-1. using global indexing
    for(k=zcor; k<zcor+lzdim; k++) {
        for(j=ycor; j<ycor+lydim; j++) {
            for(i=xcor; i<xcor+lxdim; i++) {
                row.k = k; row.j = j, row.i = i; 

                colidx=0; 
                col[colidx].i=i; col[colidx].j=j; col[colidx].k=k; 
                val[colidx++] = system->coeffs[0] + Qr + PETSC_i * Qi;
                for(l=1; l<=o; l++) {
                    // z
                    col[colidx].i=i; col[colidx].j=j; col[colidx].k=k-l; 
                    val[colidx++]=system->coeffs[l]; 
                    col[colidx].i=i; col[colidx].j=j; col[colidx].k=k+l; 
                    val[colidx++]=system->coeffs[l]; 
                    //y 
                    col[colidx].i=i; col[colidx].j=j-l; col[colidx].k=k; 
                    val[colidx++]=system->coeffs[l]; 
                    col[colidx].i=i; col[colidx].j=j+l; col[colidx].k=k; 
                    val[colidx++]=system->coeffs[l]; 
                    // x
                    col[colidx].i=i-l; col[colidx].j=j; col[colidx].k=k; 
                    val[colidx++]=system->coeffs[l]; 
                    col[colidx].i=i+l; col[colidx].j=j; col[colidx].k=k; 
                    val[colidx++]=system->coeffs[l]; 
                }
                MatSetValuesStencil(system->helmholtzOpr, 1, &row, 6*o+1, col, val, ADD_VALUES); // ADD_VALUES, add values to any existing value
            }
        }
    }
    MatAssemblyBegin(system->helmholtzOpr, MAT_FINAL_ASSEMBLY); 
    MatAssemblyEnd(system->helmholtzOpr, MAT_FINAL_ASSEMBLY); 

    PetscFree(col);
    PetscFree(val);

}
