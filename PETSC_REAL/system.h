/**
 * @file    system.h
 * @brief   This file declares data structure and the functions for system initialization
 *
 * @author  Xin Jing <xjing30@gatech.edu>
 *          Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
 * 
 * Copyright (c) 2020 Material Physics & Mechanics Group at Georgia Tech.
 */

#ifndef SYSTEM_H
#define SYSTEM_H

static char help[] = "Alternating Anderson Richardson (AAR) code\n";

#include "petsc.h"
#include "petscksp.h"
#include "petscvec.h"  
#include "assert.h"
#include "petscdmda.h"
#include <petsctime.h>
#include <mpi.h>
#include "petscsys.h"
#include "math.h"
#include "mkl_lapacke.h"
#include "mkl.h"

#define MAX_ORDER 10
#define M_PI 3.14159265358979323846

typedef struct {
    PetscInt  numPoints_x;
    PetscInt  numPoints_y;
    PetscInt  numPoints_z;
    PetscInt  order;        // half FD order
    PetscInt pc;     
    PetscReal coeffs[MAX_ORDER+1];

    PetscReal solver_tol;
    PetscInt m;
    PetscInt p;
    PetscReal omega;
    PetscReal beta;
    PetscScalar Dinv_factor;

    DM da;
    Vec RHS;
    Vec AAR;

    Mat poissonOpr;   
}petsc_real;

void Setup_and_Initialize(petsc_real* system, int argc, char **argv);
void Read_parameters(petsc_real* system, int argc, char **argv);
void Objects_Create(petsc_real* system);
void ComputeMatrixA(petsc_real* system);
void Objects_Destroy(petsc_real* system);

#endif