/**
 * @file    AAR.h
 * @brief   This file declares functions for Alternating Anderson Richardson solver
 *
 * @author  Xin Jing  < xjing30@gatech.edu>
 *          Phanish Suryanarayana  < phanish.suryanarayana@ce.gatech.edu>
 * 
 * Copyright (c) 2020 Material Physics & Mechanics Group at Georgia Tech.
 */


#ifndef AAR_H
#define AAR_H

#include "system.h"
#include "tools.h"

void AAR(POISSON *system, 
     void (*Lap_Vec_mult)(POISSON *, double, double *, double *, MPI_Comm),
     void (*Precondition)(double, double *, int), double a,
     double *x, double *rhs, double omega, double beta, int m, int p, 
     int max_iter, double tol, int Np, MPI_Comm comm);

void AndersonExtrapolation(double **DX, double **DF, double *f, double beta_mix, int m, 
     int N, double *am_vec, double **FtF, double *allredvec, double *Ftf, double *svec, MPI_Comm comm); 

#endif