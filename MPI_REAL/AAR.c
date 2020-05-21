/**
 * @file    AAR.c
 * @brief   This file contains functions for Alternating Anderson Richardson solver
 *
 * @author  Xin Jing  < xjing30@gatech.edu>
 *          Phanish Suryanarayana  < phanish.suryanarayana@ce.gatech.edu>
 * 
 * Copyright (c) 2020 Material Physics & Mechanics Group at Georgia Tech.
 */


#include "AAR.h"

/**
 * @brief   Alternating Anderson Richardson solver
 *
 *          This function is designed to solve linear system Ax = b
 *          PoissonResidual is for matrix vector multiplication A * x, which could be 
 *          replaced by any user defined function 
 *          Precondition is for applying precondition, which could be replaced by any
 *          user defined precondition function
 *
 *          a        : constant for Lap_Vec_mult and precondition function
 *          x        : initial guess and final solution
 *          rhs      : right hand side of linear system, i.e. b
 *          omega    : relaxation parameter (for Richardson update)
 *          beta     : extrapolation parameter (for Anderson update)
 *          m        : no. of previous iterations to be considered in extrapolation, Anderson history
 *          p        : Perform Anderson extrapolation at every p th iteration
 *          max_iter : maximum number of iterations
 *          tol      : convergence tolerance
 *          Np       : no. of elements in this domain
 *          comm     : MPI communicator
 */


void AAR(POISSON *system,
        void (*Lap_Vec_mult)(POISSON *, double, double *, double *, MPI_Comm),
        void (*Precondition)(double, double *, int), double a,
        double *x, double *rhs, double omega, double beta, int m, int p, 
        int max_iter, double tol, int Np, MPI_Comm comm) 
{
    int rank, i, col, iter; 
    double *f, *x_old, *f_old, **DX, **DF, *am_vec, t0, t1;
    double **FtF, *allredvec, *Ftf, *svec, relres, rhs_norm;  
    //////////////////////////////////////////////////////////

    MPI_Comm_rank(MPI_COMM_WORLD, &rank); 

    // allocate memory to store x(x) in domain
    f = (double*)calloc(Np, sizeof(double));                               // f = inv(M) * res
    x_old = (double*)calloc(Np, sizeof(double));  
    f_old = (double*)calloc(Np, sizeof(double));  
    Ftf = (double*) calloc(m, sizeof(double));                             // DF'*f 
    assert(f !=  NULL &&  x_old != NULL && f_old != NULL && Ftf != NULL); 

    // allocate memory to store DX, DF history matrices and DF'*DF
    DX = (double**) calloc(m, sizeof(double*)); 
    DF = (double**) calloc(m, sizeof(double*)); 
    FtF = (double**) calloc(m, sizeof(double*));
    assert(DF !=  NULL && DX != NULL && FtF != NULL); 

    for (i = 0; i < m; i++) {
        DX[i] = (double*)calloc(Np, sizeof(double)); 
        DF[i] = (double*)calloc(Np, sizeof(double)); 
        FtF[i] = (double*)calloc(m, sizeof(double)); 
        assert(DX[i] !=  NULL && DF[i] !=  NULL && FtF[i] !=  NULL); 
    }
    
    am_vec = (double*)calloc(Np, sizeof(double));  
    allredvec = (double*) calloc(m*m+m, sizeof(double));               
    svec = (double*) calloc(m, sizeof(double));
    assert(am_vec !=  NULL && allredvec != NULL && svec != NULL); 

    iter = 1; 
    relres = tol+1; 
    
    Vector2Norm(rhs, Np, &rhs_norm, comm); 
    tol *= rhs_norm;
    Lap_Vec_mult(system, a, x, f, comm);
    for (i = 0; i < Np; i++)
        f[i] = rhs[i] - f[i];                                              // f = rhs - Ax

#ifdef DEBUG
    Vector2Norm(f, Np, &relres, comm);
    if(!rank) printf("Relative Residual: %g\n", relres/rhs_norm);
#endif

    t0 = MPI_Wtime(); 

    while (relres > tol && iter  <=  max_iter) {
        // Apply precondition here if it is provided by user
        if (Precondition != NULL)
            Precondition(a*(3*system->coeff_lap[0]), f, Np);           // f = inv(M) * f

        if(iter>1) {
            col = ((iter-2) % m); 
            for (i = 0; i < Np; i++){
                DX[col][i] = x[i] - x_old[i];                              // DX[k] = x - x_old
                DF[col][i] = f[i] - f_old[i];                              // DF[k] = f - f_old
            }
        } 

        for (i = 0; i < Np; i++){
            x_old[i] = x[i];                                               // x_old = x
            f_old[i] = f[i];                                               // f_old = f
        }

        if(iter % p == 0 && iter>1) {
            /***********************************
             *  Anderson extrapolation update  *
             ***********************************/

            AndersonExtrapolation(DX, DF, f, beta, m, Np, am_vec, FtF, allredvec, Ftf, svec, comm); 

            for (i = 0; i < Np; i ++)
                x[i] = x_old[i] + beta * f[i] - am_vec[i];                 // x = x_old + beta * f - am_vec

            Lap_Vec_mult(system, a, x, f, comm);
            for (i = 0; i < Np; i++)
                f[i] = rhs[i] - f[i];                                      // f = rhs - Ax

            Vector2Norm(f, Np, &relres, comm);

#ifdef DEBUG
    if(rank == 0) printf("Relative Residual: %g\n", relres/rhs_norm);
#endif

        } else {
            /***********************
             *  Richardson update  *
             ***********************/

            for (i = 0; i < Np; i++)
                x[i] = x_old[i] + omega * f[i];                            // x = x + omega * res

            Lap_Vec_mult(system, a, x, f, comm);   
            for (i = 0; i < Np; i++)
                f[i] = rhs[i] - f[i];                                      // f = rhs - Ax

#ifdef DEBUG
    Vector2Norm(f, Np, &relres, comm);
    if(rank == 0) printf("Relative Residual: %g\n", relres/rhs_norm);
#endif
        }

        iter = iter+1; 
    } 

    t1 = MPI_Wtime(); 
    if(rank  ==  0) {
        if(iter < max_iter && relres  <= tol) {
            printf("AAR converged!:  Iterations = %d, Relative Residual = %g, Time = %.4f sec\n", iter-1, relres/rhs_norm, t1-t0); 
        }
        if(iter>= max_iter) {
            printf("WARNING: AAR exceeded maximum iterations.\n"); 
            printf("AAR:  Iterations = %d, Relative Residual = %g, Time = %.4f sec \n", iter-1, relres/rhs_norm, t1-t0); 
        }
    }

    free(f); 
    free(x_old); 
    free(f_old); 
    free(am_vec); 
  
    for (i = 0; i < m; i++) {
      free(DX[i]); 
      free(DF[i]); 
      free(FtF[i]); 
    }
    free(DX); 
    free(DF); 
    free(FtF); 
    free(Ftf); 
    free(svec); 
    free(allredvec); 
}

/**
 * @brief   Anderson update
 *
 *          am_vec = (DX + beta*DF)*(pinv(DF'*DF)*(DF'*f));
 */

void AndersonExtrapolation(double **DX, double **DF, double *f, double beta, int m, int Np, 
                            double *am_vec, double **FtF, double *allredvec, double *Ftf, double *svec, MPI_Comm comm) 
{
    int i, j, k, ctr, cnt; 
    double temp_sum = 0; 
    /////////////////////////////////////////////////

    for (j = 0; j < m; j++) {
        for (i = 0; i  <= j; i++) {
            temp_sum = 0; 
            for (k = 0; k < Np; k++) {
                temp_sum = temp_sum + DF[i][k]*DF[j][k];
            }
            allredvec[j*m + i] = temp_sum;                                 // allredvec(i,j) = DF[i]' * DF[j]
            allredvec[i*m + j] = temp_sum;                                 // allredvec(j,i) = DF[j]' * DF[i]
        }
    }

    ctr = m * m; 

    for (j = 0; j < m; j++) {
        temp_sum = 0; 
        for (k = 0; k < Np; k++) 
            temp_sum = temp_sum + DF[j][k]*f[k]; 
        allredvec[ctr++] = temp_sum;                                       // allredvec(i+m*m) = DF[k]' * f
    }

    MPI_Allreduce(MPI_IN_PLACE, allredvec, m*m+m, MPI_DOUBLE, MPI_SUM, comm); 

    ctr = 0; 
    for (j = 0; j < m; j++) 
        for (i = 0; i < m; i++) 
            FtF[i][j] = allredvec[ctr++];
        
    cnt = 0; 
    for (j = 0; j < m; j++) 
          Ftf[cnt++] = allredvec[ctr++]; 

    PseudoInverseTimesVec(FtF, Ftf, svec, m);                               // svec = inv(FtF) * Ftf

    for (k = 0; k < Np; k++) {
        temp_sum = 0; 
        for (j = 0; j < m; j++) {
            temp_sum = temp_sum + (DX[j][k] + beta*DF[j][k])*svec[j]; 
        }
        am_vec[k] = temp_sum;                                               // am_vec[k] = (DX + beta * DF) * svec
    }
}


