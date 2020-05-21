/**
 * @file    system.h
 * @brief   This file declares the functions for the linear system, residual 
 *          function and precondition function.
 *
 * @author  Xin Jing  < xjing30@gatech.edu>
 *          Phanish Suryanarayana  < phanish.suryanarayana@ce.gatech.edu>
 * 
 * Copyright (c) 2020 Material Physics & Mechanics Group at Georgia Tech.
 */


#ifndef SYSTEM_H
#define SYSTEM_H

#include  <math.h>
#include  <time.h>   // CLOCK
#include  <stdio.h>
#include  <stdlib.h> 
#include  <mpi.h>
#include  <assert.h>

// #define M_PI 3.14159265358979323846
#define Min(a, b) ((a) < (b) ? (a) : (b))

typedef struct {
    int FDn;
    int ssize[3];
    int psize[3];

    int np[3];
    int coords[3];
    int rem[3];

    int *send_neighs;
    int *rec_neighs;
    int *send_counts;
    int *rec_counts;
    int send_layers[6];
    int rec_layers[6];

    int *scounts;
    int *sdispls; 
    int *rcounts;
    int *rdispls;
    int n_in;
    int n_out;

    double *phi;
    double *rhs;
    double *Lap_phi;

    double solver_tol;
    int solver_maxiter;
    int m;
    int p;
    double omega;
    double beta;

    double *coeff_lap;
    MPI_Comm comm_laplacian; 
    MPI_Comm cart;
}POISSON;

void CheckInputs(POISSON *system, int argc, char ** argv);

void Processor_Domain(int ssize[3], int psize[3], int np[3], int coords[3], int rem[3], MPI_Comm comm, MPI_Comm *cart);

void Comm_topologies(POISSON *system, int FDn, int psize[3], int coords[3], int rem[3], int np[3], MPI_Comm cart, MPI_Comm *comm_laplacian,
    int *send_neighs, int *rec_neighs, int *send_counts, int *rec_counts, int send_layers[6], int rec_layers[6], int *n_in, int *n_out);

void Max_layer(int ssize[3], int np[3], int FDn, int *max_layer);

void Initialize(POISSON *system, int max_layer);

void Deallocate_memory(POISSON *system);

void Vec_copy(int *a, int *b, int n);

void Lap_coefficient(double *coeff_lap, int FDn);

void Lap_Vec_mult(POISSON *system, double a, double *phi, double *Lap_phi, MPI_Comm comm_laplacian);

void Precondition(double diag, double *res, int Np);

void Find_size_dir(int rem, int coords, int psize, int *small, int *large);

void Get_block_origin_global_coords(int coords[3], int rem[3], int psize[3], int g_origin[3]);

#endif