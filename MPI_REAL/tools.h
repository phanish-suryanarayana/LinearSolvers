/**
 * @file    AAR_Real.h
 * @brief   This file declares the functions for Alternating Anderson Richardson solver. 
 *
 * @author  Xin Jing  < xjing30@gatech.edu>
 *          Phanish Suryanarayana  < phanish.suryanarayana@ce.gatech.edu>
 * 
 * Copyright (c) 2020 Material Physics & Mechanics Group at Georgia Tech.
 */


#ifndef TOOLS_H
#define TOOLS_H

#include <mpi.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

#define SIGN(a,b) ((b) >= 0.0 ? fabs(a) : -fabs(a))
#define max(a, b) ((a) > (b) ? (a) : (b))
#define min(a, b) ((a) < (b) ? (a) : (b))

double pythag(double a, double b); 
void SingularValueDecomp(double **a,int m,int n, double *w, double **v);
void PseudoInverseTimesVec(double **A,double *b,double *x,int m); 
void Vector2Norm(double* Vec, int len, double* ResVal, MPI_Comm comm); 
void VectorDotProduct(double* Vec1, double* Vec2, int len, double* ResVal, MPI_Comm comm);

#endif