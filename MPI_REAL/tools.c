/**
 * @file    tools.c
 * @brief   This file contains related functions for 3 linear solvers
 *
 * @author  Xin Jing  < xjing30@gatech.edu>
 *          Phanish Suryanarayana  < phanish.suryanarayana@ce.gatech.edu>
 * 
 * Copyright (c) 2020 Material Physics & Mechanics Group at Georgia Tech.
 */


#include "tools.h"

/**
 * @brief   vector 2-norm
 */

void Vector2Norm(double* Vec, int len, double* ResVal, MPI_Comm comm) 
{ 
    int k; 
    double res = 0; 
    for (k = 0; k < len; k++)
        res = res + Vec[k]*Vec[k]; 

    MPI_Allreduce(&res, ResVal, 1, MPI_DOUBLE, MPI_SUM, comm); 

    *ResVal = sqrt(*ResVal);
}


/**
 * @brief   Vector Dot Product of Vec1 and Vec2
 */

void VectorDotProduct(double* Vec1, double* Vec2, int len, double* ResVal, MPI_Comm comm) 
{ 
    int k; 
    double res = 0; 
    for (k = 0; k < len; k++)
        res = res + Vec1[k] * Vec2[k]; 

    MPI_Allreduce(&res, ResVal, 1, MPI_DOUBLE, MPI_SUM, comm); 
}


/**
 * @brief   Pseudo inverse x vector
 *
 *          x = pinv(A)*b
 */

void PseudoInverseTimesVec(double **A, double *b, double *x, int m) 
{
    int i, j, k, jj; 
    double **U, **V, *w;  // A matrix in column major format as Ac. w is the array of singular values
    U = (double**) calloc(m, sizeof(double*));   
    V = (double**) calloc(m, sizeof(double*)); 
    for (k = 0; k < m; k++)
    {
      U[k] = (double*) calloc(m, sizeof(double));   
      V[k] = (double*) calloc(m, sizeof(double));   
    }
    w = (double*) calloc(m, sizeof(double));   

    for (j = 0; j < m; j++) { 
        for (i = 0; i < m; i++) { 
            U[i][j] = A[i][j]; 
        }
    }

    // Perform SVD on matrix A = UWV'.
    SingularValueDecomp(U, m, m, w, V); 

    // Find Pseudoinverse times vector (pinv(A)*b = (V * diag(1/wj) * U') * b)
    double s, *tmp; 
    tmp = (double*) calloc(m, sizeof(double));  
    // diag(1/wj) * U'*b
    for (j = 0; j < m; j++) { 
        s = 0.0; 
        if (w[j]) { 
            for (i = 0; i < m; i++) 
                s +=  U[i][j]*b[i]; 
            s /=  w[j];  
        }
        tmp[j] = s; 
    }

    // Matrix multiply by V to get answer
    for (j = 0; j < m; j++) { 
        s = 0.0; 
        for (jj = 0; jj < m; jj++) 
            s +=  V[j][jj]*tmp[jj]; 
            x[j] = s; 
    }

    // de-allocate memory
    for (k = 0; k < m; k++) {
        free(U[k]); 
        free(V[k]); 
    }
    free(U); 
    free(V); 
    free(w); 
    free(tmp); 
}

/**
 * @brief   Singular Value Decomposition
 *
 *          A = UWV', a is updated by u
 */

void SingularValueDecomp(double **a, int m, int n, double *w, double **v) 
{ 
    int flag, i, its, j, jj, k, l, nm, Max_its = 250; 
    double anorm, c, f, g, h, s, scale, x, y, z, *rv1; 

    rv1 = (double*) calloc(n, sizeof(double));   
    g = scale = anorm = 0.0; 
    // Householder reduction to bidiagonal form
    for (i = 0; i < n; i++) {
        l = i+1; 
        rv1[i] = scale*g; 
        g = s = scale = 0.0; 
        if (i < m) {
            for (k = i; k < m; k++) 
                scale +=  fabs(a[k][i]); 
            if (scale) {
                for (k = i; k < m; k++) {
                    a[k][i] /=  scale; 
                    s +=  a[k][i]*a[k][i]; 
                }
                f = a[i][i]; 
                g = -SIGN(sqrt(s), f); 
                h = f*g-s; 
                a[i][i] = f-g; 
                for (j = 1;  j < n;  j++){
                    for (s = 0.0, k = i; k < m; k++) 
                        s +=  a[k][i]*a[k][j]; 
                    f = s/h; 
                    for (k = i; k < m; k++) 
                        a[k][j] +=  f*a[k][i]; 
                }
                for (k = i; k < m; k++) 
                    a[k][i] *=  scale; 
            }
        }

        w[i] = scale *g; 
        g = s = scale = 0.0; 
        if (i <= m-1 && i!= n-1) {
            for (k = l; k < n; k++) 
                scale +=  fabs(a[i][k]); 
            if (scale) {
                for (k = l; k < n; k++)
                 {
                    a[i][k] /=  scale; 
                    s +=  a[i][k]*a[i][k]; 
                }
                f = a[i][l]; 
                g = -SIGN(sqrt(s), f); 
                h = f*g-s; 
                a[i][l] = f-g; 
                for (k = l; k < n; k++) 
                    rv1[k] = a[i][k]/h; 
                for (j = l; j < m; j++) {
                    for (s = 0.0, k = l; k < n; k++) 
                        s +=  a[j][k]*a[i][k]; 
                    for (k = l; k < n; k++) 
                        a[j][k] +=  s*rv1[k]; 
                }
                for (k = l; k < n; k++) a[i][k] *=  scale; 
            }
        }

        anorm = max(anorm, (fabs(w[i])+fabs(rv1[i]))); 

    } // end for loop over i

    // Accumulation of right-hand transformations
    for (i = n-1; i>= 0; i--) {
        if (i < n-1) {
            if (g) {
                for (j = l; j < n; j++) // Double division to avoid possible underflow
                    v[j][i] = (a[i][j]/a[i][l])/g; 

                for (j = l; j < n; j++) {
                    for (s = 0.0, k = l; k < n; k++) 
                        s +=  a[i][k]*v[k][j]; 
                    for (k = l; k < n; k++) 
                        v[k][j] +=  s*v[k][i]; 
                }
            }
            for (j = l; j < n; j++) v[i][j] = v[j][i] = 0.0; 
        }
        v[i][i] = 1.0; 
        g = rv1[i]; 
        l = i; 
    } // end for loop over i

    // Accumulation of left-hand transformations
    for (i = min(m, n)-1; i>= 0; i--) {
        l = i+1; 
        g = w[i]; 
        for (j = l; j < n; j++) 
            a[i][j] = 0.0; 
        if (g) {
            g = 1.0/g; 
            for (j = l; j < n; j++) {
                for (s = 0.0, k = l; k < m; k++) 
                    s +=  a[k][i]*a[k][j]; 

                f = (s/a[i][i])*g; 
                for (k = i; k < m; k++) 
                    a[k][j] +=  f*a[k][i]; 
            }
        for (j = i; j < m; j++) 
            a[j][i] *=  g; 
        } else 
            for (j = i; j < m; j++) 
                a[j][i] = 0.0; 
        ++a[i][i]; 
    } // end for over i

    // Diagonalization of the bidiagonal form: Loop over singular values, and over allowed iterations
    for (k = n-1; k>= 0; k--) {
        for (its = 0; its <= Max_its; its++) {
            flag = 1; 
            for (l = k; l>= 0; l--) { // Test for splitting
                nm = l-1;  // Note that rv1[0] is always zero
                if ((double)(fabs(rv1[l])+anorm)   ==   anorm) {
                    flag = 0; 
                    break; 
                }
                if ((double)(fabs(w[nm])+anorm)   ==   anorm) 
                    break; 
            } // end for over l
            if (flag) {
                c = 0.0;  // Cancellation of rv1[1], if l>1
                s = 1.0; 
                for (i = l; i <= k; i++) {
                    f = s*rv1[i]; 
                    rv1[i] = c*rv1[i]; 
                    if ((double)(fabs(f)+anorm)  ==  anorm) 
                        break; 

                    g = w[i]; 
                    h = pythag(f, g); 
                    w[i] = h; 
                    h = 1.0/h; 
                    c = g*h; 
                    s = -f*h; 
                    for (j = 0; j < m; j++)
                     {
                        y = a[j][nm]; 
                        z = a[j][i]; 
                        a[j][nm] = y*c+z*s; 
                        a[j][i] = z*c-y*s; 
                    }
                }
            }
            z = w[k]; 
            if (l  ==  k) { // Convergence
                if (z < 0.0) { // Singular value is made nonnegative
                    w[k] = -z; 
                    for (j = 0; j < n; j++) v[j][k] =  -v[j][k]; 
                }
                break; 
            }
            if (its  ==  Max_its){ printf("no convergence in %d svd iterations \n", Max_its); exit(1); }

            x = w[l];  // Shift from bottom 2-by-2 minor
            nm = k-1; 
            y = w[nm]; 
            g = rv1[nm]; 
            h = rv1[k]; 
            f = ((y-z)*(y+z)+(g-h)*(g+h))/(2.0*h*y); 
            g = pythag(f, 1.0);  
            f = ((x-z)*(x+z)+h*((y/(f+SIGN(g, f)))-h))/x; 
            c = s = 1.0;  // Next QR transformation
            for (j = l; j <= nm; j++) {
                i = j+1; 
                g = rv1[i]; 
                y = w[i]; 
                h = s*g; 
                g = c*g; 
                z = pythag(f, h); 
                rv1[j] = z; 
                c = f/z; 
                s = h/z; 
                f = x*c+g*s; 
                g = g*c-x*s; 
                h = y*s; 
                y *=  c; 
                for (jj = 0; jj < n; jj++) {
                    x = v[jj][j]; 
                    z = v[jj][i]; 
                    v[jj][j] = x*c+z*s; 
                    v[jj][i] = z*c-x*s; 
                }
                z = pythag(f, h); 
                w[j] = z;  // Rotation can be arbitrary if z = 0
                if (z) {
                    z = 1.0/z; 
                    c = f*z; 
                    s = h*z; 
                }
                f = c*g+s*y; 
                x = c*y-s*g; 
                for (jj = 0; jj < m; jj++) {
                    y = a[jj][j]; 
                    z = a[jj][i]; 
                    a[jj][j] = y*c+z*s; 
                    a[jj][i] = z*c-y*s; 
                }
            }
            rv1[l] = 0.0; 
            rv1[k] = f; 
            w[k] = x; 
        } // end for over its
    } //end for over k

    free(rv1); 

    // on output a should be u. But for square matrix u and v are the same. so re-assign a as v.
    for (j = 0; j < m; j++) {
        for (i = 0; i < m; i++)
            a[i][j] = v[i][j]; 
    }
    
    // zero out small singular values
    double wmin, wmax = 0.0;  
    for (j = 0; j < n; j++) if (w[j] > wmax) wmax = w[j]; 
        wmin = n*wmax*(2.22044605e-16);  
    for (j = 0; j < n; j++) if (w[j] < wmin) w[j] = 0.0; 

}

/**
 * @brief   Pythagorean
 *
 *          (a^2 + b^2)^0.5
 */

double pythag(double a, double b) 
{
    double absa, absb; 
    absa = fabs(a); 
    absb = fabs(b); 
    if (absa > absb) return absa*sqrt(1.0+(double)(absb*absb/(absa*absa))); 
    else return (absb   ==   0.0 ? 0.0 : absb*sqrt(1.0+(double)(absa*absa/(absb*absb)))); 
}
