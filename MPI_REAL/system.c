/**
 * @file    system.c
 * @brief   This file contains functions for linear system, residual 
 *          function and precondition function.
 *
 * @author  Xin Jing  < xjing30@gatech.edu>
 *          Phanish Suryanarayana  < phanish.suryanarayana@ce.gatech.edu>
 * 
 * Copyright (c) 2020 Material Physics & Mechanics Group at Georgia Tech.
 */


#include "system.h"
#include "tools.h"

/**
 * @brief   CheckInputs
 *
 *          Read and set parameters from command line
 */

void CheckInputs(POISSON *system, int argc, char ** argv)  
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); 
    MPI_Comm_size(MPI_COMM_WORLD, &size); 

    if (!rank) {
        printf("*************************************************************************** \n"); 
        printf("                          AAR linear solver\n"); 
        printf("*************************************************************************** \n"); 
        char* c_time_str; 
        time_t current_time = time(NULL); 
        c_time_str = ctime(&current_time);   
        printf("Starting time: %s \n", c_time_str); 
    }

    if (argc < 6) {
        if (!rank) printf("Wrong inputs"); 
        exit(-1); 
    } else {
        system->solver_tol = atof(argv[1]); 
        system->m = atof(argv[2]); 
        system->p = atof(argv[3]); 
        system->omega = atof(argv[4]); 
        system->beta = atof(argv[5]); 
    }

    if(!rank){
        printf( "***************************************************************************\n"); 
        printf( "                           INPUT PARAMETERS                                \n"); 
        printf( "***************************************************************************\n"); 
        printf( "FD Order    : %d\n", system->FDn * 2);
        printf( "Solver      : AAR\n"); 
        printf( "solver_tol  : %e \n", system->solver_tol); 
        printf( "m           : %d\n", system->m); 
        printf( "p           : %d\n", system->p); 
        printf( "omega       : %lf\n", system->omega); 
        printf( "beta        : %lf\n", system->beta); 
        printf( "Number of processors : %d\n", size);
        printf( "***************************************************************************\n"); 
    }
}


/**
 * @brief   Lap_Vec_mult
 *
 *          Laplacian Matrix multiply a Vector. a * Lap * vector
 *
 * @system          : POISSON structure
 * @a               : constant a
 * @phi             : vector 
 * @Lap_phi         : result of a * Lap * vector
 * @comm_laplacian  : graph communication topology
 *
 */

void Lap_Vec_mult(POISSON *system, double a, double *phi, double *Lap_phi, MPI_Comm comm_laplacian)
{
#define phi(i, j, k) phi[(i) + (j) * psize[0] + (k) * psize[0] * psize[1]]
#define Lap_phi(i, j, k) Lap_phi[(i) + (j) * psize[0] + (k) * psize[0] * psize[1]]
#define x_ghosted(i, j, k) x_ghosted[(i) + (j) * (psize[0] + 2 * FDn) + (k) * (psize[0] + 2 * FDn) * (psize[1] + 2 * FDn)]

    int i, j, k, dir, layer, count, ghosted_size, order, FDn, sum, size;
    int psize[3] = {system->psize[0], system->psize[1], system->psize[2]};
    double *x_in, *x_out, *x_ghosted, *Lap_weights;
    MPI_Request request;
    //////////////////////////////////////////////////////////////////////

    FDn = system->FDn;

    MPI_Comm_size(MPI_COMM_WORLD, &size); 

    ghosted_size = (psize[0] + 2 * FDn) * (psize[1] + 2 * FDn) * (psize[2] + 2 * FDn);
    x_ghosted   = (double*) calloc(ghosted_size, sizeof(double));                       // ghosted x for assembly
    Lap_weights = (double*) calloc((FDn + 1),    sizeof(double));                       // a * Lap coefficients
    assert(x_ghosted != NULL && Lap_weights != NULL);

    // update coefficient
    for (i = 0; i < FDn + 1; i++){
        Lap_weights[i] = a * system->coeff_lap[i];
    }

    if (size > 1)                                                                       // parallel communication
    {
        x_in  = (double*) calloc(system->n_in,  sizeof(double));                        // number of elements received from each neighbor 
        x_out = (double*) calloc(system->n_out, sizeof(double));                        // number of elements sent to each neighbor
        assert(x_in != NULL && x_out != NULL);

        // assemble x_out
        count = 0;
        order = 0;
        for (dir = 0; dir < 6; dir++){
            for (layer = 0; layer < system->send_layers[dir]; layer++){

                int start[3] = {0, 0 , 0};                                              // range of elements to be sent out
                int end[3]   = {psize[0], psize[1], psize[2]};

                if (dir % 2 == 0){
                    end[dir / 2] = system->send_counts[order++];
                } else {
                    start[dir / 2] = psize[dir / 2] - system->send_counts[order++];
                }

                for (k = start[2]; k < end[2]; k++)
                    for (j = start[1]; j < end[1]; j++)
                        for (i = start[0]; i < end[0]; i++)
                            x_out[count++] = phi(i, j, k);
            }   
        }

        MPI_Ineighbor_alltoallv(x_out, system->scounts, system->sdispls, MPI_DOUBLE, x_in, system->rcounts, system->rdispls, MPI_DOUBLE, comm_laplacian, &request); 
        MPI_Wait(&request, MPI_STATUS_IGNORE);


        // assemble x_in        
        count = 0;
        order = 0;
        for (dir = 0; dir < 6; dir ++){                                                 // received from right, left, front, back, up, down neighbors
            sum = 0;
            for (layer = 0; layer < system->rec_layers[dir]; layer ++){

                int start[3] = {FDn, FDn, FDn};                                         // range to be placed in x_ghosted
                int end[3]   = {FDn + psize[0], FDn + psize[1], FDn + psize[2]};

                sum += system->rec_counts[order];

                if (dir % 2 == 1){
                    start[dir / 2] = FDn - sum;
                    end[dir / 2]   = FDn - (sum - system->rec_counts[order++]); 
                } else {
                    start[dir / 2] = FDn + psize[dir / 2] + (sum - system->rec_counts[order++]) ;
                    end[dir / 2]   = FDn + psize[dir / 2] + sum; 
                }

                for (k = start[2]; k < end[2]; k++)
                    for (j = start[1]; j < end[1]; j++)
                        for (i = start[0]; i < end[0]; i++)
                            x_ghosted(i, j, k) = x_in[count++];
            }
        }

        for (k = 0; k < psize[2]; k++)
            for (j = 0; j < psize[1]; j++)
                for (i = 0; i < psize[0]; i++)
                    x_ghosted(i + FDn, j + FDn, k + FDn) = phi(i, j, k);                // assemble original nodes


        // update phi
        for (k = FDn; k < FDn + psize[2]; k++)
            for (j = FDn; j < FDn + psize[1]; j++)
                for (i = FDn; i < FDn + psize[0]; i++){

                    Lap_phi(i - FDn, j - FDn, k - FDn) = x_ghosted(i, j, k) * 3 * Lap_weights[0];

                    for (order = 1; order < FDn + 1; order ++){

                        Lap_phi(i - FDn, j - FDn, k - FDn) += (x_ghosted(i - order, j, k) + x_ghosted(i + order, j, k) 
                                                             + x_ghosted(i, j - order, k) + x_ghosted(i, j + order, k) 
                                                             + x_ghosted(i, j, k - order) + x_ghosted(i, j, k + order)) * Lap_weights[order];
                    }
                }

        free(x_in);
        free(x_out);

    } else {
        // Sequential algorithm 
        for (k = 0; k < psize[2]; k++)
            for (j = 0; j < psize[1]; j++)
                for (i = 0; i < psize[0]; i++){

                    Lap_phi(i, j, k) = phi(i, j, k) * 3 * Lap_weights[0];

                    for (order = 1; order < FDn + 1; order ++){                         // order * psize is used to make sure the index is nonnegative

                        Lap_phi(i, j, k) += (phi((i - order + order * psize[0]) % psize[0], j, k) + phi((i + order) % psize[0], j, k) 
                                           + phi(i, (j - order + order * psize[1]) % psize[1], k) + phi(i, (j + order) % psize[1], k) 
                                           + phi(i, j, (k - order + order * psize[2]) % psize[2]) + phi(i, j, (k + order) % psize[2])) * Lap_weights[order];
                    }
                }

    }

#undef phi
#undef x_ghosted
#undef Lap_phi

    free(x_ghosted);
    free(Lap_weights);
}


/**
 * @brief   Precondition
 *
 *          Precondition function. This is Jacobi preconditioner
 */

void Precondition(double diag, double *res, int Np)
{
    int i;
    for (i = 0; i < Np; i ++)
        res[i] /= diag;
}


/**
 * @brief   Comm_topologies
 *
 *          Create base Cartesian topology and efficient distributed graph 
 *          topology for parallel communication 
 */


void Comm_topologies(POISSON *system, int FDn, int psize[3], int coords[3], int rem[3], int np[3], MPI_Comm cart, MPI_Comm *comm_laplacian,
    int *send_neighs, int *rec_neighs, int *send_counts, int *rec_counts, int send_layers[6], int rec_layers[6], int *n_in, int *n_out)
{
#define send_layers(i,j) send_layers[(i)*2+(j)]
#define rec_layers(i,j)   rec_layers[(i)*2+(j)]

    int i, j, k, sum, sign, neigh_size, large, small, reorder = 0;
    int neigh_coords[3], scale[3], sources = 0, destinations = 0;
    //////////////////////////////////////////////////////////////////////

    // First sending elements to others. Left, right, back, front, down, up 
    k = 0;
    for (i = 0; i < 3; i++){                                                    // loop over x, y, z axis
        sign = -1;                                                              // sign indicates direction. e.g. left is -1. right is +1 
        Find_size_dir(rem[i], coords[i], psize[i], &small, &large);

        for (j = 0; j < 2; j++){                                                // loop over 2 directions
            Vec_copy(neigh_coords, coords, 3);
            sum  = FDn;
            send_layers(i, j) = 0;
            neigh_size = 0;

            while (sum > 0){
                neigh_coords[i] = (neigh_coords[i] + sign + np[i]) % np[i];     // shift coordinates 
                
                send_layers(i,j)++;
                
                MPI_Cart_rank(cart, neigh_coords, send_neighs + k);             // find neighbor's rank
                
                *(send_counts + k) = Min(sum, psize[i]);

                if (neigh_coords[i] < rem[i])                                   // find neighbor's size in this direction
                    neigh_size = large;
                else
                    neigh_size = small;

                sum -= neigh_size;
                k++;
            }
            sign *= (-1);
        }
    }


    // Then receiving elements from others. Right, left, front, back, up, down. First sent, first received. 
    k = 0;
    for (i = 0; i < 3; i++){
        sign = +1;
        Find_size_dir(rem[i], coords[i], psize[i], &small, &large);

        for (j = 0; j < 2; j++){
            Vec_copy(neigh_coords, coords, 3);
            sum  = FDn;
            rec_layers(i, j) = 0;

            while (sum > 0){
                neigh_coords[i] = (neigh_coords[i] + sign + np[i]) % np[i];

                if (neigh_coords[i] < rem[i])
                    neigh_size = large;
                else
                    neigh_size = small;

                MPI_Cart_rank(cart, neigh_coords, rec_neighs + k);
                
                rec_layers(i,j)++;

                sum -= neigh_size;
                
                *(rec_counts + k) = neigh_size;
                k++;
            }
            *(rec_counts + k - 1) = neigh_size + sum;

            sign *= (-1);
        }
    }

#undef send_layers
#undef rec_layers

    for (i = 0; i < 6; i++){
        sources += rec_layers[i];                                               // number of total sources in topology
        destinations += send_layers[i];                                         // number of total destinations in topology
    }

    MPI_Dist_graph_create_adjacent(MPI_COMM_WORLD, sources, rec_neighs, (int *)MPI_UNWEIGHTED, 
        destinations, send_neighs, (int *)MPI_UNWEIGHTED, MPI_INFO_NULL, reorder, comm_laplacian); 

    system->scounts = (int*) calloc(destinations, sizeof(int));                 // send counts of elements
    system->rcounts = (int*) calloc(sources,      sizeof(int));                 // receive counts of elements
    system->sdispls = (int*) calloc(destinations, sizeof(int));                 // send displacements
    system->rdispls = (int*) calloc(sources,      sizeof(int));                 // receive displacements

    scale[0] = psize[1] * psize[2];
    scale[1] = psize[0] * psize[2];
    scale[2] = psize[0] * psize[1];

    *n_in  = 0; 
    *n_out = 0;

    k = 0;
    for (i = 0; i < 6; i++)                                                     // loop over 6 directions (left, right, back, front, down, up)
        for (j = 0; j < send_layers[i]; j++){
            system->scounts[k] = send_counts[k] * scale[i / 2];                        // scaled by size of other 2 axis 
            *n_out += system->scounts[k];

            if (k > 0)
                system->sdispls[k] = system->sdispls[k - 1] + system->scounts[k - 1];
            k++;
        }

    k = 0;
    for (i = 0; i < 6; i++)
        for (j = 0; j < rec_layers[i]; j++){
            system->rcounts[k] = rec_counts[k] * scale[i / 2];
            *n_in += system->rcounts[k];
            
            if (k > 0)
                system->rdispls[k] = system->rdispls[k - 1] + system->rcounts[k - 1];
            k++;
        }
}

/**
 * @brief   Processor_domain
 *
 *          Divide each domain based on the number of processes in each direction
 */

void Processor_Domain(int ssize[3], int psize[3], int np[3], int coords[3], int rem[3], MPI_Comm comm, MPI_Comm *cart)
{
    int i, rank, size, reorder = 0;
    int period[3] = {1, 1, 1};
    /////////////////////////////////////////////////////

    MPI_Comm_size(comm, &size); 
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); 

    np[0] = 0; np[1] = 0; np[2] = 0;
    MPI_Dims_create(size, 3, np);

    MPI_Cart_create(comm, 3, np, period, reorder, cart);

    MPI_Cart_coords(*cart, rank, 3, coords);

    for (i = 0; i < 3; i++) {
        if (ssize[i] < np[i]){
            if(!rank) printf("Error: System size in %d direction is too small.\n", i);
            MPI_Barrier(MPI_COMM_WORLD); 
            exit(-1);
        }

        rem[i] = ssize[i] % np[i];
        if (coords[i] < rem[i]){
            psize[i] = ssize[i] / np[i] + 1;
        }
        else {
            psize[i] = ssize[i] / np[i];
        }
    }
}

/**
 * @brief   Max_layer
 *
 *          Estimate the upper bound of layers for receiving and send elements
 */

void Max_layer(int ssize[3], int np[3], int FDn, int *max_layer)
{
    int i;
    /////////////////////////////////////////////////////

    for (i = 0; i < 3; i++){
        *max_layer += 2 * (FDn / (ssize[i] / np[i]) + 1);
    }
}


/**
 * @brief   Initialize
 *
 *          Allocate memory space for variables and generate the initial guess
 */


void Initialize(POISSON *system, int max_layer)
{
#define rhs(i, j, k) rhs[i + j * system->psize[0] + k * system->psize[0] * system->psize[1]]

    int i, j, k, g_origin[3], g_ind, no_nodes;
    double rhs_sum = 0, rhs_sum_global = 0;
    ////////////////////////////////////////////////////////////

    system->send_neighs = (int*) calloc(max_layer, sizeof(int));
    system->rec_neighs  = (int*) calloc(max_layer, sizeof(int));
    system->send_counts = (int*) calloc(max_layer, sizeof(int));
    system->rec_counts  = (int*) calloc(max_layer, sizeof(int));
    assert(system->send_neighs != NULL && system->rec_neighs != NULL && system->send_counts != NULL && system->rec_counts != NULL);

    system->coeff_lap = (double*) calloc((system->FDn + 1), sizeof(double)); 
    assert(system->coeff_lap != NULL);

    Lap_coefficient(system->coeff_lap, system->FDn);

    no_nodes = system->psize[0] * system->psize[1] * system->psize[2];
    system->phi = (double*) calloc(no_nodes, sizeof(double)); 
    system->rhs = (double*) calloc(no_nodes, sizeof(double)); 
    system->Lap_phi = (double*) calloc(no_nodes, sizeof(double)); 
    assert(system->phi != NULL && system->rhs != NULL && system->Lap_phi != NULL);

    Get_block_origin_global_coords(system->coords, system->rem, system->psize, g_origin);

    // generate right hand side by setting random seed = global index + 1 (> 0) for debug.
    for (k = 0; k < system->psize[2]; k ++)    
        for (j = 0; j < system->psize[1]; j ++)    
            for (i = 0; i < system->psize[0]; i ++) {
                g_ind = (g_origin[0] + i) + (g_origin[1] + j) * system->ssize[0] + (g_origin[2] + k) * system->ssize[0] * system->ssize[1];
                srand(g_ind + 1);
                system->rhs(i, j, k) = 2 * (double)(rand()) / (double)(RAND_MAX); 
                rhs_sum += system->rhs(i, j, k);
            }

    MPI_Allreduce(&rhs_sum, &rhs_sum_global, 1, MPI_DOUBLE, MPI_SUM, system->cart);
    rhs_sum_global /= (system->ssize[0] * system->ssize[1] * system->ssize[2]);

    for (i = 0; i < no_nodes; i++){
        system->rhs[i] -= rhs_sum_global;
        system->phi[i] = 1.0;
    }

#undef rhs
}


/**
 * @brief   Lap_coefficient
 *
 *          Calculate the finite difference coefficients for 1 dimensional Laplacian Matrix with order FDn
 */

void Lap_coefficient(double *coeff_lap, int FDn)
{
    int i, p;
    double  Nr, Dr, val;
    ////////////////////////////////////////////////////

    for (p = 1;  p <= FDn; p++)
        coeff_lap[0] +=  -(2.0 / (p * p)); 

    for (p = 1; p <= FDn; p++) {
        Nr = 1; Dr = 1; 
        for (i = FDn - p + 1; i <= FDn; i++)
            Nr *= i; 
        for (i = FDn + 1; i <= FDn + p; i++)
            Dr *= i; 
        val = Nr / Dr; 
        coeff_lap[p] = (2 * pow(-1, p + 1) * val / (p * p));  
    }
}


/**
 * @brief   Vec_copy
 *
 *          Copy the first n elements of vector b to vector a
 */

void Vec_copy(int *a, int *b, int n)
{
    int i;
    
    for (i = 0; i < n; i++)
        a[i] = b[i];
}


/**
 * @brief   Deallocate_memory
 *
 *          Free all previously allocated memory space
 */

void Deallocate_memory(POISSON *system)
{
    free(system->send_neighs);
    free(system->rec_neighs);
    free(system->send_counts);
    free(system->rec_counts);

    free(system->scounts);
    free(system->sdispls);
    free(system->rcounts);
    free(system->rdispls);

    free(system->coeff_lap);
    free(system->phi);
    free(system->rhs);
    free(system->Lap_phi);

    MPI_Comm_free(&system->comm_laplacian);
    MPI_Comm_free(&system->cart);

    free(system);
}


/**
 * @brief   Get_block_origin_global_coords
 *
 *          Get global coordinates of origin in each block
 */

void Get_block_origin_global_coords(int coords[3], int rem[3], int psize[3], int g_origin[3])
{
    int i, j, small, large;
    ///////////////////////////////////////////////////////////////

    for (i = 0; i < 3; i++){
        Find_size_dir(rem[i], coords[i], psize[i], &small, &large);
        g_origin[i] = 0;
        j = 0;
        while (j < coords[i]){
            if (j < rem[i])
                g_origin[i] += large;
            else
                g_origin[i] += small;
            j++;
        }
    }
}


/**
 * @brief   Find_size_dir
 *
 *          Find the possible sizes (small and large) in one direction
 */

void Find_size_dir(int rem, int coords, int psize, int *small, int *large)
{   
    if (coords < rem){
        *large = psize;
        *small = psize - 1;
    } else {
        *large = psize + 1;
        *small = psize;
    }
}
