/*=============================================================================================
| Alternating Anderson Richardson (AAR) code and tests in real-valued systems.
| Copyright (C) 2020 Material Physics & Mechanics Group at Georgia Tech.
| 
| Authors: Xin Jing, Phanish Suryanarayana
|
| Last Modified: 6 April 2020
|-------------------------------------------------------------------------------------------*/


#include "system.h"
#include "AAR.h"
#include "tools.h"

int main(int argc, char ** argv) 
{
    MPI_Init(&argc, &argv); 
    int rank, size, i, j, max_layer = 0, no_nodes, no_tests;
    double t_begin, t_end, t0, t1; 
    FILE *fp;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank); 
    MPI_Comm_size(MPI_COMM_WORLD, &size); 

    POISSON *system = malloc(sizeof(POISSON));
    assert(system != NULL);

    system->ssize[0] = 48;
    system->ssize[1] = 48;
    system->ssize[2] = 48;
    system->FDn = 6;

    system->solver_maxiter = 1e5;

    CheckInputs(system, argc, argv); 

    t_begin = MPI_Wtime(); 

    // Compute range of each process
    Processor_Domain(system->ssize, system->psize, system->np, system->coords, system->rem, MPI_COMM_WORLD, &system->cart); 
    
    // Estimate maximum layers 
    Max_layer(system->ssize, system->np, system->FDn, &max_layer);

    // Initialize necessary variables
    Initialize(system, max_layer);

    // Create distributed graph topology for parallel communication
    Comm_topologies(system, system->FDn, system->psize, system->coords, system->rem, system->np, system->cart, &system->comm_laplacian,
        system->send_neighs, system->rec_neighs, system->send_counts, system->rec_counts, system->send_layers, system->rec_layers, &system->n_in, &system->n_out);

    // start tests
    no_nodes = system->psize[0] * system->psize[1] * system->psize[2];

    for (i = 0; i < no_nodes; i ++)
        system->phi[i] = 1;

    if (!rank) printf("AAR starts.\n"); 
    t0 = MPI_Wtime();
    AAR(system, Lap_Vec_mult, Precondition, -1.0/(4*M_PI), system->phi, system->rhs, system->omega, system->beta, system->m, system->p, system->solver_maxiter, system->solver_tol, no_nodes, system->comm_laplacian);  
    t1 = MPI_Wtime();
    if (!rank) printf("\n*************************************************************************** \n\n"); 
    
    // Deallocate_memory(system);
    
    t_end = MPI_Wtime(); 
    if (!rank) {
        printf("Total wall time   = %.4f seconds. \n", t_end-t_begin); 
        char* c_time_str; 
        time_t current_time = time(NULL); 
        c_time_str = ctime(&current_time);   
        printf("Ending time: %s", c_time_str); 
        printf("*************************************************************************** \n \n"); 
    }
    MPI_Barrier(MPI_COMM_WORLD);    

    MPI_Finalize(); 
    return 0; 
}
