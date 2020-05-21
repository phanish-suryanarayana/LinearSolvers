/**
 * @file    tools.c
 * @brief   This file contains functions to get and restore local values 
 *          of a vector
 *
 * @author  Xin Jing <xjing30@gatech.edu>
 *          Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
 * 
 * Copyright (c) 2020 Material Physics & Mechanics Group at Georgia Tech.
 */

#include "tools.h"

void GetLocalVector(DM da, Vec res, Vec *localv, PetscInt *blockinfo, PetscInt Np, PetscScalar *local, PetscScalar ****r)
{
    int i, j, k, t, *ix;
    PetscInt xcor = blockinfo[0], ycor = blockinfo[1], zcor = blockinfo[2], 
              lxdim = blockinfo[3], lydim = blockinfo[4], lzdim = blockinfo[5];
    /////////////////////////////////////////////////

    DMDAVecGetArray(da, res, r);                                    // r is the ghosted local vectors of res

    // Extract local values
    t = 0;
    for (k = zcor; k < lzdim+zcor; k++)
        for (j = ycor; j < lydim+ycor; j++)
            for (i = xcor; i < lxdim+xcor; i++)
                local[t++] = (*r)[k][j][i];
            
    ix = (int *) calloc (Np, sizeof(int));
    for (i = 0; i < Np; i++)
        ix[i] = i;
    VecSetValues(*localv, Np, ix, local, INSERT_VALUES);
    VecAssemblyBegin(*localv);
    VecAssemblyEnd(*localv);
    free(ix);
}

void RestoreGlobalVector(DM da, Vec global_v, Vec local_v, PetscInt *blockinfo, PetscScalar *local, PetscScalar ****r)
{
    int t, k, j, i;
    PetscInt xcor = blockinfo[0], ycor = blockinfo[1], zcor = blockinfo[2], 
              lxdim = blockinfo[3], lydim = blockinfo[4], lzdim = blockinfo[5];
    /////////////////////////////////////////////////

    VecGetArray(local_v, &local);                                      // Get preconditioned residual

    // update local rvalues
    t = 0;
    for (k = zcor; k < lzdim+zcor; k++)
        for (j = ycor; j < lydim+ycor; j++)
            for (i = xcor; i < lxdim+xcor; i++)
                (*r)[k][j][i] = local[t++];
    DMDAVecRestoreArray(da, global_v, r);
}