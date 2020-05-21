/**
 * @file    tools.h
 * @brief   This file declares functions to get and restore local values 
 *          of a vector
 *
 * @author  Xin Jing <xjing30@gatech.edu>
 *          Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
 * 
 * Copyright (c) 2020 Material Physics & Mechanics Group at Georgia Tech.
 */

#ifndef TOOLS_H
#define TOOLS_H

#include "system.h"

void GetLocalVector(DM da, Vec res, Vec *localv, PetscInt *blockinfo, PetscInt Np, PetscScalar *local, PetscScalar ****r);

void RestoreGlobalVector(DM da, Vec global_v, Vec local_v, PetscInt *blockinfo, PetscScalar *local, PetscScalar ****r);

#endif