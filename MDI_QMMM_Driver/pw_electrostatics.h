#ifndef PW_ELECTROSTATICS
#define PW_ELECTROSTATICS

#include <mpi.h>
extern "C" {
#include "mdi.h"
}

int recenter( int natoms, MPI_Comm world_comm, int qm_start, int qm_end, double* cell, double* coords );

int pw_electrostatic_potential( int natoms, int ntypes, int* types, double* masses, int ngrid, double* grid, double* density, double* coords, double* charges, MPI_Comm world_comm, int qm_start, int qm_end, double* qm_charges, double* forces_mm, MDI_Comm comm );

int pw_electrostatic_forces( int natoms, int ntypes, int* types, double* masses, int ngrid, double* grid, double* density, double* coords, double* charges, MPI_Comm world_comm, int qm_start, int qm_end, double* qm_charges, double* forces_mm );

#endif
