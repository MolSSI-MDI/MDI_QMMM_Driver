#include <iostream>
#include <math.h>
#include "pw_electrostatics.h"

using namespace std;

#define NELEMENTS 118
#define NISOTOPES 2

static const int isotope_elements[NISOTOPES] = {
  1, 
  8
};

static const double isotope_masses[NISOTOPES] = {
  1.00782503224, 
  15.99491461960
};


static const double covalent_radii[NELEMENTS+1] = {
  -1,       // N/A
  0.718096, // H
  0.604712, // He
  2.53223,  // Li 
  1.70075,  // Be
  1.54958,  // B
  1.45509,  // C
  1.41729,  // N
  1.37950,  // O
  1.34171,  // F
  1.30391,  // Ne
  2.91018,  // Na
  2.45664,  // Mg
  2.22988,  // Al
  2.09760,  // Si
  2.00311,  // P
  1.92752,  // S
  1.87083,  // Cl
  1.83303,  // Ar
  3.70386,  // K
  3.28812,  // Ca
  2.72121,  // Sc
  2.57003, // Ti
  2.36216, // V
  2.39995, // Cr
  2.62672, // Mn
  2.36216, // Fe
  2.38105, // Co
  2.28657, // Ni
  2.60782, // Cu
  2.47554, // Zn
  2.38105, // Ca
  2.30547, // Ge
  2.24877, // As
  2.19208, // Se
  2.15429, // Br
  2.07870, // Kr
  3.98732, // Rb
  3.62827, // Sr
  3.06136, // Y
  2.79679, // Zr
  2.58892, // Nb
  2.74010, // Mo
  2.94797, // Tc
  2.38105, // Ru
  2.55113, // Rh
  2.47554, // Pd
  2.89128, // Ag
  2.79679, // Cd
  2.72121, // In
  2.66451, // Sn
  2.60782, // Sb
  2.55113, // Te
  2.51334, // I
  2.45664, // Xe
  4.25188, // Cs
  3.74166, // Ba
  3.19364, // La
  -1,   // Ce
  -1,   // Pr
  -1,   // Nd
  -1,   // Pm
  -1,   // Sm
  -1,   // Eu
  -1,   // Gd
  -1,   // Tb
  -1,   // Dy
  -1,   // Ho
  -1,   // Er
  -1,   // Tm
  -1,   // Yb
  3.02356, // Lu
  2.83459, // Hf
  2.60782, // Ta
  2.75900, // W
  3.00466, // Re
  2.41885, // Os
  2.58892, // Ir
  2.41885, // Pt
  2.72121, // Au
  2.81569, // Hg
  2.79679, // Tl
  2.77790, // Pb
  2.75900, // Bi
  -1,   // Po
  -1,   // At
  2.74010, // Rn
  -1,   // Fr
  -1,   // Ra
  -1,   // Ac
  -1,   // Th
  -1,   // Pa
  -1,   // U
  -1,   // Np
  -1,   // Pu
  -1,   // Am
  -1,   // Cm
  -1,   // Bk
  -1,   // Cf
  -1,   // Es
  -1,   // Fm
  -1,   // Md
  -1,   // No
  -1,   // Lr
  -1,   // Rf
  -1,   // Db
  -1,   // Sg
  -1,   // Bh
  -1,   // Hs
  -1,   // Mt
  -1,   // Ds
  -1,   // Rg
  -1,   // Cn
  -1,   // Nh
  -1,   // Fl
  -1,   // Mc
  -1,   // Lv
  -1,   // Ts
  -1    // Og
};


int get_element_from_mass(double mass) {
  double diff;
  double closest = 100000.0;
  int element = -1;

  for (int i=0; i<NISOTOPES; i++) {
    diff = fabs(mass - isotope_masses[i]);
    if ( diff < closest ) {
      closest = diff;
      element = isotope_elements[i];
    }
  }

  return element;
}


int pw_electrostatic_potential( int natoms, double* masses, int ngrid, double* grid, double* density, double* coords, double* charges, MPI_Comm world_comm, int qm_start, int qm_end, double* qm_charges, double* forces_mm, MDI_Comm comm ) {

  int myrank;
  MPI_Comm_rank(world_comm, &myrank);
  int nranks;
  MPI_Comm_size(world_comm, &nranks);

  if ( myrank == 0 ) {
    MDI_Send_Command(">NPOTENTIAL", comm);
    MDI_Send( &ngrid, 1, MDI_INT, comm );
  }

  // cutoff for nearest-neighbor interactions
  double r_nn = 50000.0;

  // identify the atomic number of each atom
  // because MM engines don't directly track this information, use the atomic mass to determine atomic number
  double* radii = new double[natoms];
  int* elements = new int[natoms];
  for (int i=0; i < natoms; i++) {
    elements[i] = get_element_from_mass( masses[i] );
    radii[i] = covalent_radii[ elements[i] ];
  }

  // calculate the electrostatic potential
  double* potential = new double[ngrid];
  double* potential_sum = new double[ngrid];

  for (int igrid = 0; igrid < ngrid; igrid += 1) {
    potential[igrid] = 0.0;
  }

  for (int igrid = myrank; igrid < ngrid; igrid += nranks) {
    double contribution = 0.0;

    for (int iatom = 0; iatom < natoms; iatom++) {

      double dx = coords[3*iatom + 0] - grid[3*igrid + 0];
      double dy = coords[3*iatom + 1] - grid[3*igrid + 1];
      double dz = coords[3*iatom + 2] - grid[3*igrid + 2];
      double dr = sqrt(dx*dx + dy*dy + dz*dz);

      double dr3 = dr*dr*dr;
      double dr4 = dr3*dr;
      double dr5 = dr4*dr;
      double radii1 = radii[iatom];
      double radii4 = radii1*radii1*radii1*radii1;
      double radii5 = radii4*radii1;

      if ( dr <= r_nn ) {
	contribution -= charges[iatom]*(radii4 - dr4)/(radii5 - dr5);
      }
      
    }

    potential[igrid] += contribution;
  }

  // sum the potential across all ranks
  MPI_Reduce( potential, potential_sum, ngrid, MPI_DOUBLE, MPI_SUM, 0, world_comm);

  if ( myrank == 0 ) {
    MDI_Send_Command(">POTENTIAL", comm);
    MDI_Send( potential_sum, ngrid, MDI_DOUBLE, comm );
  }

  delete [] radii;
  delete [] elements;
  delete [] potential;
  delete [] potential_sum;
  return 0;
}


int pw_electrostatic_forces( int natoms, double* masses, int ngrid, double* grid, double* density, double* coords, double* charges, MPI_Comm world_comm, int qm_start, int qm_end, double* qm_charges, double* forces_mm ) {

  int myrank;
  MPI_Comm_rank(world_comm, &myrank);
  int nranks;
  MPI_Comm_size(world_comm, &nranks);

  // identify the atomic number of each atom
  // because MM engines don't directly track this information, use the atomic mass to deterime atomic number
  double* radii = new double[natoms];
  int* elements = new int[natoms];
  for (int i=0; i < natoms; i++) {
    elements[i] = get_element_from_mass( masses[i] );
    radii[i] = covalent_radii[ elements[i] ];
  }

  double side = grid[3] - grid[0];

  // Compute forces on MM atoms
  double* force_sum = new double[3*natoms];
  for (int icoord = 0; icoord < 3*natoms; icoord++) {
    force_sum[icoord] = 0.0;
  }
  for (int iatom = myrank; iatom < natoms; iatom += nranks) {
    double radii1 = radii[ iatom ];
    double radii4 = radii1*radii1*radii1*radii1;
    double radii5 = radii4*radii1;
    if ( (iatom < qm_start-1) or (iatom > qm_end-1) ) {
      for (int igrid = 0; igrid < ngrid; igrid++) {
	// WARNING: TEMPORARY - SHOULD REPLACE WITH <VDENSITY
	double grid_volume = side*side*side;

	double dx = coords[3*iatom + 0] - grid[3*igrid + 0];
	double dy = coords[3*iatom + 1] - grid[3*igrid + 1];
	double dz = coords[3*iatom + 2] - grid[3*igrid + 2];
	double dr = sqrt(dx*dx + dy*dy + dz*dz);
	double dr3 = dr*dr*dr;
	double dr4 = dr3*dr;
	double dr5 = dr4*dr;
	double fder = ( 5.0*dr4*( radii4 - dr4 ) - 4.0*dr3*( radii5 - dr5 ) ) / ( (radii5 - dr5)*(radii5 - dr5) );
	double prefactor = grid_volume*density[igrid]*fder/dr;

	force_sum[3*iatom + 0] += prefactor*dx;
	force_sum[3*iatom + 1] += prefactor*dy;
	force_sum[3*iatom + 2] += prefactor*dz;
      }

      force_sum[3*iatom + 0] *= charges[iatom];
      force_sum[3*iatom + 1] *= charges[iatom];
      force_sum[3*iatom + 2] *= charges[iatom];
    }
  }



  for (int iatom = myrank; iatom < natoms; iatom += nranks) {
    double radii1 = radii[ iatom ];
    double radii4 = radii1*radii1*radii1*radii1;
    double radii5 = radii4*radii1;
    if ( (iatom < qm_start-1) or (iatom > qm_end-1) ) {
      for (int jatom = qm_start-1; jatom <= qm_end-1; jatom++) {
	double dx = coords[3*iatom + 0] - coords[3*jatom + 0];
	double dy = coords[3*iatom + 1] - coords[3*jatom + 1];
	double dz = coords[3*iatom + 2] - coords[3*jatom + 2];
	double dr = sqrt(dx*dx + dy*dy + dz*dz);
	double dr3 = dr*dr*dr;
	double dr4 = dr3*dr;
	double dr5 = dr4*dr;
	double fder = ( 5.0*dr4*( radii4 - dr4 ) - 4.0*dr3*( radii5 - dr5 ) ) / ( (radii5 - dr5)*(radii5 - dr5) );
	double chargej = qm_charges[jatom-qm_start+1];

	double prefactor = charges[iatom]*chargej*fder/dr;

	// add the forces on the MM atoms
	force_sum[3*iatom + 0] -= prefactor*dx;
	force_sum[3*iatom + 1] -= prefactor*dy;
	force_sum[3*iatom + 2] -= prefactor*dz;

	// add the forces on the QM atoms
	force_sum[3*jatom + 0] += prefactor*dx;
	force_sum[3*jatom + 1] += prefactor*dy;
	force_sum[3*jatom + 2] += prefactor*dz;
      }
    }
  }

  MPI_Reduce( force_sum, forces_mm, 3*natoms, MPI_DOUBLE, MPI_SUM, 0, world_comm);

  delete [] radii;
  delete [] elements;
  delete [] force_sum;

  return 0;
}



int recenter( int natoms, MPI_Comm world_comm, int qm_start, int qm_end, double* cell, double* mm_cell, double* coords ) {

  int myrank;
  MPI_Comm_rank(world_comm, &myrank);
  int nranks;
  MPI_Comm_size(world_comm, &nranks);

  // identify the center of the box
  double box_center[3];
  box_center[0] = 0.5*( cell[0] + cell[3] + cell[6] );
  box_center[1] = 0.5*( cell[1] + cell[4] + cell[7] );
  box_center[2] = 0.5*( cell[2] + cell[5] + cell[8] );

  // identify the average location of the nuclei
  double nuclei_center[3];
  for (int ix = 0; ix < 3; ix++) {
    nuclei_center[ix] = 0.0;
    for (int iatom = qm_start-1; iatom <= qm_end-1; iatom++) {
      nuclei_center[ix] += coords[ 3*iatom + ix ];
    }
    nuclei_center[ix] /= double( qm_end - qm_start + 1 );
  }

  // shift all of the coordinates
  for (int iatom = 0; iatom < natoms; iatom++) {
    for (int ix = 0; ix < 3; ix++) {
      coords[ 3*iatom + ix ] += box_center[ix] - nuclei_center[ix];
    }
  }

  // move the MM coordinates into the nearest image, relative to the center of the QM box
  double mm_size[3];
  mm_size[0] = mm_cell[0] + mm_cell[3] + mm_cell[6];
  mm_size[1] = mm_cell[1] + mm_cell[4] + mm_cell[7];
  mm_size[2] = mm_cell[2] + mm_cell[5] + mm_cell[8];
  for (int iatom = 0; iatom < natoms; iatom++) {
    for (int ix = 0; ix < 3; ix++) {
      double fx;
      fx = coords[ 3*iatom + ix ] - box_center[ix];
      fx /= mm_size[ix];
      fx -= round( fx );
      fx *= mm_size[ix];
      coords[ 3*iatom + ix ] = fx + box_center[ix];
    }
  }

  return 0;
}
