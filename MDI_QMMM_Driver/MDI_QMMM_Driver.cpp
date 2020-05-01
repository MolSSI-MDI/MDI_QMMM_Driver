#include <iostream>
#include <mpi.h>
#include <stdexcept>
#include <string.h>
#include <stdlib.h>
#include <chrono>
#include "pw_electrostatics.h"
extern "C" {
#include "mdi.h"
}

//using namespace std;
typedef std::chrono::time_point<std::chrono::high_resolution_clock> chrono_time;

int main(int argc, char **argv) {
  int ret;

  // Initialize the MPI environment
  MPI_Comm world_comm;
  MPI_Init(&argc, &argv);

  // VARIABLES THAT SHOULD BE READ FROM AN INPUT FILE
  int qmmm_mode = 0;
  int niterations = 0;
  int qm_start = 0;
  int qm_end = 0;

  // Read through all the command line options
  int iarg = 1;
  bool initialized_mdi = false;
  while ( iarg < argc ) {

    if ( strcmp(argv[iarg],"-mdi") == 0 ) {

      // Ensure that the argument to the -mdi option was provided
      if ( argc-iarg < 2 ) {
	throw std::runtime_error("The -mdi argument was not provided.");
      }

      // Initialize the MDI Library
      world_comm = MPI_COMM_WORLD;
      int ret = MDI_Init(argv[iarg+1], &world_comm);
      if ( ret != 0 ) {
	throw std::runtime_error("The MDI library was not initialized correctly.");
      }
      initialized_mdi = true;
      iarg += 2;

    }
    else if ( strcmp(argv[iarg],"-qm_start") == 0 ) {

      // Ensure that the argument to the -qm_start option was provided
      if ( argc-iarg < 2 ) {
	throw std::runtime_error("The -qm_start argument was not provided.");
      }

      qm_start = strtol(argv[iarg+1], nullptr, 0);
      iarg += 2;
    }
    else if ( strcmp(argv[iarg],"-qm_end") == 0 ) {

      // Ensure that the argument to the -qm_end option was provided
      if ( argc-iarg < 2 ) {
	throw std::runtime_error("The -qm_end argument was not provided.");
      }

      qm_end = strtol(argv[iarg+1], nullptr, 0);
      iarg += 2;

    }
    else if ( strcmp(argv[iarg],"-iter") == 0 ) {

      // Ensure that the argument to the -iter option was provided
      if ( argc-iarg < 2 ) {
	throw std::runtime_error("The -iter argument was not provided.");
      }

      niterations = strtol(argv[iarg+1], nullptr, 0);
      iarg += 2;

    }
    else {
      throw std::runtime_error("Unrecognized option.");
    }

  }

  // Confirm that all required command-line arguments have been received
  if ( not initialized_mdi ) {
    throw std::runtime_error("The -mdi command line option was not provided.");
  }
  if ( niterations == 0 ) {
    throw std::runtime_error("The -iter command line option was not provided.");
  }
  if ( qm_start == 0 ) {
    throw std::runtime_error("The -qm_start command line option was not provided.");
  }
  if ( qm_end == 0 ) {
    throw std::runtime_error("The -qm_end command line option was not provided.");
  }
  if ( qm_start < 0 ) {
    throw std::runtime_error("Invalid value for the -qm_start option.");
  }
  if ( qm_end < qm_start ) {
    throw std::runtime_error("Invalid value for the -qm_end option.");
  }
  if ( niterations < 0 ) {
    throw std::runtime_error("Invalid value for the -iter option.");
  }

  // Get the rank of this process
  int myrank;
  MPI_Comm_rank(world_comm, &myrank);

  // Get the current time
  double time_query = 0.0;
  double time_recenter = 0.0;
  double time_scf = 0.0;
  double time_qm_forces = 0.0;
  double time_qm_density = 0.0;
  double time_pot = 0.0;
  double time_forces = 0.0;
  double time_coords = 0.0;
  chrono_time time_start_section;
  chrono_time time_end_section;
  chrono_time time_start;
  if (myrank == 0 ) {
    time_start = std::chrono::high_resolution_clock::now();
  }

  // Connect to the engines
  MDI_Comm mm_comm = MDI_COMM_NULL;
  MDI_Comm mm_sub_comm = MDI_COMM_NULL;
  MDI_Comm qm_comm = MDI_COMM_NULL;
  int nengines = 3;
  for (int iengine=0; iengine < nengines; iengine++) {
    MDI_Comm comm;
    ret = MDI_Accept_Communicator(&comm);
 
    // Determine the name of this engine
    char* engine_name = new char[MDI_NAME_LENGTH];
    if ( myrank == 0 ) {
      MDI_Send_Command("<NAME", comm);
      MDI_Recv(engine_name, MDI_NAME_LENGTH, MDI_CHAR, comm);
    }
    MPI_Bcast( engine_name, MDI_NAME_LENGTH, MPI_CHAR, 0, world_comm );

    if ( myrank == 0 ) {
      std::cout << "Engine name: " << engine_name << std::endl;
    }
 
    if ( strcmp(engine_name, "MM") == 0 ) {
      if ( mm_comm != MDI_COMM_NULL ) {
	throw std::runtime_error("Accepted a communicator from a second main MM engine.");
      }
      mm_comm = comm;
    }
    else if ( strcmp(engine_name, "QM") == 0 ) {
      if ( qm_comm != MDI_COMM_NULL ) {
	throw std::runtime_error("Accepted a communicator from a second QM engine.");
      }
      qm_comm = comm;
    }
    else if ( strcmp(engine_name, "MM_SUB") == 0 ) {
      if ( mm_sub_comm != MDI_COMM_NULL ) {
	throw std::runtime_error("Accepted a communicator from a second subset MM engine.");
      }
      mm_sub_comm = comm;
    }
    else {
      throw std::runtime_error("Unrecognized engine name.");
    }
 
    delete[] engine_name;
  }

  // Simulation variables
  int natoms, natoms_qm;
  double qm_energy;
  double mm_energy;

  // Receive the number of atoms from the main MM engine
  if ( myrank == 0 ) {
    MDI_Send_Command("<NATOMS", mm_comm);
    MDI_Recv(&natoms, 1, MDI_INT, mm_comm);
  }
  MPI_Bcast( &natoms, 1, MPI_INT, 0, world_comm );

  // Receive the number of QM atoms from the subset MM engine
  if ( myrank == 0 ) {
    MDI_Send_Command("<NATOMS", mm_sub_comm);
    MDI_Recv(&natoms_qm, 1, MDI_INT, mm_sub_comm);
  }
  MPI_Bcast( &natoms_qm, 1, MPI_INT, 0, world_comm );

  if ( myrank == 0 ) {
    std::cout << "MM atoms: " << natoms << std::endl;
    std::cout << "QM atoms: " << natoms_qm << std::endl;
  }

  // Receive the number of grid points used to represent the density from the QM engine
  int ngrid;
  if ( myrank == 0 ) {
    MDI_Send_Command("<NDENSITY", qm_comm);
    MDI_Recv(&ngrid, 1, MDI_INT, qm_comm);
    std::cout << "NGRID: " << ngrid << std::endl;
  }
  MPI_Bcast( &ngrid, 1, MPI_INT, 0, world_comm );

  double* grid = new double[3*ngrid];
  double* density = new double[ngrid];
  if ( myrank == 0 ) {
    MDI_Send_Command("<CDENSITY", qm_comm);
    MDI_Recv(grid, 3*ngrid, MDI_DOUBLE, qm_comm);
  }
  MPI_Bcast( grid, 3*ngrid, MPI_DOUBLE, 0, world_comm );

  // Allocate the arrays for the coordinates and forces
  double* mm_coords = new double[3*natoms];
  double* mm_charges = new double[natoms];
  double* masses = new double[natoms];
  double* forces_mm = new double[3*natoms];
  double* qm_charges = new double[natoms];
  double* forces_qm = new double[3*natoms_qm];
  double* forces_ec_mm = new double[3*natoms];
  double* mm_force_on_qm_atoms = new double[3*natoms_qm];
  double* qm_cell = new double[9];
  double* mm_cell = new double[9];
  int* mm_mask = new int[natoms];

  // Set the MM mask
  // this is -1 for non-QM atoms, and 1 for QM atoms
  for (int i=0; i<natoms; i++) {
    mm_mask[i] = -1;
  }
  for (int i=qm_start-1; i<=qm_end-1; i++) {
    mm_mask[i] = 1;
  }

  // Have the MD engine initialize a new MD simulation
  if ( myrank == 0 ) {
    MDI_Send_Command("@INIT_MD", mm_comm);
  }

  // Perform each iteration of the simulation
  for (int iiteration = 0; iiteration < niterations; iiteration++) {

    // Receive the cell dimensions from the MM engine
    if ( myrank == 0 ) {
      time_start_section = std::chrono::high_resolution_clock::now();
    }
    if ( myrank == 0 ) {
      MDI_Send_Command("<CELL", mm_comm);
      MDI_Recv(mm_cell, 9, MDI_DOUBLE, mm_comm);
    }
    MPI_Bcast( mm_cell, 9, MPI_DOUBLE, 0, world_comm );

    // Receive the QM cell dimensions from the QM engine
    if ( myrank == 0 ) {
      MDI_Send_Command("<CELL", qm_comm);
      MDI_Recv(qm_cell, 9, MDI_DOUBLE, qm_comm);
    }
    MPI_Bcast( qm_cell, 9, MPI_DOUBLE, 0, world_comm );

    // Receive the coordinates from the MM engine
    if ( myrank == 0 ) {
      MDI_Send_Command("<COORDS", mm_comm);
      MDI_Recv(mm_coords, 3*natoms, MDI_DOUBLE, mm_comm);
    }
    MPI_Bcast( mm_coords, 3*natoms, MPI_DOUBLE, 0, world_comm );

    // Receive the charges from the MM engine
    if ( myrank == 0 ) {
      MDI_Send_Command("<CHARGES", mm_comm);
      MDI_Recv(mm_charges, natoms, MDI_DOUBLE, mm_comm);
    }
    MPI_Bcast( mm_charges, natoms, MPI_DOUBLE, 0, world_comm );

    // Receive the charges from the QM engine
    if ( myrank == 0 ) {
      MDI_Send_Command("<CHARGES", qm_comm);
      MDI_Recv(qm_charges, natoms_qm, MDI_DOUBLE, qm_comm);
    }
    MPI_Bcast( qm_charges, natoms_qm, MPI_DOUBLE, 0, world_comm );

    // Receive the masses from the MM engine
    if ( myrank == 0 ) {
      MDI_Send_Command("<MASSES", mm_comm);
      MDI_Recv(masses, natoms, MDI_DOUBLE, mm_comm);
    }
    MPI_Bcast( masses, natoms, MPI_DOUBLE, 0, world_comm );
    if ( myrank == 0 ) {
      time_end_section = std::chrono::high_resolution_clock::now();
      double time_sec =std::chrono::duration_cast<std::chrono::milliseconds>(time_end_section-time_start_section).count() / 1000.0;
      time_query += time_sec;
    }

    // Recenter the coordinates
    if ( myrank == 0 ) {
      time_start_section = std::chrono::high_resolution_clock::now();
    }
    if ( myrank == 0 ) {
      recenter(natoms, world_comm, qm_start, qm_end, qm_cell, mm_cell, mm_coords);
    }
    MPI_Bcast( mm_coords, 3*natoms, MPI_DOUBLE, 0, world_comm );

    // Send the QM coordinates to the QM engine
    if ( myrank == 0 ) {
      MDI_Send_Command(">COORDS", qm_comm);
      MDI_Send(&mm_coords[3*(qm_start-1)], 3*natoms_qm, MDI_DOUBLE, qm_comm);
    }
    if ( myrank == 0 ) {
      time_end_section = std::chrono::high_resolution_clock::now();
      double time_sec =std::chrono::duration_cast<std::chrono::milliseconds>(time_end_section-time_start_section).count() / 1000.0;
      time_recenter += time_sec;
    }

    // Calculate the electrostatic potential from the MM atoms on a grid
    if ( myrank == 0 ) {
      time_start_section = std::chrono::high_resolution_clock::now();
    }
    pw_electrostatic_potential(natoms, masses, ngrid, grid, density, mm_coords, mm_charges, world_comm, 
			       qm_start, qm_end, qm_charges, forces_ec_mm, qm_comm);
    if ( myrank == 0 ) {
      time_end_section = std::chrono::high_resolution_clock::now();
      double time_sec =std::chrono::duration_cast<std::chrono::milliseconds>(time_end_section-time_start_section).count() / 1000.0;
      time_pot += time_sec;
    }

    // Get the QM energy
    if ( myrank == 0 ) {
      time_start_section = std::chrono::high_resolution_clock::now();
    }
    if ( myrank == 0 ) {
      MDI_Send_Command("<ENERGY", qm_comm);
      MDI_Recv(&qm_energy, 1, MDI_DOUBLE, qm_comm);
    }
    if ( myrank == 0 ) {
      time_end_section = std::chrono::high_resolution_clock::now();
      double time_sec =std::chrono::duration_cast<std::chrono::milliseconds>(time_end_section-time_start_section).count() / 1000.0;
      time_scf += time_sec;
    }

    // Get the QM forces
    if ( myrank == 0 ) {
      time_start_section = std::chrono::high_resolution_clock::now();
    }
    if ( myrank == 0 ) {
      MDI_Send_Command("<FORCES", qm_comm);
      MDI_Recv(forces_qm, 3*natoms_qm, MDI_DOUBLE, qm_comm);
    }
    if ( myrank == 0 ) {
      time_end_section = std::chrono::high_resolution_clock::now();
      double time_sec =std::chrono::duration_cast<std::chrono::milliseconds>(time_end_section-time_start_section).count() / 1000.0;
      time_qm_forces += time_sec;
    }

    // Have the QM engine send the electronic density on a grid
    if ( myrank == 0 ) {
      time_start_section = std::chrono::high_resolution_clock::now();
    }
    if ( myrank == 0 ) {
      MDI_Send_Command("<DENSITY", qm_comm);
      MDI_Recv(density, ngrid, MDI_DOUBLE, qm_comm);
    }
    MPI_Bcast( density, ngrid, MPI_DOUBLE, 0, world_comm );
    if ( myrank == 0 ) {
      time_end_section = std::chrono::high_resolution_clock::now();
      double time_sec =std::chrono::duration_cast<std::chrono::milliseconds>(time_end_section-time_start_section).count() / 1000.0;
      time_qm_density += time_sec;
    }

    // Calculate the forces on the MM atoms from the QM electric field
    if ( myrank == 0 ) {
      time_start_section = std::chrono::high_resolution_clock::now();
    }
    pw_electrostatic_forces(natoms, masses, ngrid, grid, density, mm_coords, mm_charges, world_comm, 
			       qm_start, qm_end, qm_charges, forces_ec_mm);
    if ( myrank == 0 ) {
      time_end_section = std::chrono::high_resolution_clock::now();
      double time_sec =std::chrono::duration_cast<std::chrono::milliseconds>(time_end_section-time_start_section).count() / 1000.0;
      time_forces += time_sec;
    }

    // Send the coordinates to the MM subset engine
    if ( myrank == 0 ) {
      time_start_section = std::chrono::high_resolution_clock::now();
    }
    if ( myrank == 0 ) {
      MDI_Send_Command(">COORDS", mm_sub_comm);
      MDI_Send(&mm_coords[3*(qm_start-1)], 3*natoms_qm, MDI_DOUBLE, mm_sub_comm);
    }

    // Get the forces from the MM subset engine
    if ( myrank == 0 ) {
      MDI_Send_Command("<FORCES", mm_sub_comm);
      MDI_Recv(mm_force_on_qm_atoms, 3*natoms_qm, MDI_DOUBLE, mm_sub_comm);
    }

    // Have the MM engine proceed to the @PRE-FORCES node
    if ( myrank == 0 ) {
      MDI_Send_Command("@PRE-FORCES", mm_comm);
    }

    // Get the MM forces
    if ( myrank == 0 ) {
      MDI_Send_Command("<FORCES", mm_comm);
      MDI_Recv(forces_mm, 3*natoms, MDI_DOUBLE, mm_comm);
    }

    // Add the QM forces to the MM forces
    if ( myrank == 0 ) {
      int i_qm = 0;
      for (int i_atom=0; i_atom < natoms; i_atom++) {
	if ( mm_mask[i_atom] != -1 ) {
	  forces_mm[3*i_atom+0] += forces_qm[3*i_qm+0] - mm_force_on_qm_atoms[3*i_qm+0] + forces_ec_mm[3*i_atom+0];
	  forces_mm[3*i_atom+1] += forces_qm[3*i_qm+1] - mm_force_on_qm_atoms[3*i_qm+1] + forces_ec_mm[3*i_atom+1];
	  forces_mm[3*i_atom+2] += forces_qm[3*i_qm+2] - mm_force_on_qm_atoms[3*i_qm+2] + forces_ec_mm[3*i_atom+2];
	  i_qm++;
	}
	else {
	  forces_mm[3*i_atom+0] += forces_ec_mm[3*i_atom+0];
	  forces_mm[3*i_atom+1] += forces_ec_mm[3*i_atom+1];
	  forces_mm[3*i_atom+2] += forces_ec_mm[3*i_atom+2];
	}
      }
    }

    // Send the updated forces to the MM main engine
    if ( myrank == 0 ) {
      MDI_Send_Command(">FORCES", mm_comm);
      MDI_Send(forces_mm, 3*natoms, MDI_DOUBLE, mm_comm);
    }

    // Get the MM energy
    if ( myrank == 0 ) {
      MDI_Send_Command("<ENERGY", mm_comm);
      MDI_Recv(&mm_energy, 1, MDI_DOUBLE, mm_comm);
    }

    // Have the MM engine proceed to the @COORDS node, which completes the timestep
    if ( myrank == 0 ) {
      MDI_Send_Command("@COORDS", mm_comm);
    }
    if ( myrank == 0 ) {
      time_end_section = std::chrono::high_resolution_clock::now();
      double time_sec =std::chrono::duration_cast<std::chrono::milliseconds>(time_end_section-time_start_section).count() / 1000.0;
      time_coords += time_sec;
    }

    if ( myrank == 0 ) {
      std::cout << "timestep: " << iiteration << " " << mm_energy << " " << qm_energy << std::endl;
    }
  }

  // Free memory
  delete [] grid;
  delete [] density;
  delete [] mm_coords;
  delete [] mm_charges;
  delete [] masses;
  delete [] forces_mm;
  delete [] qm_charges;
  delete [] forces_qm;
  delete [] qm_cell;
  delete [] mm_cell;
  delete [] forces_ec_mm;
  delete [] mm_force_on_qm_atoms;
  delete [] mm_mask;

  // Send the "EXIT" command to each of the engines
  if ( myrank == 0 ) {
    MDI_Send_Command("EXIT", mm_comm);
    MDI_Send_Command("EXIT", mm_sub_comm);
    MDI_Send_Command("EXIT", qm_comm);
  }

  // Synchronize all MPI ranks
  MPI_Barrier(world_comm);

  // Get the current time
  chrono_time time_end;
  if (myrank == 0 ) {
    time_end = std::chrono::high_resolution_clock::now();
    double total_time = std::chrono::duration_cast<std::chrono::milliseconds>(time_end-time_start).count() / 1000.0;
    std::cout << "Query time:         " << time_query << std::endl;
    std::cout << "Recenter time:      " << time_recenter << std::endl;
    std::cout << "QM SCF time:        " << time_scf << std::endl;
    std::cout << "QM forces time:     " << time_qm_forces << std::endl;
    std::cout << "QM density time:    " << time_qm_density << std::endl;
    std::cout << "Driver pot time:    " << time_pot << std::endl;
    std::cout << "Driver forces time: " << time_forces << std::endl;
    std::cout << "MM coords time:     " << time_coords << std::endl;
    std::cout << "Total time:         " << total_time << std::endl;
  }

  return 0;
}
