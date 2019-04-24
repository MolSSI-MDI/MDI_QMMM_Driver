#include <iostream>
#include <mpi.h>
#include <stdexcept>
#include <string.h>
extern "C" {
#include "mdi.h"
}

using namespace std;

int main(int argc, char **argv) {

  // Initialize the MPI environment
  MPI_Comm world_comm;
  MPI_Init(&argc, &argv);

  // Read through all the command line options
  int iarg = 1;
  bool initialized_mdi = false;
  while ( iarg < argc ) {

    if ( strcmp(argv[iarg],"-mdi") == 0 ) {

      // Ensure that the argument to the -mdi option was provided
      if ( argc-iarg < 2 ) {
	throw runtime_error("The -mdi argument was not provided.");
      }

      // Initialize the MDI Library
      world_comm = MPI_COMM_WORLD;
      int ret = MDI_Init(argv[iarg+1], &world_comm);
      if ( ret != 0 ) {
	throw runtime_error("The MDI library was not initialized correctly.");
      }
      initialized_mdi = true;
      iarg += 2;

    }
    else {
      throw runtime_error("Unrecognized option.");
    }

  }
  if ( not initialized_mdi ) {
    throw runtime_error("The -mdi command line option was not provided.");
  }

  // Connect to the engines
  MDI_Comm mm_comm = MDI_NULL_COMM;
  MDI_Comm mm_sub_comm = MDI_NULL_COMM;
  MDI_Comm qm_comm = MDI_NULL_COMM;
  int nengines = 3;
  for (int iengine=0; iengine < nengines; iengine++) {
    MDI_Comm comm = MDI_Accept_Communicator();
 
    // Determine the name of this engine
    char* engine_name = new char[MDI_NAME_LENGTH];
    MDI_Send_Command("<NAME", comm);
    MDI_Recv(engine_name, MDI_NAME_LENGTH, MDI_CHAR, comm);
 
    cout << "Engine name: " << engine_name << endl;
 
    if ( strcmp(engine_name, "MM") == 0 ) {
      if ( mm_comm != MDI_NULL_COMM ) {
	throw runtime_error("Accepted a communicator from a second main MM engine.");
      }
      mm_comm = comm;
    }
    else if ( strcmp(engine_name, "QM") == 0 ) {
      if ( qm_comm != MDI_NULL_COMM ) {
	throw runtime_error("Accepted a communicator from a second QM engine.");
      }
      qm_comm = comm;
    }
    else if ( strcmp(engine_name, "MM_SUB") == 0 ) {
      if ( mm_sub_comm != MDI_NULL_COMM ) {
	throw runtime_error("Accepted a communicator from a second subset MM engine.");
      }
      mm_sub_comm = comm;
    }
    else {
      throw runtime_error("Unrecognized engine name.");
    }
 
    delete[] engine_name;
  }

  // VARIABLES THAT SHOULD BE READ FROM AN INPUT FILE
  int niterations = 30;  // Number of MD iterations
  int qmmm_mode = 2;
  int qm_start = 25;
  int qm_end = 27;

  // Simulation variables
  int natoms, natoms_qm;
  int ntypes;
  double qm_energy;
  double mm_energy;

  // Receive the number of atoms from the main MM engine
  MDI_Send_Command("<NATOMS", mm_comm);
  MDI_Recv(&natoms, 1, MDI_INT, mm_comm);

  // Receive the number of MM atom types from the main MM engine
  MDI_Send_Command("<NTYPES", mm_comm);
  MDI_Recv(&ntypes, 1, MDI_INT, mm_comm);

  // Receive the number of QM atoms from the subset MM engine
  MDI_Send_Command("<NATOMS", mm_sub_comm);
  MDI_Recv(&natoms_qm, 1, MDI_INT, mm_sub_comm);

  cout << "MM atoms: " << natoms << endl;
  cout << "QM atoms: " << natoms_qm << endl;

  // Allocate the arrays for the coordinates and forces
  double mm_coords[3*natoms];
  double mm_charges[natoms];
  double masses[ntypes+1];
  double forces_mm[3*natoms];
  double forces_qm[3*natoms_qm];
  double forces_ec[3*natoms_qm];
  double forces_ec_mm[3*natoms];
  double mm_force_on_qm_atoms[3*natoms_qm];
  double mm_cell[9];
  int types[natoms];
  int mm_mask[natoms];

  // Receive the MM types
  MDI_Send_Command("<TYPES", mm_comm);
  MDI_Recv(&types, natoms, MDI_INT, mm_comm);

  // Set the MM mask
  // this is -1 for non-QM atoms, and 1 for QM atoms
  for (int i=0; i<natoms; i++) {
    mm_mask[i] = -1;
  }
  for (int i=qm_start-1; i<=qm_end-1; i++) {
    mm_mask[i] = types[i];
  }

  // Send the QMMM mode to QE
  // SHOULD CHANGE THIS, IF POSSIBLE
  MDI_Send_Command(">QMMM_MODE", qm_comm);
  MDI_Send(&qmmm_mode, 1, MDI_INT, qm_comm);
  

  // Have the MD engine initialize a new MD simulation
  MDI_Send_Command("MD_INIT", mm_comm);

  // Perform each iteration of the simulation
  for (int iiteration = 0; iiteration < niterations; iiteration++) {

    // Send the number of QM atoms to the QM engine
    // NOTE: THIS IS CURRENTLY REQUIRED IN ORDER TO INITIALIZE ARRAYS - SHOULD CHAGE
    MDI_Send_Command(">NATOMS", qm_comm);
    MDI_Send(&natoms_qm, 1, MDI_INT, qm_comm);

    // Send the number of MM atoms to the QM engine
    // SHOULD CHANGE THIS
    MDI_Send_Command(">MM_NATOMS", qm_comm);
    MDI_Send(&natoms, 1, MDI_INT, qm_comm);

    // Send the number of MM atom types to the QM engine
    // SHOULD NOT BE REQUIRED
    MDI_Send_Command(">NTYPES", qm_comm);
    MDI_Send(&ntypes, 1, MDI_INT, qm_comm);

    // Receive the MM cell dimensions from the MM engine
    // SHOULD NOT BE REQUIRED
    MDI_Send_Command("<CELL", mm_comm);
    MDI_Recv(&mm_cell, 9, MDI_DOUBLE, mm_comm);

    // Send the MM cell dimensions to the QM engine
    MDI_Send_Command(">MM_CELL", qm_comm);
    MDI_Send(&mm_cell, 9, MDI_DOUBLE, qm_comm);

    // Receive the coordinates from the MM engine
    MDI_Send_Command("<COORDS", mm_comm);
    MDI_Recv(&mm_coords, 3*natoms, MDI_DOUBLE, mm_comm);

    // Receive the charges from the MM engine
    MDI_Send_Command("<CHARGES", mm_comm);
    MDI_Recv(&mm_charges, natoms, MDI_DOUBLE, mm_comm);

    // Receive the masses from the MM engine
    MDI_Send_Command("<MASSES", mm_comm);
    MDI_Recv(&masses, ntypes+1, MDI_DOUBLE, mm_comm);

    // Send the MM mask to the QM engine
    MDI_Send_Command(">MM_MASK", qm_comm);
    MDI_Send(&mm_mask, natoms, MDI_INT, qm_comm);

    // Send the MM types to the QM engine
    MDI_Send_Command(">MM_TYPES", qm_comm);
    MDI_Send(&types, natoms, MDI_INT, qm_comm);

    // Send the MM charges to the QM engine
    MDI_Send_Command(">MM_CHARGES", qm_comm);
    MDI_Send(&mm_charges, natoms, MDI_DOUBLE, qm_comm);

    // Send the MM masses to the QM engine
    MDI_Send_Command(">MM_MASSES", qm_comm);
    MDI_Send(&masses, ntypes+1, MDI_DOUBLE, qm_comm);

    // Send the MM coordinates to the QM engine
    MDI_Send_Command(">MM_COORDS", qm_comm);
    MDI_Send(&mm_coords, 3*natoms, MDI_DOUBLE, qm_comm);

    // Send the QM coordinates to the QM engine
    /*
    cout.precision(17);
    cout << "   O: " << mm_coords[3*(qm_start-1)+0] << " " << mm_coords[3*(qm_start-1)+1] << " " << mm_coords[3*(qm_start-1)+2] << endl;
    cout << "   H: " << mm_coords[3*(qm_start+0)+0] << " " << mm_coords[3*(qm_start+0)+1] << " " << mm_coords[3*(qm_start+0)+2] << endl;
    cout << "   H: " << mm_coords[3*(qm_start+1)+0] << " " << mm_coords[3*(qm_start+1)+1] << " " << mm_coords[3*(qm_start+1)+2] << endl;
    */
    MDI_Send_Command(">COORDS", qm_comm);
    MDI_Send(&mm_coords[3*(qm_start-1)], 3*natoms_qm, MDI_DOUBLE, qm_comm);

    // Have the QM process recenter the coordinates
    MDI_Send_Command("RECENTER", qm_comm);

    // Have the QM engine perform an SCF calculation
    MDI_Send_Command("SCF", qm_comm);

    // Get the QM energy
    MDI_Send_Command("<ENERGY", qm_comm);
    MDI_Recv(&qm_energy, 1, MDI_DOUBLE, qm_comm);

    // Get the QM forces
    MDI_Send_Command("<FORCES", qm_comm);
    MDI_Recv(&forces_qm, 3*natoms_qm, MDI_DOUBLE, qm_comm);

    if ( qmmm_mode == 2 ) {

      // Get the EC forces on the QM atoms
      MDI_Send_Command("<EC_FORCES", qm_comm);
      MDI_Recv(&forces_ec, 3*natoms_qm, MDI_DOUBLE, qm_comm);

      // Get the EC forces on the MM atoms
      MDI_Send_Command("<MM_FORCES", qm_comm);
      MDI_Recv(&forces_ec_mm, 3*natoms, MDI_DOUBLE, qm_comm);

    }

    // Send the coordinates to the MM subset engine
    MDI_Send_Command(">COORDS", mm_sub_comm);
    MDI_Send(&mm_coords[3*(qm_start-1)], 3*natoms_qm, MDI_DOUBLE, mm_sub_comm);

    // Get the forces from the MM subset engine
    MDI_Send_Command("<FORCES", mm_sub_comm);
    MDI_Recv(mm_force_on_qm_atoms, 3*natoms_qm, MDI_DOUBLE, mm_sub_comm);

    // Have the MM engine proceed to the @PRE-FORCES node
    MDI_Send_Command("@PRE-FORCES", mm_comm);

    // Get the MM forces
    MDI_Send_Command("<FORCES", mm_comm);
    MDI_Recv(&forces_mm, 3*natoms, MDI_DOUBLE, mm_comm);

    // Add the QM forces to the MM forces
    int i_qm = 0;
    for (int i_atom=0; i_atom < natoms; i_atom++) {
      if ( mm_mask[i_atom] != -1 ) {
	forces_mm[3*i_atom+0] += forces_qm[3*i_qm+0] - mm_force_on_qm_atoms[3*i_qm+0];
	forces_mm[3*i_atom+1] += forces_qm[3*i_qm+1] - mm_force_on_qm_atoms[3*i_qm+1];
	forces_mm[3*i_atom+2] += forces_qm[3*i_qm+2] - mm_force_on_qm_atoms[3*i_qm+2];
	i_qm++;
      }
      else {
	forces_mm[3*i_atom+0] += forces_ec_mm[3*i_atom+0];
	forces_mm[3*i_atom+1] += forces_ec_mm[3*i_atom+1];
	forces_mm[3*i_atom+2] += forces_ec_mm[3*i_atom+2];
      }
    }

    // Send the updated forces to the MM main engine
    MDI_Send_Command(">FORCES", mm_comm);
    MDI_Send(&forces_mm, 3*natoms, MDI_DOUBLE, mm_comm);

    // Get the MM energy
    MDI_Send_Command("<ENERGY", mm_comm);
    MDI_Recv(&mm_energy, 1, MDI_DOUBLE, mm_comm);

    // Have the MM engine proceed to the @COORDS node, which completes the timestep
    MDI_Send_Command("@COORDS", mm_comm);

    cout << "timestep: " << iiteration << " " << mm_energy << " " << qm_energy << endl;
  }

  // Send the "EXIT" command to each of the engines
  MDI_Send_Command("EXIT", mm_comm);
  MDI_Send_Command("EXIT", mm_sub_comm);
  MDI_Send_Command("EXIT", qm_comm);

  // Synchronize all MPI ranks
  MPI_Barrier(world_comm);

  return 0;
}
