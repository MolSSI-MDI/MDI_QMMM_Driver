git clone https://github.com/MolSSI-MDI/lammps.git
cd lammps
git checkout mdi
cd src
make yes-standard
make no-gpu
make no-kim
make no-kokkos
make no-kspace
make no-latte
make no-meam
make no-mpiio
make no-mscg
make no-poems
make no-python
make no-reax
make no-voronoi
make no-user-qmmm
make yes-user-mdi
cd ../lib/mdi
python Install.py -m gcc
cd ../../src
  
# Build LAMMPS
if test "${LAMMPS_INSTALL}" = 'serial'; then make mpi-stubs; fi
make -j 4 "${LAMMPS_INSTALL}"
