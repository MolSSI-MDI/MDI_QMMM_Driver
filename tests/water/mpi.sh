#location of required codes
DRIVER_LOC=$(cat ../locations/MDI_QMMM_Driver)
LAMMPS_LOC=$(cat ../locations/LAMMPS)
QE_LOC=$(cat ../locations/QE)

#remove old files
if [ -d work ]; then 
  rm -r work
fi

#create work directory
cp -r data work
cd work

#launch the codes
mpiexec -n 1 ${DRIVER_LOC} -mdi "-role DRIVER -name driver -method MPI" : \
    -n 1 ${LAMMPS_LOC} -mdi "-role ENGINE -name MM -method MPI" -in lammps.in > lammps.out : \
    -n 8 ${QE_LOC} -mdi "-role ENGINE -name QM -method MPI" -in qe.in > qe.out

wait
