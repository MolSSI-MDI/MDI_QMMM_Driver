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

export OMP_NUM_THREADS=1

#launch the codes
mpiexec -n 1 ${DRIVER_LOC} -mdi "-role DRIVER -name driver -method MPI -out driver.out" : \
    -n 1 ${LAMMPS_LOC} -mdi "-role ENGINE -name MM -method MPI -out lammps_main.out" -in lammps_main.in : \
    -n 1 ${LAMMPS_LOC} -mdi "-role ENGINE -name MM_SUB -method MPI -out lammps_sub.out" -in lammps_sub.in : \
    -n 32 ${QE_LOC} -mdi "-role ENGINE -name QM -method MPI -out qe.out" -in qe.in

wait
