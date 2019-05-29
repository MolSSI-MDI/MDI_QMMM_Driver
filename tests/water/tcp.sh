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

#launch QE
mpirun -n 32 ${QE_LOC} -mdi "-role ENGINE -name QM -method TCP -port 8022 -hostname localhost" -in qe.in > qe.out &

#launch LAMMPS
${LAMMPS_LOC} -mdi "-role ENGINE -name MM -method TCP -port 8022 -hostname localhost" -in lammps_main.in > lammps_main.out &
${LAMMPS_LOC} -mdi "-role ENGINE -name MM_SUB -method TCP -port 8022 -hostname localhost" -in lammps_sub.in > lammps_sub.out &

#launch driver
mpirun -n 16 ${DRIVER_LOC} -mdi "-role DRIVER -name driver -method TCP -port 8022" &

wait
