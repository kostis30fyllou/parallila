#!/bin/bash

#Max VM size
#PBS -l pvmem=2G

# Max Wall time
#PBS -l walltime=0:01:00  		# Example, 1 minute

# How many nodes and tasks per node
#PBS -l nodes=20:ppn=8 			# 20 nodes with 8 tasks/node => 160 tasks

#Which Queue
#PBS -q parsys			 	# This is the only accessible queue for rbs

#PBS -N 160   			# Jobname - it gives the stdout/err filenames

### Merge std[out|err]
#PBS -k oe

#Change Working directory to SUBMIT directory
cd $PBS_O_WORKDIR  			# THIS IS MANDATORY,  PBS Starts everything from $HOME, one should change to submit directory

#OpenMP Threads
#export OMP_NUM_THREADS=2
# OMP_NUM_THREADS * ppn should be max 8 (the total number of node cores= 8).
# To use OpenMPI remember to include -fopenmp in compiler flags in order to activate OpenMP directives.

# Having modules mpiP and openmpi loaded i.e.
module load mpiP openmpi

mpirun -np 160 mpi.out				# That was compiled on front-end
	
 
# No need for -np -machinefile etc. MPI gets this information from PBS