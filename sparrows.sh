#!/bin/bash
#SBATCH --job-name=Sparrows				# Job name 
#SBATCH --partition=batch				# Partition name
#SBATCH --mail-type=ALL 				# e-mail me when anything happens
#SBATCH --mail-user=heg08492@uga.edu  			# my email to send things to
#SBATCH --ntasks=1					# Run a single task
#SBATCH --cpus-per-task=3				# I want to run this in parallel on 5 cores
#SBATCH --mem=50gb   					# I’m requesting X GB of memory
#SBATCH --time=6-10:00:00   				# How long (max) I want this to run in hours:mins:seconds
#SBATCH --output=%x_%j.out 				# name of output log (not your model output!)
#SBATCH --error=%x.%j.err			# name of error log 
#SBATCH --constraint=EDR				# this is because of the version of R I'm using
 
cd $SLURM_SUBMIT_DIR					# submit my request

module load R/4.3.1-foss-2022a	 			# load R

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK 		#tell it how to run the parallel stuff

R CMD BATCH Heather_Sparrows.R  				# Run my R script
