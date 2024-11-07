#!/bin/bash
#
#SBATCH -p C-Short   # partition 
#SBATCH -N 1 # number of nodes 
#SBATCH --ntasks=2 
#SBATCH --mem 32G # memory pool for all cores 
#SBATCH -t 0-1:00 # time (D-HH:MM) 

./runJL dev/pareto.opt.dev.jl 1bs_sens_epr.jl 2 20241106/slurm-firstest

