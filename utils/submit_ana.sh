#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:T4:16
#SBATCH --time=00:20:00
#SBATCH --mem=500GB
#SBATCH --nodes=1
#SBATCH --output=./anupam-17052024-neural_nets.log
#SBATCH --job-name=anupam-neural-net
#SBATCH --constraint=el7

source /etc/profile.d/modules.sh
module use /apps/modulefiles
module load anaconda3/23.3.1
module load cuda/11.4.2

# load your environment (your environment might be named differently)
cd /home/anupam/project/pinn-halld
source activate pinnd_base_env

# Set the directory where the Jef Net is stored: (your directory might be named differently)
anupamDir=/home/anupam/project/pinn-halld/mlflow_jlab

# for running registry -----------------------------
# #python $jefDir/run_registry/jef_driver.py --device cpu
# python $anupamDir/run_registry/jef_driver.py 
# --------------------------------------------------

#for running cfg -----------------------------------
python $anupamDir/run_registry/anupam_driver.py 

