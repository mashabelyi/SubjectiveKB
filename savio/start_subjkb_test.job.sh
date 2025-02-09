#!/bin/bash
# Job name:
#SBATCH --job-name=subjkb_tuning
# dbamman faculty computing account
#SBATCH --account=fc_dbamman
# Partition:
#SBATCH --partition=savio2_gpu
#
# Wall clock limit: (this runs for 48 hrs max)
#SBATCH --time=48:00:00
#SBATCH --output=/global/scratch/mashabelyi/subjkb/output/slurm_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#
# run multiple iterations of the same model
#_ignore_SBATCH --array=0-1
#
# Set email notifications
#_ignore_SBATCH --mail-user=mashabelyi@berkeley.edu
#_ignore_SBATCH --mail-type=all
#
#
# Usage
# -a (architecture) {transE, subjD, subjM, ff, ffs}
# -d (data folder) string
# -o (output(model directory) name) string
#
while getopts a:d:o: option
do
case "${option}"
in
a) MODEL=${OPTARG};;
d) DATA=${OPTARG};;
o) NAME=${OPTARG};;
esac
done

echo ${MODEL} ${MODE} ${NORM} ${LR} ${MARGIN} ${DIM}


source /global/home/users/mashabelyi/anaconda3/bin/activate /global/home/users/mashabelyi/anaconda3/envs/subjkb

# python run.py --model ${MODEL} --mode test --f --data /global/home/users/mashabelyi/subjkb/data/yelp2 --name /global/home/users/mashabelyi/subjkb/models/YELP2_transe_d100_m0.5_lr0.0001_n1 --num_epochs 200 --embedding_dim 100 --patience 10 --norm 1 --optim adam --learning_rate 0.0001 --margin 0.5
python /global/home/users/mashabelyi/subjkb/src/run.py --model ${MODEL} --mode test --data ${DATA} --name ${NAME}

source /global/home/users/mashabelyi/anaconda3/bin/deactivate