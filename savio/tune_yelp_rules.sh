#!/bin/bash
#
#
# Inputs
# -d (path to data)
# -o (output name)
# -m (model name) {transE, subjD, subjM, ff, ffs}
#
# Usage
# tune_model.sh -m transE -d /path/to/data/ -o /path/to/output/dirname 

while getopts d:o:m: option
do
case "${option}"
in
m) MODEL=${OPTARG};;
d) DATA=${OPTARG};;
o) OUT=${OPTARG};;
esac
done

#margins='0 0.5 1 2'
margins='0.5'
lrs='0.0001 0.001 0.01 0.1'
dims='50 100'
norms='1 2'

for margin in $margins
do
	for lr in $lrs
	do
		for dim in $dims
		do
			for norm in $norms
			do
				sbatch start_subjkb.job -m train-val -a ${MODEL} -n ${norm} -l ${lr} -r ${margin} -s ${dim} -d ${DATA} -o ${OUT}
				# python3 train.py --data ../SubjKB/data/yelp/by_rels/ --name YELP_d${dim}_m${margin}_lr${lr}_n${norm} --num_epochs 100 --validation_step 100 --embedding_dim $dim --patience 3 --l2reg 0 --norm $norm --optim sgd --learning_rate $lr --margin $margin
			done
		done
	done

done