#!/bin/bash

margins='0 0.5 1 2 3'
lrs='0.0001 0.0005 0.001 0.005'
dims='50 100'
norms='1 2'

counter=1

for margin in $margins
do
	for lr in $lrs
	do
		for dim in $dims
		do
			for norm in $norms
			do
				# echo $dim $margin $lr $norm
				python3 train.py --data ../SubjKB/data/yelp/by_rels/ --name YELP_d${dim}_m${margin}_lr${lr}_n${norm} --num_epochs 100 --validation_step 100 --embedding_dim $dim --patience 3 --l2reg 0 --norm $norm --optim sgd --learning_rate $lr --margin $margin --val_pkl val_dict.pkl
			done
		done
	done

done
