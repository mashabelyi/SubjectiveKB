#!/bin/bash

# margins='0 0.5 1 2 3'
margin=$1
lrs='0.005 0.001 0.0005 0.0005'
dims='100 50'
norms='1 2'

for lr in $lrs
do
	for dim in $dims
	do
		for norm in $norms
		do
			# echo $dim $margin $lr $norm
			python3 train_sourceModel.py --data ../SubjKB/data/yelp/by_rels/ --name YELP_src_d${dim}_m${margin}_lr${lr}_n${norm} --num_epochs 100 --validation_step 100 --embedding_dim $dim --patience 3 --l2reg 0 --norm $norm --optim sgd --learning_rate $lr --margin $margin --val_pkl val_dict.pkl
		done
	done
done
