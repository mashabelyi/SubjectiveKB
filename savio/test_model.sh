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


for margin in $margins
do
	for lr in $lrs
	do
		for dim in $dims
		do
			for norm in $norms
			do
				sbatch start_subjkb_test.job -a ${MODEL} -d ${DATA} -o ${OUT}
			done
		done
	done

done