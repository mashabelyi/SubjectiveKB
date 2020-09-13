#!/bin/bash
while getopts m:a:n:l:r:d: option
do
case "${option}"
in
m) MODEL=${OPTARG};;
a) MODE=${OPTARG};;
n) NORM=${OPTARG};;
l) LR=${OPTARG};;
r) MARGIN=${OPTARG};;
d) DIM=${OPTARG};;
esac
done

echo ${MODEL} ${MODE} ${NORM} ${LR} ${MARGIN} ${DIM}