# Submiting jobs on savio

My home folder on savio
|
|- jobs # bash scripts to start batch jobs
|- subjkb
|  |- data # one subdir per dataset
|  |- src # python code for training

Model training files are written to my scratch folder (unlimited storage, but files deleted if not used for 6 months)
'/global/scratch/mashabelyi/'


## Using Conda

### Instal Anaconda3
https://docs.anaconda.com/anaconda/install/linux/

Create new environment
Will create an environment folder in anaconda3/envs
```
anaconda3/bin/conda create --name subjkb python=3  
```

Enter environment
```
source anaconda3/bin/activate anaconda3/envs/subjkb
```
or
```
conda activate subjkb
```

Check installed packages in environment
```
conda list
```

Instal Dependencies
```
conda install -c anaconda pytorch-gpu
```

Check that pytorch was instaleld and works
```
python
>>> import torch
>>> torch.cuda.is_available()
```

Check size of folders in root dir (there is a 10 GB limit, and conda takes up 7.3G)
```
du -sh /global/home/users/mashabelyi/* | sort -h
```

## Submit jobs

```
sbatch /path/to/test.job
```

Check status of running jobs
```
squeue -u mashabelyi
```

Login to the node your job is running
```
srun --jobid=$your_job_id --pty /bin/bash
```

Check personal SU usage
```
check_usage.sh -E -u mashabelyi
```

Check group SU usage
```
check_usage.sh -E -a fc_dbamman
```

To cancel one job:
```
scancel <jobid>
```

To cancel all the jobs for a user:
```
scancel -u <username>
```

To cancel all the pending jobs for a user:
```
scancel -t PENDING -u <username>
```

scratch folder location
```
/global/scratch/mashabelyi
```

More useful commands here: https://docs.rc.fas.harvard.edu/kb/convenient-slurm-commands/

## My Scripts

Edit `tune_model.sh` to set the search grid for hyperparameters. Input args: -d (path to data), -o (path to output folder), -m (model name {transE, subjD, subjM, ff, ffs}) 

Running `tune_model.sh` will activate `sbatch start_subjkb.job` for every combination of hyperparameters defined in `tune_model.sh`.

tune transE
```
./tune_model.sh -m transE -d ../subjkb/data/yelp2/ -o /global/scratch/mashabelyi/subjkb/models/YELP2

./tune_model.sh -m transE -d ../subjkb/data/yelp2_rules/ -o /global/scratch/mashabelyi/subjkb/models/YELP2_rules/transe/yelp2_rules

./tune_model.sh -m transE -d ../subjkb/data/1900_1920_S18_0419/ -o /global/scratch/mashabelyi/subjkb/models/suffrage/transe/suffrage

./tune_model.sh -m transE -d ../subjkb/data/1900_1920_pybart/ -o /global/scratch/mashabelyi/subjkb/models/1900_1920_pybart/transe/suffrage

./tune_model.sh -m transE -d ../subjkb/data/1900_1920_pybart_full/ -o /global/scratch/mashabelyi/subjkb/models/1900_1920_pybart_full/transe/suffrage

./tune_model.sh -m transE -d ../subjkb/data/climate_tv/ -o /global/scratch/mashabelyi/subjkb/models/climate_tv/transe/climate

./tune_model.sh -m transE -d ../subjkb/data/climate_tv/ -o /global/scratch/mashabelyi/subjkb/models/climate_tv_normFix/transe/climate

./tune_model.sh -m transE -d ../subjkb/data/climate_tv/ -o /global/scratch/mashabelyi/subjkb/models/climate_tv_normEnts/transe/climate





./tune_model.sh -m transE -d ../subjkb/data/yelp5_minie_0619/ -o /global/scratch/mashabelyi/subjkb/models/yelp5_minie_0619/transe/yelp5



```
./tune_model.sh -m transE -d ../subjkb/data/yelp2/ -o /global/scratch/mashabelyi/subjkb/subjkb/models/YELP2

tune subjD
```
./tune_model.sh -m subjD -d ../subjkb/data/yelp2/ -o /global/scratch/mashabelyi/subjkb/models/YELP2

./tune_model.sh -m subjD -d ../subjkb/data/yelp2_rules/ -o /global/scratch/mashabelyi/subjkb/models/YELP2_rules/subjd/yelp2_rules

./tune_model.sh -m subjD -d ../subjkb/data/1900_1920_S18_0419/ -o /global/scratch/mashabelyi/subjkb/models/suffrage/subjd/suffrage

./tune_model.sh -m subjD -d ../subjkb/data/1900_1920_pybart/ -o /global/scratch/mashabelyi/subjkb/models/1900_1920_pybart/subjd/suffrage

./tune_model.sh -m subjD -d ../subjkb/data/1900_1920_pybart_full/ -o /global/scratch/mashabelyi/subjkb/models/1900_1920_pybart_full/subjd/suffrage

./tune_model.sh -m subjD -d ../subjkb/data/climate_tv/ -o /global/scratch/mashabelyi/subjkb/models/climate_tv/subjd/climate

./tune_model.sh -m subjD -d ../subjkb/data/climate_tv/ -o /global/scratch/mashabelyi/subjkb/models/climate_tv_normFix/subjd/climate

./tune_model.sh -m subjD -d ../subjkb/data/climate_tv/ -o /global/scratch/mashabelyi/subjkb/models/climate_tv_normEnts/subjd/climate

./tune_model.sh -m subjD -d ../subjkb/data/yelp5_minie_0619/ -o /global/scratch/mashabelyi/subjkb/models/yelp5_minie_0619/subjd/yelp5

## running subjD after adding regularization of the golden scores
./tune_model.sh -m subjD -d ../subjkb/data/yelp5_minie_0619/ -o /global/scratch/mashabelyi/subjkb/models/yelp5_minie_0619/subjd_reggold/yelp5

```

tune subjM
```
./tune_model.sh -m subjM -d ../subjkb/data/yelp2/ -o /global/scratch/mashabelyi/subjkb/models/YELP2

./tune_model.sh -m subjM -d ../subjkb/data/yelp2_pybart/ -o /global/scratch/mashabelyi/subjkb/models/yelp2_pybart/subjm/yelp2

./tune_model.sh -m subjM -d ../subjkb/data/1900_1920_pybart/ -o /global/scratch/mashabelyi/subjkb/models/1900_1920_pybart/subjm/suffrage

./tune_model.sh -m subjM -d ../subjkb/data/climate_tv/ -o /global/scratch/mashabelyi/subjkb/models/climate_tv_normEnts/subjm/climate


```

tune HyTe
```
./tune_model.sh -m hyte -d ../subjkb/data/yelp5_minie_0619/ -o /global/scratch/mashabelyi/subjkb/models/yelp5_minie_0619/hyte/yelp5
```

## start single job
```
sbatch start_subjkb.job -a transE -d ../subjkb/data/yelp2_pybart -m train-val -n 1 -l 0.0001 -r 0.5 -s 100 -o /global/scratch/mashabelyi/subjkb/models/yelp2_pybart_normfix/transe/yelp2

sbatch start_subjkb.job -a subjD -d ../subjkb/data/yelp2_pybart -m train-val -n 1 -l 0.0001 -r 0.5 -s 100 -o /global/scratch/mashabelyi/subjkb/models/yelp2_pybart_normfix/subjd/yelp2
```

## Find best performing models based on H@10 on validation set

```
python find_best_model_dir.py /global/scratch/mashabelyi/subjkb/models/yelp2_pybart/transe/

Best heads H@10 =  0.2927739195316542
yelp2_transE_d100_m0.5_lr0.001_n1_1

Best tails H@10 =  0.12584897357898564
yelp2_transE_d100_m0.5_lr0.001_n1_1

```

```
python find_best_model_dir.py /global/scratch/mashabelyi/subjkb/models/yelp2_pybart/subjm/

Best heads H@10 =  0.3103177791807765
yelp2_subjM_d100_m0.5_lr0.0001_n1

Best tails H@10 =  0.1300579693508581
yelp2_subjM_d100_m0.5_lr0.0001_n1

```

```
python find_best_model_dir.py /global/scratch/mashabelyi/subjkb/models/yelp2_pybart/subjd/

Best heads H@10 =  0.3429566282117517
yelp2_subjD_d100_m0.5_lr0.0001_n1_1

Best tails H@10 =  0.17320017601255192
yelp2_subjD_d100_m0.5_lr0.0001_n1_1

```

```
python find_best_model_dir.py /global/scratch/mashabelyi/subjkb/models/1900_1920_pybart/transe/

Best heads H@10 =  0.14658131255886733
suffrage_transE_d100_m0.5_lr0.001_n1

Best tails H@10 =  0.04307083220255192
suffrage_transE_d100_m0.5_lr0.0001_n1
```

```
python find_best_model_dir.py /global/scratch/mashabelyi/subjkb/models/1900_1920_pybart/subjd  

Best heads H@10 =  0.19106761812018822
suffrage_subjD_d100_m0.5_lr0.0001_n1

Best tails H@10 =  0.06233936239842932
suffrage_subjD_d100_m0.5_lr0.0001_n1
```



## Evaluate on Test Set
```
sbatch start_subjkb_test.job -a transE -d ../subjkb/data/yelp2_pybart/ -o /global/scratch/mashabelyi/subjkb/models/yelp2_pybart/transe/yelp2_transE_d100_m0.5_lr0.001_n1_1

sbatch start_subjkb_test.job -a subjD -d ../subjkb/data/yelp2_pybart/ -o /global/scratch/mashabelyi/subjkb/models/yelp2_pybart/subjd/yelp2_subjD_d100_m0.5_lr0.0001_n1_1

sbatch start_subjkb_test.job -a subjM -d ../subjkb/data/yelp2_pybart/ -o /global/scratch/mashabelyi/subjkb/models/yelp2_pybart/subjm/yelp2_subjM_d100_m0.5_lr0.0001_n1


sbatch start_subjkb_test.job -a transE -d ../subjkb/data/1900_1920_pybart/ -o /global/scratch/mashabelyi/subjkb/models/1900_1920_pybart/transe/suffrage_transE_d100_m0.5_lr0.001_n1

## last one left to checK:
sbatch start_subjkb_test.job -a subjD -d ../subjkb/data/1900_1920_pybart/ -o /global/scratch/mashabelyi/subjkb/models/1900_1920_pybart/subjd/suffrage_subjD_d100_m0.5_lr0.0001_n1

ls /global/scratch/mashabelyi/subjkb/models/1900_1920_pybart/subjd/suffrage_subjD_d100_m0.5_lr0.0001_n1



```