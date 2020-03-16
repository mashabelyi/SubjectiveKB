# Subjective KB

Adapting traditional knowledge base completion models to encode opinions.

## Dependencies
- python3
- pytorch

## Main Files

`transE.py`

Model definitions
- `TransE`: baseline model, implementation of [Bordes et al, 2013](https://www.utc.fr/~bordesan/dokuwiki/_media/en/transe_nips13.pdf)
- `TransE_SourceFull`: (Deviation Model) extension of TransE with additional srouce-specific deviation vectors for each entity and relation
- `TransE_SourceMatrix`: (Matrix Model) extension of TransE with a source-specifi matrix transormation applied to each entity and relation vector

`train.py`

Train baseline model. Usage:
```
python3 train.py --data data/yelp --name yelp_baseline --num_epochs 10 --validation_step 10 --embedding_dim 100 --patience 3 --norm 2 --optim sgd --learning_rate 0.0005 --margin 0.5 --val_pkl /data0/mashabelyi/TransE/val_dict.pkl
```


`train_sourceModel.py`

Train baseline model. Usage:
```
python3 train_sourceModel.py --data data/yelp --name yelp_deviation --num_epochs 10 --validation_step 10 --embedding_dim 100 --patience 3 --norm 2 --optim sgd --learning_rate 0.0005 --margin 0.5 --val_pkl /data0/mashabelyi/TransE/val_dict.pkl
```


`train_sourceMatrix.py`

Train baseline model. Usage:
```
python3 train_sourceMatrix.py --data data/yelp --name yelp_matrix --num_epochs 10 --validation_step 10 --embedding_dim 100 --patience 3 --norm 2 --optim sgd --learning_rate 0.0005 --margin 0.5 --val_pkl /data0/mashabelyi/TransE/val_dict.pkl
```

## Parameters

**data**: path to data folder
**name**: model name (or path), training scripts will create a directory with input name to store training log, model weights, evaluation results
**batch_size**: batch size, default=128
**margin**: margin used in loss function, default=1
**norm**: norm used in loss function, default=2
**patience**: flag to stop training if validation loss does not decrease in `patience` number of epochs
**optin**: 'sgd' or 'adam', default='adam'
**learning_rate**: optimizer learning rate, default=0.001
**num_epochs**: max number of epochs to train
**validation_step**: run evaluation on validation set every `validation_step` number of epochs. This can take a while, so set `validation_step` >= `num_epochs` to prevent any evaluation during training.
**embedding_dim**: dimensionality of entity and relation embeddings
**--val_pkl**: path to a pre-generated pickle file for validation. Passing in this file speeds up evaluation at the end of training. Use a pre-generated file on Redwood at `/data0/mashabelyi/TransE/val_dict.pkl`
 
## Development Paramteres
For development purposes, you may want to train and evaluate on a smaller subset of samples. To do this, use the following parameters to subset the train, validation, and test sets

**--debug_nTrain**: 