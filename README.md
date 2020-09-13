# Subjective KB

Adapting traditional knowledge base completion models to encode opinions.

## Dependencies
- python3
- pytorch


## Data
Processed datasets are uploaded on [google drive](https://drive.google.com/drive/u/1/folders/1u9MOgsu5QnBzvSiMEYznOhrGVI9ggkpK)

Data sources:
- [YELP dataset](https://www.yelp.com/dataset) - filtered to include only restaurant reviews.
- Suffrage newspaper dataset (on Redwood) - filtered to include articles from 18 newspapers between 1900-1920.
- Climate change TV discourse

## Train Models

The line below will initiate model training with specified parameters. Output files will be stored in `path/to/OUTPUT_FOLDER`.
```
python3 run.py --model transE --mode train-val --data /data/yelp2 --name path/to/OUTPUT_FOLDER --num_epochs 200 --embedding_dim 100 --patience 10 --norm 1 --optim adam --learning_rate 0.0001 --margin 0.5
```

### Parameters

**model**: model architecture to train. Options: `transE, subjD, subjM, ff, ffs, hyte, transH`. See `models.py` for code specific to each model.
- `transE`: implementation of Bordes et al 2013
- `subjD`: implementation of SubjKB with deviation vectors
- `subjM`: implementation of SubjKB with a source-specific transformation matrix
- `ff`:  simple 2-layer neural net that treats KB completion as a classifiation problem (not fully developed or tested)
- `ffs`: source-aware vesion of `ff` (not fully developed or tested)
- `TransH`: implementation of Wang et al., 2014
- `hyte`: implementation of [HyTE](https://www.aclweb.org/anthology/D18-1225.pdf)

**mode**: Options: `train, val, test, train-val`
- `train`: train and save the best performing model
- `train-val`: train and save the best performing model. Also run full evaluation on the validation set and report the metrics.
- `val`: load pre-trained model and evaluate on validation set
- `test`: load pre-trained model and evaluate on test set

**data**: path to data directory. The data directory shold contain 3 files:  `train.tsv`, `val.tsv`, `test.tsv`. Each file has 4 columns: head (str), relation (str), tail (str), sourceId.

**name**: path to output directory where all model files will be stored.

### Optional Parameters

**batch_size**: batch size, default=128

**margin**: margin used in loss function, default=1

**norm**: norm used in loss function, default=2

**patience**: flag to stop training if validation loss does not decrease in `patience` number of epochs

**optim**: `sgd` or `adam`, `adagrad`, default=`adam`

**learning_rate**: optimizer learning rate, default=0.001

**num_epochs**: max number of epochs to train, default=200

**embedding_dim**: dimensionality of entity and relation embeddings, default=100
 
### Development Paramteres
For development purposes, you may want to train and evaluate on a smaller subset of samples. To do this, use the following parameters to subset the train, validation, and test sets

**debug_nTrain**: number of samples in training. e.g. `--debug_nTrain 100`

**debug_nVal**: number of samples in validation. e.g. `--debug_nVal 100`

**debug_nTest**: number of samples in testing. e.g. `--debug_nTest 100`


### Relation Extraction

The `relation_extraction` directory contains scripts used to extract (head, rel, tail) tuples from raw corpora.

For extractino with MinIE use open source code: https://github.com/uma-pi1/minie

