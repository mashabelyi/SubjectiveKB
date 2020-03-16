"""
Train TransE baseline model fro KB Completion

USAGE

python3 train.py --data data --name NEWS --batch_size 128 --num_epochs 50 --validation_step 10 \
--debug_nTrain 100 --debug_nVal 100

"""
import json
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau
from torch.nn import MarginRankingLoss

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from builddata import *
from batching import BatchLoader
from logger import Logger
from transE import TransE_SourceMatrix

import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Parameters
# ==================================================
parser = ArgumentParser("TransE", formatter_class=ArgumentDefaultsHelpFormatter, conflict_handler='resolve')

parser.add_argument("--data", default="data/NEWS", help="Data sources.", required=True)
parser.add_argument("--name", default="NEWS", required=True, help="Model name. Will store model files in directory with name ")

parser.add_argument("--test_pkl", required=False, help="Path to pickle file with test dict")
parser.add_argument("--val_pkl", required=False, help="Path to pickle file with val dict")


args = parser.parse_args()
print(args)


model_dir = args.name


# LOAD DATA
# =========================================
print("Loading data...")

if os.path.isfile(os.path.join(args.data, 'train2id.txt')):
	train, val, test, bernoulli_p, goldens, \
	all_ents, all_rels, rel2id, ent2id, src2id  = OpenKELoader(path=args.data)
	all_sources = []
else:
	train, val, test, bernoulli_p, goldens, \
	all_ents, all_rels, rel2id, ent2id, src2id  = build_data(path=args.data)
	all_sources = list(src2id.values()) #[0,1]

print("\nLoaded {} entities".format(len(all_ents)))
print("Loaded {} relations\n".format(len(all_rels)))

# DEV - Subset training and validation sets
if args.debug_nTrain > 0:
	print("subsetting data to {} Train".format(min(len(train), args.debug_nTrain)))
	train = train[:min(len(train), args.debug_nTrain)]

if args.debug_nVal > 0:
	print("subsetting data to {} Val".format(min(args.debug_nVal, len(val))))
	val = val[:min(args.debug_nVal, len(val))]

if args.debug_nTest > 0:
	print("subsetting data to {} Test".format(min(args.debug_nTest, len(test))))
	test = test[:min(args.debug_nTest, len(test))]

# SAVE CONFIG
# =========================================
# save input arguments in model folder
# config = args.__dict__
# config['numTrain'] = len(train)
# config['numVal'] = len(val)
# config['numTest'] = len(test)
# with open(os.path.join(model_dir, 'config.json'), 'w') as f:
# 	json.dump(config, f, indent=2)

with open(os.path.join(model_dir, 'config.json'), 'r') as f:
	config = json.load(f)

# BATCH LOADER
# =========================================
loader = BatchLoader(train, bernoulli_p, goldens, all_ents, all_sources, batch_size=config['batch_size'], neg_ratio=config['neg_ratio'])

print("Loading data... finished!")

# INITIALIZE MODEL
# =========================================
transE = TransE_SourceMatrix(len(all_rels), len(ent2id), len(src2id), dim=config['embedding_dim'], norm=config['norm'])
transE.to(device)


print("====================================\n")



print("\nTESTING")
print("====================================\n")


# EVAL ON TEST SET
transE.load_state_dict(torch.load(os.path.join(model_dir, 'best_val_loss_state_dict.pt')))
transE.eval()


# test_dict = loader.validation_triples(test)

if args.val_pkl:
	print("Loading val from pickle file")
	val_dict = pickle.load(open(args.val_pkl,"rb"))
else:
	val_dict = loader.validation_triples(val)


MR, MRR, h10 = transE.validate_fast(val, val_dict, all_ents, all_rels, all_sources)
print("\n\nRESULTS ON TEST SET")
print("====================================")
print("MR = {}, MRR = {:.4}, H@10 = {:.4}\n".format(round(MR), MRR, h10))


with open(os.path.join(model_dir, 'val_eval1.tsv'), 'w') as f:
	f.write("MR\tMRR\tH@10\n")
	f.write("{}\t{}\t{}\n".format(MR, MRR, h10))











