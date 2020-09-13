import os, json
import pickle
import util as u
from builddata import *
from logger import Logger
from trainer import Trainer
from models import TransE, SubjKB_Deviation, SubjKB_Matrix, FeedForward, FeedForward_Source, HyTE, TransH

import torch.optim as optim
from torch.nn import MarginRankingLoss
from logger import Logger
from batching import BatchLoader
from DataLoader import OpenKELoader

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

args = u.parse_args()					# process input arguments

slurm_task_id = os.environ.get('SLURM_ARRAY_TASK_ID')
if slurm_task_id:
	args.name = args.name + '_' + slurm_task_id

if args.mode.startswith('train'):
	u.create_model_folder(args.name, args.f)		# create model directory

# load and subset data
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

# =========================================
# DEV - Subset training and validation sets
# =========================================
if args.debug_nTrain > 0:
	print("subsetting data to {} Train".format(min(len(train), args.debug_nTrain)))
	train = train[:min(len(train), args.debug_nTrain)]
	

if args.debug_nVal > 0:
	print("subsetting data to {} Val".format(min(args.debug_nVal, len(val))))
	val = val[:min(args.debug_nVal, len(val))]

if args.debug_nTest > 0:
	print("subsetting data to {} Test".format(min(args.debug_nTest, len(test))))
	test = test[:min(args.debug_nTest, len(test))]


# =========================================
# SAVE CONFIG arguments in model folder
# =========================================
if args.mode.startswith('train'):
	config = args.__dict__
	config['numTrain'] = len(train)
	config['numVal'] = len(val)
	config['numTest'] = len(test)
	config['numEntities'] = len(all_ents)
	config['numRelations'] = len(all_rels)
	with open(os.path.join(args.name, 'config.json'), 'w') as f:
		json.dump(config, f, indent=2)
else:
	#TODO load config into config variable
	with open(os.path.join(args.name, 'config.json'), 'r') as f:
		config = json.load(f)


ent_pretrained, rel_pretrained = None, None
if args.pretrained:
	# load pretrained weights from transe checkpoint 
	print("loading pretrained weights")
	model = TransE(len(rel2id), len(ent2id), dim=config['embedding_dim'], norm=config['norm'])
	model.load_state_dict(torch.load(args.pretrained))

	ent_pretrained = model.ents.weight.data
	rel_pretrained = model.rels.weight.data


# =========================================
# Initialize MODEL
# =========================================
if args.model == 'transE':
	model = TransE(len(rel2id), len(ent2id), dim=config['embedding_dim'], norm=config['norm'], margin=config['margin'], l2reg=config['l2reg'])
if args.model == 'transH':
	model = TransH(len(rel2id), len(ent2id), dim=config['embedding_dim'], norm=config['norm'], margin=config['margin'], l2reg=config['l2reg'])
elif args.model == 'subjD':
	model = SubjKB_Deviation(len(rel2id), len(ent2id), len(src2id), dim=config['embedding_dim'], norm=config['norm'], margin=config['margin'], l2reg=config['l2reg'], relPretrained=rel_pretrained, entPretrained=ent_pretrained)
elif args.model == 'subjM':
	model = SubjKB_Matrix(len(rel2id), len(ent2id), len(src2id), dim=config['embedding_dim'], norm=config['norm'],  nonlinearity='tanh')
elif args.model == 'ff':
	model = FeedForward(len(rel2id), len(ent2id), dim=config['embedding_dim'])
elif args.model == 'ffs':
	model = FeedForward_Source(len(rel2id), len(ent2id), len(src2id), dim=config['embedding_dim'])
elif args.model == 'hyte':
	model = HyTE(len(rel2id), len(ent2id), len(src2id), dim=config['embedding_dim'], norm=config['norm'], margin=config['margin'], l2reg=config['l2reg'])

# model.to(device)

# Logger
if args.mode.startswith('train'):
	logger = Logger(config['name'], ['loss', 'val_loss', 'MR', 'MRR', 'h@10'])
else:
	logger = None

# Loss function
criterion = MarginRankingLoss(config['margin'], reduction='sum')

# Batch loader
loader = BatchLoader(train, bernoulli_p, goldens, all_ents, all_sources, batch_size=config['batch_size'], neg_ratio=config['neg_ratio'])

# =========================================
# Initialize OPTIMIZER
# =========================================
if config['optim']== 'adam':
	optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['l2reg'])
elif config['optim'] == 'adagrad':
	optimizer = optim.Adagrad(model.parameters(), lr=config['learning_rate'], weight_decay=config['l2reg'])
else:
	optimizer = optim.SGD(model.parameters(), lr=config['learning_rate'], weight_decay=config['l2reg'])

trainer = Trainer(model, train, val, test, optimizer, criterion, logger, loader, config)

if args.mode.startswith('train'):

	# =========================================
	# TRAIN
	# =========================================
	trainer.train(config['num_epochs'], patience=config['patience'])

	if args.mode=='train-val':
		# # VALIDATE HEADS
		# print("validating heads prediction task")
		# outfile = os.path.join(config['name'], 'val_heads.tsv')
		# val_heads = loader.get_pred_heads(val, train, len(ent2id))
		# trainer.eval_filt(val_heads, outfile)

		# # VALIDATE TAILS
		# print("validating tails prediction task")
		# val_tails = loader.get_pred_tails(val, train, len(ent2id))	
		# outfile = os.path.join(config['name'], 'val_tails.tsv')
		# trainer.eval_filt(val_tails, outfile)

		# VALIDATE HEADS
		print("validating heads prediction task")
		outfile = os.path.join(config['name'], 'val_heads.tsv')
		trainer.eval(loader.get_eval_samples(val, train, len(ent2id), 'head'), len(val), outfile)
		print()

		# VALIDATE TAILS
		print("validating tails prediction task")
		outfile = os.path.join(config['name'], 'val_tails.tsv')
		trainer.eval(loader.get_eval_samples(val, train, len(ent2id), 'tail'), len(val), outfile)
		print()


	
elif args.mode == 'val':
	
	# VALIDATE HEADS
	print("validating heads prediction task")
	outfile = os.path.join(config['name'], 'val_heads.tsv')
	trainer.eval(loader.get_eval_samples(val, train, len(ent2id), 'head'), len(val), outfile)
	print()

	# VALIDATE TAILS
	print("validating tails prediction task")
	outfile = os.path.join(config['name'], 'val_tails.tsv')
	trainer.eval(loader.get_eval_samples(val, train, len(ent2id), 'tail'), len(val), outfile)
	print()
	

elif args.mode == 'val-old':

	print("Loading eval samples from pickle file")
	eval_dict = pickle.load(open(args.val_pkl,"rb"))
	# eval_dict = {
	# 	(1,1,1,1):[(1,1,1,1), (2,2,2,1)],
	# 	(1,1,1,0):[(1,1,1,1), (2,2,2,0)],
	# 	(1,1,1,0):[(3,1,1,1), (12,10,2,0)],
	# 	(1,1,1,0):[(1,14,3,1), (2,4,2,1)],
	# 	(1,1,1,0):[(8,7,2,1), (10,2,2,1)]
	# 	}

	if args.eval_result is not None:
		outfile = args.eval_result
	else:
		outfile = os.path.join(config['name'], 'val_eval.tsv')

	trainer.eval_filt(eval_dict, outfile)

elif args.mode == 'test':
	
	# VALIDATE HEADS
	print("evaluating heads prediction task")
	outfile = os.path.join(config['name'], 'test_heads.tsv')
	trainer.eval(loader.get_eval_samples(test, train, len(ent2id), 'head'), len(test), outfile)
	print()

	# VALIDATE TAILS
	print("evaluating tails prediction task")
	outfile = os.path.join(config['name'], 'test_tails.tsv')
	trainer.eval(loader.get_eval_samples(test, train, len(ent2id), 'tail'), len(test), outfile)
	print()
	






