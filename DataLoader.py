"""
Data loader for TransE

@mashabelyi
"""

import os, json
from collections import defaultdict
import numpy as np


_UNK_ = 'UNK'

def load_triple_ids(fpath):
	"""
	Load data from file.
	Each line contains at least 4 columns:
	head | tail | relaton
	"""
	res = []
	with open(fpath, 'r') as f:
		for line in f:
			data = line.strip().split()
			if len(data) == 1:
				continue
			h,t,r = data
			res.append((int(h),int(r),int(t),1)) # append 1 for source for now to make it work with the rest of the code
	return res

def load_ids(fpath):
	res = {}
	with open(fpath, 'r') as f:
		for line in f:
			data = line.strip().split()
			if len(data) == 1:
				continue
			else:
				res[data[0]] = int(data[1])
	return res

def vectorize(triples):
	"""
	create relation and entity id lookup tables
	"""

	rel2id, ent2id, src2id = {_UNK_:1},{_UNK_:1},{_UNK_:1}
	nRel, nEnt, nSrc = 1, 1, 1

	for h,r,t,s in triples:
			
		if h not in ent2id:
			nEnt += 1
			ent2id[h] = nEnt
		
		if r not in rel2id:
			nRel += 1
			rel2id[r] = nRel
		
		if t not in ent2id:
			nEnt += 1
			ent2id[t] = nEnt

		if s not in src2id:
			nSrc += 1
			src2id[s] = nSrc

	return rel2id, ent2id, src2id

def get_tok_id(tok, tok2id):
	if tok not in tok2id:
		return tok2id[_UNK_]
	else:
		return tok2id[tok]

def OpenKELoader(path='data'):

	train = load_triple_ids(os.path.join(path, 'train2id.txt'))
	val = load_triple_ids(os.path.join(path, 'valid2id.txt'))
	test = load_triple_ids(os.path.join(path, 'test2id.txt'))

	rel2id = load_ids(os.path.join(path, 'relation2id.txt'))
	ent2id = load_ids(os.path.join(path, 'entity2id.txt'))
	src2id = {}

	# calculate probabilities for Bernoulli trick (Wang et al.,2014; Lin et al., 2015b)
	#
	# NOTE: in SubjKB most relations are 1-to-many (there are a lot more tails than heads),
	# so we will be replacing heads with high probability

	head_rel = defaultdict(set)
	rel_tail = defaultdict(set)
	goldens = set()
	all_rels = set()
	all_ents = set()
	for h,r,t,s in train:
	    # h = list2tup(h)
	    # r = list2tup(r)
	    # t = list2tup(t)

	    head_rel[(h,r)].add(t)
	    rel_tail[(r,t)].add(h)

	    all_rels.add(r)
	    all_ents.add(h)
	    all_ents.add(t)
	    goldens.add((h,r,t))
	    # goldens.add((h,r,t,s))

	hpt = defaultdict(list)
	tph = defaultdict(list)
	# NUmber of tails tails per head
	for hr, tails in head_rel.items():
	    h,r = hr
	    tph[r].append(len(tails))
	# NUmber of heads per tail  
	for rt, heads in rel_tail.items():
	    r,t = rt
	    hpt[r].append(len(heads)) 

	# average number of head/tail per tail/head
	hpt = {r:np.mean(hpt[r]) for r in hpt}
	tph = {r:np.mean(tph[r]) for r in tph}
	# Calculate probability of replacing the head
	bernoulli_p = {}
	for r in all_rels:
	    bernoulli_p[r] = tph[r]/(tph[r] + hpt[r])
	 

	return (train, val, test, bernoulli_p, goldens, 
		all_ents, all_rels, rel2id, ent2id, src2id)

