"""
Data loader for TransE

@mashabelyi
"""

import os, json
from collections import defaultdict
import numpy as np


_UNK_ = 'UNK'

def load_triples(fpath):
	"""
	Load data from file.
	Each line contains at least 4 columns:
	head | relation | tail | sourceAttribute
	"""
	data = []
	with open(fpath, 'r') as f:
		for line in f:
			h,r,t,s = line.strip().split('\t')
			data.append((h,r,t,s))
	return data


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

def build_data(path='data'):

	train = load_triples(os.path.join(path, 'train.tsv'))
	val = load_triples(os.path.join(path, 'val.tsv'))
	test = load_triples(os.path.join(path, 'test.tsv'))

	rel2id, ent2id, src2id = vectorize(train)

	# save maps for later 
	with open(os.path.join(path, 'rel2id.json'), 'w') as outfile:
		json.dump(rel2id, outfile)

	with open(os.path.join(path, 'ent2id.json'), 'w') as outfile:
		json.dump(ent2id, outfile)

	with open(os.path.join(path, 'src2id.json'), 'w') as outfile:
		json.dump(src2id, outfile)

	train = [(get_tok_id(h, ent2id), get_tok_id(r, rel2id), get_tok_id(t, ent2id), get_tok_id(s, src2id)) for h,r,t,s in train]
	val = [(get_tok_id(h, ent2id), get_tok_id(r, rel2id), get_tok_id(t, ent2id), get_tok_id(s, src2id)) for h,r,t,s in val]
	test = [(get_tok_id(h, ent2id), get_tok_id(r, rel2id), get_tok_id(t, ent2id), get_tok_id(s, src2id)) for h,r,t,s in test]

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
	bernoulli_p = {1:0.5} # UNK token has equal probability of replacement
	for r in all_rels:
	    bernoulli_p[r] = tph[r]/(tph[r] + hpt[r])
	
	all_rels.add(1)
	all_ents.add(1)    

	return (train, val, test, bernoulli_p, 
			goldens, all_ents, all_rels, 
			rel2id, ent2id, src2id)

