import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import LongTensor, FloatTensor
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

torch.autograd.set_detect_anomaly(True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TransE(nn.Module):
	def __init__(self, nRels, nEnts, dim=100, norm=2, relPretrained=None, entPretrained=None):
		super(TransE, self).__init__()

		self.nRels = nRels + 1
		self.nEnts = nEnts + 1
		self.dim = dim
		self.norm = norm

		self.relPretrained = relPretrained
		self.entPretrained = entPretrained

		self.rels = nn.Embedding(nRels, self.dim)
		self.ents = nn.Embedding(nEnts, self.dim)

		if self.relPretrained is not None:
			self.rels.weight.data.copy_(torch.from_numpy(self.relPretrained))

		if self.entPretrained is not None:
			self.ents.weight.data.copy_(torch.from_numpy(self.entPretrained))


	def normalize_weights(self):
		 # normalize embeddings
	    self.ents.weight.data = F.normalize(self.ents.weight.data, self.norm, -1)
	    self.rels.weight.data = F.normalize(self.rels.weight.data, self.norm, -1)


	def fwd(self, all_heads, all_rels, all_tails, all_sources):
		## 1. Convert input sequences into vector representations
		heads_vec = self.ents(LongTensor(all_heads).to(device))
		rels_vec = self.rels(LongTensor(all_rels).to(device))
		tails_vec = self.ents(LongTensor(all_tails).to(device))

		scores = torch.norm(heads_vec + rels_vec - tails_vec, self.norm, 1)
		return scores

	def forward(self, heads, rels, tails, sources, heads_bad, rels_bad, tails_bad, sources_bad):
		self.normalize_weights()

		all_heads = np.concatenate((heads,heads_bad))
		all_rels = np.concatenate((rels,rels_bad))
		all_tails = np.concatenate((tails,tails_bad))
		all_sources = np.concatenate((sources,sources_bad))

		scores = self.fwd(all_heads,all_rels,all_tails,all_sources)
		scores = scores.view(2,len(heads))

		return scores[0], scores[1]

	def get_metrics(self, scores, true_idx):
		f_true = scores[true_idx]
		scores = scores[1:]

		rank = sum(scores<f_true)+1
		rrank = 1/(rank)

		return rank, rrank, int(rank <= 10)

	def eval_filt(self, samples, samples_dict):
		"""
		Calculate filtered scores
		- requires a samples_dict input of 
		a list candidate triples that do not appear in original training set for each test sample
		"""
		self.normalize_weights()

		nSamples = len(samples_dict)
		MR, MRR, h10 = 0, 0, 0

		i = 0
		for h,r,t,s in samples_dict:
			i += 1
			print("run validation ... {:.2%}".format(i/nSamples), end="\r")

			candidates = samples_dict[(h,r,t,s)]
			# prepend the test sample
			candidates = np.insert(candidates, 0, [h,r,t,s], axis=0)
			hh = candidates[:,0]
			rr = candidates[:,1]
			tt = candidates[:,2]

			hh = self.ents(LongTensor(hh).to(device))
			rr = self.rels(LongTensor(rr).to(device))
			tt = self.ents(LongTensor(tt).to(device))


			scores = torch.norm(hh + rr - tt, self.norm, 1).detach().cpu().numpy()
			rank, rrank, h10 = self.get_metrics(scores, 0)

			MR += rank/nSamples
			MRR += rrank/nSamples
			h10 += int(rank <10)/nSamples

		return MR, MRR, h10

	def eval_raw(self, samples):
		"""
		Calculates 'raw' scores
		- does not filter out corrupt candidates that appear in the training data
		"""

		self.normalize_weights()

		N = len(samples)

		heads = {'MR':0.0, 'MRR':0.0, 'h10':0.0}
		tails = {'MR':0.0, 'MRR':0.0, 'h10':0.0}
		i = 0
		for h,r,t,s in samples:
			i += 1
			print("fast validation ... {:.2%}".format(i/N), end="\r")

			# relation, source is constant
			rr = np.repeat(r, self.nEnts)
			ss = np.repeat(s, self.nEnts)

			# REPLACE HEADS
			######################################
			hh = np.arange(self.nEnts) # 'true' candidate is at index h
			tt = np.repeat(t, self.nEnts)

			scores = self.fwd(hh,rr,tt,ss).detach().cpu().numpy()
			rank, rrank, h10 = self.get_metrics(scores, h)

			heads['MR'] += rank/N
			heads['MRR'] += rrank/N
			heads['h10'] += h10/N


			# REPLACE TAILS
			######################################
			hh = np.repeat(h, self.nEnts)
			tt = np.arange(self.nEnts) # 'true' candidate is at index t

			scores = self.fwd(hh,rr,tt,ss).detach().cpu().numpy()
			rank, rrank, h10 = self.get_metrics(scores, h)

			tails['MR'] += rank/N
			tails['MRR'] += rrank/N
			tails['h10'] += h10/N

		MR = (heads['MR'] + tails['MR']) / 2
		MRR = (heads['MRR'] + tails['MRR']) / 2
		h10 = (heads['h10'] + tails['h10']) / 2

		return MR, MRR, h10


class SubjKB_Deviation(TransE):
	"""
	TransE with Source deviation vectors
	"""
	def __init__(self, nRels, nEnts, nSrc, dim=100, norm=2, relPretrained=None, entPretrained=None):
		super(SubjKB_Deviation, self).__init__(nRels, nEnts, dim, norm, relPretrained, entPretrained)

		self.nSrc = nSrc+1 # 0=padding, 1=UNK because of the way data is loaded

		# initialize 'add-on' deviation embeddings for each source
		self.srcEnts = nn.ModuleList()
		self.srcRels = nn.ModuleList()
		
		for srcId in range(2, nSrc):

			rels = nn.Embedding(self.nRels, self.dim)
			ents = nn.Embedding(self.nEnts, self.dim)

			# pretrained vectors?
			if self.relPretrained is not None:
				rels.weight.data.copy_(torch.from_numpy(self.relPretrained))

			if self.entPretrained is not None:
				ents.weight.data.copy_(torch.from_numpy(self.entPretrained))

			self.srcEnts.append(ents.to(device))
			self.srcRels.append(rels.to(device))

	def normalize_weights(self):
		 # normalize embeddings
	    self.ents.weight.data = F.normalize(self.ents.weight.data, self.norm, -1)
	    self.rels.weight.data = F.normalize(self.rels.weight.data, self.norm, -1)

	    # TODO: normalize addons
	    for srcId in range(2, self.nSrc):
	    	self.srcEnts[srcId-2].weight.data =  F.normalize(self.srcEnts[srcId-2].weight.data, self.norm, -1)
	    	self.srcRels[srcId-2].weight.data =  F.normalize(self.srcRels[srcId-2].weight.data, self.norm, -1)


	def fwd(self, all_heads, all_rels, all_tails, all_sources):
	    
	    ## 1. Convert input sequences into vector representations
	    heads_vec = self.ents(LongTensor(all_heads).to(device))
	    rels_vec = self.rels(LongTensor(all_rels).to(device))
	    tails_vec = self.ents(LongTensor(all_tails).to(device))

	    ## 2. Add source-specific vectors
	    for srcId in range(2, self.nSrc):

    		mask = all_sources == srcId
    		if sum(mask) < 1:
    			continue

    		# apply source mask vector to all heads, rels, tails
    		head_ids = LongTensor(all_heads * mask).to(device)
    		rel_ids = LongTensor(all_rels * mask).to(device)
    		tail_ids = LongTensor(all_tails * mask).to(device)

    		# add nonzero vectors only to rows where source == srcId
    		heads_vec += self.srcEnts[srcId-2](head_ids) 
    		rels_vec += self.srcRels[srcId-2](rel_ids)
    		tails_vec += self.srcEnts[srcId-2](tail_ids)

    	# normalize the summed vectors
	    heads_vec = F.normalize(heads_vec, self.norm, -1)
	    rels_vec = F.normalize(rels_vec, self.norm, -1)
	    tails_vec = F.normalize(tails_vec, self.norm, -1)

	    scores = torch.norm(heads_vec + rels_vec - tails_vec, self.norm, 1)
	    return scores


class SubjKB_Matrix(TransE):
	"""
	Matrix model - same transformation for entities and relations
	"""
	def __init__(self, nRels, nEnts, nSrc, dim=100, norm=2, dropout=0.05, nonlinearity=None):
		super(SubjKB_Matrix, self).__init__(nRels, nEnts, dim, norm)

		self.nSrc = nSrc+1
		self.dropout = dropout	

		# initialize transform matrix
		self.sourceNN = nn.ModuleList()
		
		for srcId in range(2, nSrc):

			# forward layer + dropout
			seq = nn.Sequential(
				nn.Linear(self.dim, self.dim),
				nn.Dropout(self.dropout)
				).to(device)

			# add nonlinearity
			if nonlinearity == 'tanh':
				seq.add_module('nonlinearity', nn.Tanh())
			elif nonlinearity == 'relu':
				seq.add_module('nonlinearity', nn.ReLU())

			self.sourceNN.append(seq)


	def normalize_weights(self):
		pass

	def fwd(self, all_heads, all_rels, all_tails, all_sources):
	    
	    h = LongTensor(all_heads).to(device)
	    r = LongTensor(all_rels).to(device)
	    t = LongTensor(all_tails).to(device)

	    heads_vec = torch.zeros([len(all_heads), self.dim]).to(device)
	    rels_vec = torch.zeros([len(all_rels), self.dim]).to(device)
	    tails_vec = torch.zeros([len(all_tails), self.dim]).to(device)

	    ## It is faster to feed each entity,rel vector through 
	    ## each source feed-forward layer. Then mask out rows that belong to other srcsIds
	    for srcId in range(2, self.nSrc):
	    	mask = all_sources == srcId
	    	if sum(mask) < 1:
	    		continue
	    	mask_tensor = FloatTensor(mask).view(len(all_sources),1).to(device)

	    	h_hidden = self.sourceNN[srcId-2](self.ents(h))
	    	r_hidden = self.sourceNN[srcId-2](self.rels(r))
	    	t_hidden = self.sourceNN[srcId-2](self.ents(t))

	    	heads_vec += h_hidden*mask_tensor
	    	rels_vec += r_hidden*mask_tensor
	    	tails_vec += t_hidden*mask_tensor

    	# Dont need to normalize because using a TanH activation function already
    	# Do it anyway???
	    heads_vec = F.normalize(heads_vec, self.norm, -1)
	    rels_vec = F.normalize(rels_vec, self.norm, -1)
	    tails_vec = F.normalize(tails_vec, self.norm, -1)

	    scores = torch.norm(heads_vec + rels_vec - tails_vec, self.norm, 1)
	    return scores