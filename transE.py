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
	def __init__(self, nRels, nEnts, dim=100, norm=2):
		super(TransE, self).__init__()

		self.nRels = nRels
		self.nEnts = nEnts
		self.dim = dim
		self.norm = norm

		self.rels = nn.Embedding(nRels+1, self.dim)
		self.ents = nn.Embedding(nEnts+1, self.dim)


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
	    n = len(heads)

	    self.normalize_weights()

	    all_heads = np.concatenate((heads,heads_bad))
	    all_rels = np.concatenate((rels,rels_bad))
	    all_tails = np.concatenate((tails,tails_bad))
	    all_sources = np.concatenate((sources,sources_bad))

	    ## 1. Convert input sequences into vector representations
	    heads_vec = self.ents(LongTensor(all_heads).to(device))
	    rels_vec = self.rels(LongTensor(all_rels).to(device))
	    tails_vec = self.ents(LongTensor(all_tails).to(device))

	    scores = torch.norm(heads_vec + rels_vec - tails_vec, self.norm, 1)
	    scores = scores.view(2,n)

	    # return positive and negative scores 
	    return scores[0], scores[1]

	def validate(self, samples, samples_dict, all_ents, all_rels, all_sources):
		"""
		Calculate filtered scores
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
			f_true = scores[0]
			scores = scores[1:]

			rank = sum(scores<f_true)
			rrank = 1/(rank+1)

			MR += rank/nSamples
			MRR += rrank/nSamples
			h10 += int(rank <10)/nSamples

		return MR, MRR, h10

	def score(self, i, hh, rr, tt):
		"""
		i: index of 'true' triple
		hh,rr,tt: all heads, rels, tails
		"""
		# print(hh.size())
		# print(rr.size())
		# print(tt.size())

		scores = torch.norm(hh + rr - tt, self.norm, 1).detach().cpu().numpy()
		f_true = scores[i]

		rank = sum(scores<f_true)+1
		rrank = 1/rank

		return rank, rrank, int(rank < 11)

	def eval_raw(self, samples):
		"""
		Calculates 'raw' scores
		- does not filter out corrupt candidates that appear in the training data
		
		REQUIRES A LOT OF MEMORY
		"""

		self.normalize_weights()

		N = len(samples)
		batch_size = 256

		heads = {'MR':0.0, 'MRR':0.0, 'h10':0.0}
		tails = {'MR':0.0, 'MRR':0.0, 'h10':0.0}
		i = 0
		for h,r,t,s in samples:
			i += 1
			print("fast validation ... {:.2%}".format(i/N), end="\r")

			# relation is constant
			rr = self.rels(LongTensor([r]).to(device)).repeat(self.nEnts, 1)
			hh = self.ents(LongTensor([h]).to(device)).repeat(self.nEnts, 1)
			tt = self.ents(LongTensor([t]).to(device)).repeat(self.nEnts, 1)


			# Replace Heads
			head_candidates = np.arange(self.nEnts) # 'true' candidate is at index h
			head_candidates = self.ents(LongTensor(head_candidates).to(device))

			rank, rrank, h10 = self.score(h, head_candidates, rr, tt)
			heads['MR'] += rank/N
			heads['MRR'] += rrank/N
			heads['h10'] += h10/N

			head_candidates.detach()


			# Replace Tails
			tail_candidates = np.arange(self.nEnts) # 'true' candidate is at index t
			tail_candidates = self.ents(LongTensor(tail_candidates).to(device))

			rank, rrank, h10 = self.score(t, hh, rr, tail_candidates)
			tails['MR'] += rank/N
			tails['MRR'] += rrank/N
			tails['h10'] += h10/N

			tail_candidates.detach()

			rr.detach()
			hh.detach()
			tt.detach()
			

			# if i > 3:
			# 	break


		MR = (heads['MR'] + tails['MR']) / 2
		MRR = (heads['MRR'] + tails['MRR']) / 2
		h10 = (heads['h10'] + tails['h10']) / 2

		return MR, MRR, h10


		# return tails['MR'], tails['MRR'], tails['h10']

class TransE_SourceMatrix(nn.Module):
	"""
	TransE with Source
	"""
	def __init__(self, nRels, nEnts, nSrc, dim=100, norm=2):
		super(TransE_SourceMatrix, self).__init__()

		self.nRels = nRels
		self.nEnts = nEnts
		self.nSrc = nSrc
		self.dim = dim
		self.norm = norm
		self.use_source_info = True 

		self.ents = nn.Embedding(nEnts+1, self.dim, padding_idx=0)
		self.rels = nn.Embedding(nRels+1, self.dim, padding_idx=0)

		# initialize 'add-on' embeddings for each source
		self.entNN = nn.ModuleList()
		self.relNN = nn.ModuleList()
		
		for srcId in range(2, nSrc+1):
			# self.entNN.append(nn.Linear(self.dim, self.dim).to(device))
			# self.relNN.append(nn.Linear(self.dim, self.dim).to(device))

			self.entNN.append(nn.Sequential(
				nn.Linear(self.dim, self.dim),
				nn.Dropout(0.05),
				nn.Tanh()
				).to(device))

			self.relNN.append(nn.Sequential(
				nn.Linear(dim, dim),
				nn.Dropout(0.05),
				nn.Tanh()
				).to(device))


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
	    for srcId in range(2, self.nSrc+1):
	    	mask = all_sources == srcId
	    	if sum(mask) < 1:
	    		continue
	    	mask_tensor = FloatTensor(mask).view(len(all_sources),1).to(device)

	    	h_hidden = self.entNN[srcId-2](self.ents(h))
	    	r_hidden = self.relNN[srcId-2](self.rels(r))
	    	t_hidden = self.entNN[srcId-2](self.ents(t))

	    	heads_vec += h_hidden*mask_tensor
	    	rels_vec += r_hidden*mask_tensor
	    	tails_vec += t_hidden*mask_tensor

    	# Dont need to normalize because using a TanH activation function already
    	# Do it anyway???
	    heads_vec = F.normalize(heads_vec, self.norm, -1)
	    rels_vec = F.normalize(rels_vec, self.norm, -1)
	    tails_vec = F.normalize(tails_vec, self.norm, -1)

	    scores = torch.norm(heads_vec + rels_vec - tails_vec, self.norm, 1)
	    # scores = scores.view(2,n)

	    # return positive and negative scores 
	    return scores

	def forward(self, heads, rels, tails, sources, heads_bad, rels_bad, tails_bad, sources_bad):
	    

	    # self.normalize_weights()

	    all_heads = np.concatenate((heads,heads_bad))
	    all_rels = np.concatenate((rels,rels_bad))
	    all_tails = np.concatenate((tails,tails_bad))
	    all_sources = np.concatenate((sources,sources_bad))

	    scores = self.fwd(all_heads,all_rels,all_tails,all_sources)

	    n = len(heads)
	    scores = scores.view(2,n)

	    # return positive and negative scores 
	    return scores[0], scores[1]

	def validate_fast(self, samples, samples_dict, all_ents, all_rels, all_sources):
		self.normalize_weights()

		nSamples = float(len(samples_dict))
		MR, MRR, h10 = 0.0, 0.0, 0.0

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
			ss = candidates[:,3] ## THIS WAS 2 BEFORE

			scores = self.fwd(hh,rr,tt,ss).detach().cpu().numpy()
			
			f_true = scores[0]
			scores = scores[1:]

			rank = sum(scores<f_true)
			rrank = 1/(rank+1)

			MR += rank/nSamples
			MRR += rrank/nSamples
			h10 += int(rank <10)/nSamples

		return MR, MRR, h10

class TransE_SourceFull(nn.Module):
	"""
	TransE with Source
	"""
	def __init__(self, nRels, nEnts, nSrc, dim=100, norm=2):
		super(TransE_SourceFull, self).__init__()

		self.nRels = nRels
		self.nEnts = nEnts
		self.nSrc = nSrc
		self.dim = dim
		self.norm = norm
		self.use_source_info = True 

		self.rels = nn.Embedding(nRels+1, self.dim, padding_idx=0)
		self.ents = nn.Embedding(nEnts+1, self.dim, padding_idx=0)

		# initialize 'add-on' embeddings for each source
		self.srcEnts = nn.ModuleList()
		self.srcRels = nn.ModuleList()
		
		for srcId in range(2, nSrc+1):
			self.srcEnts.append(nn.Embedding(nEnts+1, self.dim, padding_idx=0).to(device))
			self.srcRels.append(nn.Embedding(nRels+1, self.dim, padding_idx=0).to(device))

			# TODO start them off as the same?
			# IDEA: INITIALIZE WITH pretrained baseline vectors?

		# self.source_emb = {}
		# self.source_emb = {}
		# for srcId in range(2, nSrc+1):
		# 	self.source_emb[srcId] = {
		# 		'rels': nn.Embedding(nRels+1, self.dim, padding_idx=0).to(device),
		# 		'ents': nn.Embedding(nEnts+1, self.dim, padding_idx=0).to(device)
		# 	}

	def normalize_weights(self):
		 # normalize embeddings
	    self.ents.weight.data = F.normalize(self.ents.weight.data, self.norm, -1)
	    self.rels.weight.data = F.normalize(self.rels.weight.data, self.norm, -1)

	    # TODO: normalize addons
	    for srcId in range(2, self.nSrc+1):
	    	self.srcEnts[srcId-2].weight.data =  F.normalize(self.srcEnts[srcId-2].weight.data, self.norm, -1)
	    	self.srcRels[srcId-2].weight.data =  F.normalize(self.srcRels[srcId-2].weight.data, self.norm, -1)

	def get_regularizer(self):
		reg = 0
		for srcId in range(2, self.nSrc+1):
			reg += torch.sum(torch.norm(self.source_emb[srcId]['ents'].weight.data, 2,-1))
			reg += torch.sum(torch.norm(self.source_emb[srcId]['rels'].weight.data, 2,-1))
		return reg


	def fwd(self, all_heads, all_rels, all_tails, all_sources):
	    
	    ## 1. Convert input sequences into vector representations
	    heads_vec = self.ents(LongTensor(all_heads).to(device))
	    rels_vec = self.rels(LongTensor(all_rels).to(device))
	    tails_vec = self.ents(LongTensor(all_tails).to(device))

	    ## 2. Add source-specific vectors
	    for srcId in range(2, self.nSrc+1):

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
	    # scores = scores.view(2,n)

	    # return positive and negative scores 
	    return scores

	def forward(self, heads, rels, tails, sources, heads_bad, rels_bad, tails_bad, sources_bad):
	    

	    self.normalize_weights()

	    all_heads = np.concatenate((heads,heads_bad))
	    all_rels = np.concatenate((rels,rels_bad))
	    all_tails = np.concatenate((tails,tails_bad))
	    all_sources = np.concatenate((sources,sources_bad))

	    scores = self.fwd(all_heads,all_rels,all_tails,all_sources)

	    n = len(heads)
	    scores = scores.view(2,n)

	    # return positive and negative scores 
	    return scores[0], scores[1]

	def validate_fast(self, samples, samples_dict, all_ents, all_rels, all_sources):
		self.normalize_weights()

		nSamples = float(len(samples_dict))
		MR, MRR, h10 = 0.0, 0.0, 0.0

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
			ss = candidates[:,3] ## ERRORRRRR was candidates[:,2]

			scores = self.fwd(hh,rr,tt,ss).detach().cpu().numpy()
			
			f_true = scores[0]
			scores = scores[1:]

			rank = sum(scores<f_true)
			rrank = 1/(rank+1)

			MR += rank/nSamples
			MRR += rrank/nSamples
			h10 += int(rank <10)/nSamples

		return MR, MRR, h10


	def validate_fast_1(self, samples, samples_dict, all_ents, all_rels, all_sources):
		
		self.normalize_weights()

		nSamples = len(samples_dict)
		MR, MRR, h10 = 0.0, 0.0, 0.0

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

			# NOTE: Source is always the same, 
			# so can query self.source_emb[s] directly
			hh = (self.ents(LongTensor(hh).to(device)) + self.source_emb[s]['ents'](LongTensor(hh).to(device)) )
			rr = (self.ents(LongTensor(rr).to(device)) + self.source_emb[s]['rels'](LongTensor(rr).to(device)) )
			tt = (self.ents(LongTensor(tt).to(device)) + self.source_emb[s]['ents'](LongTensor(tt).to(device)) )


			scores = torch.norm(hh + rr - tt, self.norm, 1).detach().cpu().numpy()
			f_true = scores[0]
			scores = scores[1:]

			rank = float(sum(scores<f_true))
			rrank = 1/(rank+1)

			MR += rank/nSamples
			MRR += rrank/nSamples
			h10 += int(rank <10)/float(nSamples)

		return MR, MRR, h10


