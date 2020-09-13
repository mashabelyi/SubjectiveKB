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
	"""
	TransE Implementation
	Bordes et al 2013
	"""
	def __init__(self, nRels, nEnts, dim=100, norm=2, margin=1, l2reg=0.1, relPretrained=None, entPretrained=None):
		super(TransE, self).__init__()

		self.nRels = nRels # nRels is the total count including PAD and UNK tokens
		self.nEnts = nEnts # nEnts is the total count including PAD and UNK tokens
		self.dim = dim
		self.norm = norm
		self.margin = margin
		self.reg_weight = l2reg

		self.relPretrained = relPretrained
		self.entPretrained = entPretrained

		self.ebatch = None
		self.rbatch = None

		# NOTE: it is important to use padding for compatibility with 
		# subjective models
		self.rels = nn.Embedding(nRels, self.dim, padding_idx=0)
		self.ents = nn.Embedding(nEnts, self.dim, padding_idx=0)

		if self.relPretrained is not None:
			self.rels.weight.data.copy_(self.relPretrained)

		if self.entPretrained is not None:
			self.ents.weight.data.copy_(self.entPretrained)


	def normalize_weights(self):
		pass

		# # normalize embeddings
		# self.ents.weight.data = F.normalize(self.ents.weight.data, self.norm, -1)
		# self.rels.weight.data = F.normalize(self.rels.weight.data, self.norm, -1)

		# if self.ebatch != None:
		# 	self.ents.weight.data[self.ebatch] = F.normalize(self.ents.weight.data[self.ebatch], self.norm, -1)
		# 	self.rels.weight.data[self.rbatch] = F.normalize(self.rels.weight.data[self.rbatch], self.norm, -1)
		# else:
		# 	self.ents.weight.data = F.normalize(self.ents.weight.data, self.norm, -1)
		# 	self.rels.weight.data = F.normalize(self.rels.weight.data, self.norm, -1)


	def fwd(self, all_heads, all_rels, all_tails, all_sources):
		## 1. Convert input sequences into vector representations
		heads_vec = self.ents(LongTensor(all_heads).to(device))
		rels_vec = self.rels(LongTensor(all_rels).to(device))
		tails_vec = self.ents(LongTensor(all_tails).to(device))

		scores = torch.norm(heads_vec + rels_vec - tails_vec, self.norm, 1)
		return scores

	def forward(self, heads, rels, tails, sources, heads_bad, rels_bad, tails_bad, sources_bad):
		# self.normalize_weights()

		all_heads = np.concatenate((heads,heads_bad))
		all_rels = np.concatenate((rels,rels_bad))
		all_tails = np.concatenate((tails,tails_bad))
		all_sources = np.concatenate((sources,sources_bad))

		# store which h,r,ts are used in this batch
		# self.ebatch = list(set(np.concatenate((all_heads,all_tails))))
		# self.rbatch = list(set(all_rels))

		scores = self.fwd(all_heads,all_rels,all_tails,all_sources)
		scores = scores.view(2,len(heads))

		# print(scores[0], scores[1])
		return self.loss(scores[0], scores[1])

	
	def loss(self, gold_scores, corrupt_scores):
		zeros = torch.tensor([0], dtype=torch.float).to(device)
		margin_tensor = torch.tensor([self.margin], dtype=torch.float).to(device)
		rank_loss = torch.sum(torch.max(margin_tensor + gold_scores - corrupt_scores, zeros)).to(device)
		
		# regularize l2 norm of all entities to be <= 1
		# so penalize any entities with norm > 1
		regularization = self.reg_weight * torch.sum( 
			torch.max( 
				torch.norm(self.ents.weight, 2, 1) - torch.tensor([1], dtype=torch.float).to(device),
				torch.tensor([0], dtype=torch.float).to(device),
			)
			).to(device)

		# 4. Also regularize the scores of all golden tuples to be close to zero
		regularization += self.reg_weight*torch.sum(gold_scores)

		return rank_loss + regularization



	def norm_step(self):
		pass

		# normalize the source vectors to have norm_2 == 1
		# self.src.weight =  F.normalize(self.src.weight, 2, -1)
		# norm = torch.norm(self.ents.weight, p=2, dim=1, keepdim=True)
		# self.ents.weight = torch.nn.Parameter( self.ents.weight.div(norm) )

	def get_metrics(self, scores, true_idx):
		f_true = scores[true_idx]
		# scores = scores[1:]

		rank = sum(scores<f_true)+1.0
		rrank = 1.0/rank

		return rank, rrank, int(rank <= 10)

	def eval_run(self, iterator, nSamples):
		"""
		Run evaluation on samples returned from iterator
		"""
		with torch.no_grad():
			# self.ebatch = None
			# self.rbatch = None
			# self.normalize_weights()

			MR, MRR, H10, total = 0.0, 0.0, 0.0, 0.0
			ranks = []
			source_metrics = {}

			i = 0
			for hh,rr,tt,ss in iterator:
				i += 1
				h = hh[0]
				t = tt[0]
				r = rr[0]
				s = ss[0]
				# print("run validation (filt) ... {:.2%}".format(i/nSamples), end="\r")

				print("validation (filt) ... {:.2%}, MR: {:.4}, MRR: {:.4}, H@10: {:.4}".format(i/nSamples, MR, MRR, H10), end="\r")
				
				if s not in source_metrics:
					source_metrics[s] = {'total':0.0, 'MR':0.0, 'MRR':0.0, 'H10':0.0}

				scores = self.fwd(hh,rr,tt,ss).detach().cpu().numpy()
				rank, rrank, h10 = self.get_metrics(scores, 0)

				ranks.append((h,r,t,s,rank))
				MR = (MR*total + rank)/(total+1)
				MRR = (MRR*total + rrank)/(total+1)
				H10 = (H10*total + h10)/(total+1)
				total += 1

				source_metrics[s]['MR'] = (source_metrics[s]['MR']*source_metrics[s]['total'] + rank)/(source_metrics[s]['total']+1)
				source_metrics[s]['MRR'] = (source_metrics[s]['MRR']*source_metrics[s]['total'] + rrank)/(source_metrics[s]['total']+1)
				source_metrics[s]['H10'] = (source_metrics[s]['H10']*source_metrics[s]['total'] + h10)/(source_metrics[s]['total']+1)
				source_metrics[s]['total'] += 1

		return MR, MRR, H10, source_metrics, ranks

	def eval_filt(self, samples_dict):
		"""
		Calculate filtered scores
		- requires a samples_dict input of 
		a list candidate triples that do not appear in original training set for each test sample

		Return an array of ranks of the correct prediction for each sample
		"""
		with torch.no_grad():
			self.ebatch = None
			self.rbatch = None
			# self.normalize_weights()

			nSamples = len(samples_dict)
			MR, MRR, H10, total = 0.0, 0.0, 0.0, 0.0
			ranks = []
			source_metrics = {}

			i = 0
			for h,r,t,s in samples_dict:
				i += 1
				# print("run validation (filt) ... {:.2%}".format(i/nSamples), end="\r")

				print("validation (filt) ... {:.2%}, MR: {:.4}, MRR: {:.4}, H@10: {:.4}".format(i/nSamples, MR, MRR, H10), end="\r")

				if s not in source_metrics:
					source_metrics[s] = {'total':0.0, 'MR':0.0, 'MRR':0.0, 'H10':0.0}

				candidates = samples_dict[(h,r,t,s)]
				# prepend the test sample
				candidates = np.insert(candidates, 0, [h,r,t,s], axis=0)
				hh = candidates[:,0]
				rr = candidates[:,1]
				tt = candidates[:,2]
				ss = candidates[:,3]

				scores = self.fwd(hh,rr,tt,ss).detach().cpu().numpy()
				rank, rrank, h10 = self.get_metrics(scores, 0)

				ranks.append((h,r,t,s,rank))
				MR = (MR*total + rank)/(total+1)
				MRR = (MRR*total + rrank)/(total+1)
				H10 = (H10*total + h10)/(total+1)
				total += 1

				source_metrics[s]['MR'] = (source_metrics[s]['MR']*source_metrics[s]['total'] + rank)/(source_metrics[s]['total']+1)
				source_metrics[s]['MRR'] = (source_metrics[s]['MRR']*source_metrics[s]['total'] + rrank)/(source_metrics[s]['total']+1)
				source_metrics[s]['H10'] = (source_metrics[s]['H10']*source_metrics[s]['total'] + h10)/(source_metrics[s]['total']+1)
				source_metrics[s]['total'] += 1

		return MR, MRR, H10, source_metrics, ranks



	def eval_raw(self, samples):
		"""
		Calculates 'raw' scores
		- does not filter out corrupt candidates that appear in the training data
		"""

		with torch.no_grad():
			# self.normalize_weights()

			N = len(samples)

			heads = {'MR':0.0, 'MRR':0.0, 'h10':0.0}
			tails = {'MR':0.0, 'MRR':0.0, 'h10':0.0}
			i = 0
			for h,r,t,s in samples:
				i += 1
				print("run validation (raw) ... {:.2%}".format(i/N), end="\r")

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



class TransH(TransE):
	"""
	TransH Implementation
	Wang et al., 2014
	"""
	def __init__(self, nRels, nEnts, dim=100, norm=2, margin=1, l2reg=0.1):
		super(TransH, self).__init__(nRels, nEnts, dim, norm, margin, l2reg)

		# intialize relation hyperplanes
		self.relProj = nn.Embedding(self.nRels, self.dim)

	def fwd(self, all_heads, all_rels, all_tails, all_sources):
		heads_vec = self.ents(LongTensor(all_heads).to(device))
		rels_vec = self.rels(LongTensor(all_rels).to(device))
		tails_vec = self.ents(LongTensor(all_tails).to(device))

		# project heads and tails into the relation-specific hyperplane
		w = self.relProj(LongTensor(all_rels).to(device))
		N = len(all_heads)

		heads_proj = heads_vec - torch.bmm(w.unsqueeze(1),heads_vec.unsqueeze(2)).view(N,1) * w
		tails_proj = tails_vec - torch.bmm(w.unsqueeze(1),tails_vec.unsqueeze(2)).view(N,1) * w

		scores = torch.norm(heads_proj + rels_vec - tails_proj, self.norm, 1)
		return scores

	def norm_step(self):
		# normalize the projection vectors to unit norm
		norm = torch.norm(self.relProj.weight, p=2, dim=1, keepdim=True).data
		self.relProj.weight.data = self.relProj.weight.data.div(norm)

	def loss(self, gold_scores, corrupt_scores):
		
		# 1. Rank loss
		zeros = torch.tensor([0], dtype=torch.float).to(device)
		margin_tensor = torch.tensor([self.margin], dtype=torch.float).to(device)
		rank_loss = torch.sum(torch.max(margin_tensor + gold_scores - corrupt_scores, zeros)).to(device)
		
		# 2. regularize l2 norm of all entities to be <= 1
		# so penalize any entities with norm > 1
		regularization = self.reg_weight * torch.sum( 
			torch.max( 
				torch.norm(self.ents.weight, 2, -1) - torch.tensor([1], dtype=torch.float).to(device),
				torch.tensor([0], dtype=torch.float).to(device)
			)
			).to(device)

		# 3. Regularize all relation vectors to be in their hyperplanes
		N = len(self.rels.weight)-1
		wr = self.relProj.weight[1:] # ignore padding idx 0
		r = self.rels.weight[1:] # ignore padding idx 0

		r_norm_sq = torch.clamp(
			torch.bmm(r.unsqueeze(1), r.unsqueeze(2)).view(N,1), # dot product = norm^2
			min = 0.0000000001 # prevent division by zero in future operations
			)

		eps = torch.tensor([0.01], dtype=torch.float).to(device) # margin of error
		zeros = torch.tensor([0], dtype=torch.float).to(device)

		# dot product of hyperplane orthogonal vectors with the corresponding relation vectors
		# want this to be as close to 0 as possible. Normalize by dividing by the norm.
		# - taking squares because want to penalize large negative and large positive dot product values.
		tmp = (torch.bmm(wr.unsqueeze(1), r.unsqueeze(2))**2).view(N,1).div(r_norm_sq).view(N,)
		# penalize relation vectors that are not orthogonal -- i.e. tmp is to farr away from zero
		regularization += self.reg_weight * torch.sum(torch.max(tmp - eps, zeros))		

		# 4. Also regularize the scores of all golden tuples to be close to zero
		regularization += self.reg_weight*torch.sum(gold_scores)


		return rank_loss + regularization 

	def get_ent_vect(self, entId, relId):
		vec = self.ents(LongTensor([entId]).to(device))
		w = self.relProj(LongTensor([relId]).to(device))
		proj = vec - torch.bmm(w.unsqueeze(1),vec.unsqueeze(2)).view(1,1) * w

		return proj

	def get_rel_vect(self, relId):
		vec = self.rels(LongTensor([relId]).to(device))
		return vec


class HyTE(TransE):
	"""
	HyTE implementation
	https://www.aclweb.org/anthology/D18-1225.pdf

	h_s = h - w_s*(h)w_s where w_s is the normal vector to a source-specific hyperplane
	"""
	def __init__(self, nRels, nEnts, nSrc, dim=100, norm=2, margin=1, l2reg=0.1):
		super(HyTE, self).__init__(nRels, nEnts, dim, norm, margin, l2reg)
		self.nSrc = nSrc
		self.src = nn.Embedding(self.nSrc, self.dim)

		# IDIDOT! Forgot to comment this out
		# for i in range(nSrc):
		# 	self.src.weight.data[i] = LongTensor(np.zeros(dim) + i+1)


	def fwd(self, all_heads, all_rels, all_tails, all_sources):
		# 
		heads_vec = self.ents(LongTensor(all_heads).to(device))
		rels_vec = self.rels(LongTensor(all_rels).to(device))
		tails_vec = self.ents(LongTensor(all_tails).to(device))
		w = self.src(LongTensor(all_sources).to(device))
		N = len(all_heads)

		# project entities and relations into source hyperplane
		heads_proj = heads_vec - torch.bmm(w.unsqueeze(1),heads_vec.unsqueeze(2)).view(N,1) * w
		rels_proj = rels_vec - torch.bmm(w.unsqueeze(1),rels_vec.unsqueeze(2)).view(N,1) * w
		tails_proj = tails_vec - torch.bmm(w.unsqueeze(1),tails_vec.unsqueeze(2)).view(N,1) * w

		# normalize w instead of regularizing for |w| = 1
		# (get regularization working below)
		# wnorm = w.div(torch.norm(w, p=2, dim=1).view(N,1))
		# heads_proj = heads_vec - torch.bmm(wnorm.unsqueeze(1),heads_vec.unsqueeze(2)).view(N,1) * wnorm
		# rels_proj = rels_vec - torch.bmm(wnorm.unsqueeze(1),rels_vec.unsqueeze(2)).view(N,1) * wnorm
		# tails_proj = tails_vec - torch.bmm(wnorm.unsqueeze(1),tails_vec.unsqueeze(2)).view(N,1) * wnorm

		# calculate scores in the hyperplane
		scores = torch.norm(heads_proj + rels_proj - tails_proj, self.norm, 1)
		return scores


	def loss(self, gold_scores, corrupt_scores):
		zeros = torch.tensor([0], dtype=torch.float).to(device)
		margin_tensor = torch.tensor([self.margin], dtype=torch.float).to(device)
		rank_loss = torch.sum(torch.max(margin_tensor + gold_scores - corrupt_scores, zeros)).to(device)
		
		# 2. Regularize l2 norm of all entities to be <= 1
		# so penalize any entities with norm > 1
		regularization = self.reg_weight * torch.sum( 
			torch.max( 
				torch.norm(self.ents.weight, 2, -1) - torch.tensor([1], dtype=torch.float).to(device),
				torch.tensor([0], dtype=torch.float).to(device)
			)
			).to(device)

		# 3. Also regularize the scores of all golden tuples to be close to zero
		regularization += self.reg_weight*torch.sum(gold_scores)


		return rank_loss + regularization

	def norm_step(self):
		# pass
		# normalize the source vectors to have |norm|_2 == 1
		norm = torch.norm(self.src.weight, p=2, dim=1, keepdim=True).data#.detach()
		# self.src.weight = torch.nn.Parameter( self.src.weight.data.div(norm) )
		self.src.weight.data = self.src.weight.data.div(norm)

		# self.src.weight.data.copy_(F.normalize(model.src.weight, 2, -1).detach())

	def get_ent_vect(self, entId, srcId):
		vec = self.ents(LongTensor([entId]).to(device))
		w = self.src(LongTensor([srcId]).to(device))

		proj = vec - torch.bmm(w.unsqueeze(1),vec.unsqueeze(2)).view(1,1) * w

		return proj

	def get_rel_vect(self, relId, srcId):
		vec = self.rels(LongTensor([relId]).to(device))
		w = self.src(LongTensor([srcId]).to(device))

		proj = vec - torch.bmm(w.unsqueeze(1),vec.unsqueeze(2)).view(1,1) * w

		return proj





class SubjKB_Deviation(TransE):
	"""
	TransE with Source deviation vectors
	"""
	def __init__(self, nRels, nEnts, nSrc, dim=100, norm=2, margin=1, l2reg=0.1, relPretrained=None, entPretrained=None):
		super(SubjKB_Deviation, self).__init__(nRels, nEnts, dim, norm, margin, l2reg, relPretrained, entPretrained)

		self.nSrc = nSrc

		# initialize 'add-on' deviation embeddings for each source
		self.srcEnts = nn.ModuleList()
		self.srcRels = nn.ModuleList()
		
		for srcId in range(nSrc):

			# NOTE: self.fwd() relies on 0 padding tokens.
			rels = nn.Embedding(self.nRels, self.dim, padding_idx=0)
			ents = nn.Embedding(self.nEnts, self.dim, padding_idx=0)

			# pretrained vectors?
			if self.relPretrained is not None:
				rels.weight.data.copy_(self.relPretrained)

			if self.entPretrained is not None:
				ents.weight.data.copy_(self.entPretrained)

			self.srcEnts.append(ents.to(device))
			self.srcRels.append(rels.to(device))

	def normalize_weights(self):
		pass

		# normalize embeddings
		# if self.ebatch != None:
		# 	self.ents.weight.data[self.ebatch] = F.normalize(self.ents.weight.data[self.ebatch], self.norm, -1)
		# 	self.rels.weight.data[self.rbatch] = F.normalize(self.rels.weight.data[self.rbatch], self.norm, -1)
		# else:
		# 	self.ents.weight.data = F.normalize(self.ents.weight.data, self.norm, -1)
		# 	self.rels.weight.data = F.normalize(self.rels.weight.data, self.norm, -1)
		
		# for srcId in range(self.nSrc):
		# 	# ony need to normalize embeddings that were updated in last batch
		# 	if self.ebatch != None:
		# 		self.srcEnts[srcId].weight.data[self.ebatch] =  F.normalize(self.srcEnts[srcId].weight.data[self.ebatch], self.norm, -1)
		# 		self.srcRels[srcId].weight.data[self.rbatch] =  F.normalize(self.srcRels[srcId].weight.data[self.rbatch], self.norm, -1)
		# 	else:
		# 		self.srcEnts[srcId].weight.data =  F.normalize(self.srcEnts[srcId].weight.data, self.norm, -1)
		# 		self.srcRels[srcId].weight.data =  F.normalize(self.srcRels[srcId].weight.data, self.norm, -1)


	def fwd(self, all_heads, all_rels, all_tails, all_sources):
		
		## 1. Convert input sequences into vector representations
		heads_vec = self.ents(LongTensor(all_heads).to(device))
		rels_vec = self.rels(LongTensor(all_rels).to(device))
		tails_vec = self.ents(LongTensor(all_tails).to(device))

		## 2. Add source-specific vectors
		for srcId in range(self.nSrc):

			mask = all_sources == srcId
			if sum(mask) < 1:
				continue

			# apply source mask vector to all heads, rels, tails
			head_ids = LongTensor(all_heads * mask).to(device)
			rel_ids = LongTensor(all_rels * mask).to(device)
			tail_ids = LongTensor(all_tails * mask).to(device)

			# add nonzero vectors only to rows where source == srcId
			heads_vec += self.srcEnts[srcId](head_ids) 
			rels_vec += self.srcRels[srcId](rel_ids)
			tails_vec += self.srcEnts[srcId](tail_ids)

		# normalize the summed vectors
		# heads_vec = F.normalize(heads_vec, self.norm, -1)
		# rels_vec = F.normalize(rels_vec, self.norm, -1)
		# tails_vec = F.normalize(tails_vec, self.norm, -1)

		scores = torch.norm(heads_vec + rels_vec - tails_vec, self.norm, 1)
		return scores

	def loss(self, gold_scores, corrupt_scores):
		zeros = torch.tensor([0], dtype=torch.float).to(device)
		margin_tensor = torch.tensor([self.margin], dtype=torch.float).to(device)
		rank_loss = torch.sum(torch.max(margin_tensor + gold_scores - corrupt_scores, zeros)).to(device)
		
		# regularize l2 norm of all base entity vectors to be <= 1
		regularization = self.reg_weight * torch.sum( 
				torch.max( 
					torch.norm(self.ents.weight, 2, 1) - torch.tensor([1], dtype=torch.float).to(device),
					torch.tensor([0], dtype=torch.float).to(device)
				)
			).to(device)

		# regularize l2 norm of all deviation vectors to be small
		for srcId in range(self.nSrc):
			regularization += self.reg_weight * torch.sum( torch.norm(self.srcEnts[srcId].weight, 2, 1) )

		# also regularize for the gold_scores to be close to 0
		return rank_loss + regularization + self.reg_weight*torch.sum(gold_scores)

	def get_ent_vect(self, entId, srcId):
		ent_tensor = LongTensor([entId]).to(device)

		vec = self.ents(ent_tensor)
		src = self.srcEnts[srcId](ent_tensor) 

		return vec+src

	def get_rel_vect(self, relId, srcId):
		rel_tensor = LongTensor([relId]).to(device)

		vec = self.rels(rel_tensor)
		src = self.srcRels[srcId](rel_tensor) 

		return vec+src

	def norm_step(self):
		pass



class SubjKB_Matrix(TransE):
	"""
	Matrix model - same transformation for entities and relations
	"""
	def __init__(self, nRels, nEnts, nSrc, dim=100, norm=2, dropout=0.05, nonlinearity=None):
		super(SubjKB_Matrix, self).__init__(nRels, nEnts, dim, norm)

		self.nSrc = nSrc
		self.dropout = dropout	

		# initialize transform matrix
		self.sourceNN = nn.ModuleList()
		
		for srcId in range(nSrc):

			# forward layer + dropout
			seq = nn.Sequential(
				nn.Linear(self.dim, self.dim),
				nn.Dropout(self.dropout)
				)

			# add nonlinearity
			if nonlinearity == 'tanh':
				seq.add_module('nonlinearity', nn.Tanh())
			elif nonlinearity == 'relu':
				seq.add_module('nonlinearity', nn.ReLU())

			self.sourceNN.append(seq.to(device))


	# def normalize_weights(self):
	# 	pass

	def fwd(self, all_heads, all_rels, all_tails, all_sources):
		
		h = LongTensor(all_heads).to(device)
		r = LongTensor(all_rels).to(device)
		t = LongTensor(all_tails).to(device)

		heads_vec = torch.zeros([len(all_heads), self.dim]).to(device)
		rels_vec = torch.zeros([len(all_rels), self.dim]).to(device)
		tails_vec = torch.zeros([len(all_tails), self.dim]).to(device)

		## It is faster to feed each entity,rel vector through 
		## each source feed-forward layer. Then mask out rows that belong to other srcsIds
		for srcId in range(self.nSrc):
			mask = all_sources == srcId
			if sum(mask) < 1:
				continue
			mask_tensor = FloatTensor(mask).view(len(all_sources),1).to(device)

			h_hidden = self.sourceNN[srcId](self.ents(h))
			r_hidden = self.sourceNN[srcId](self.rels(r))
			t_hidden = self.sourceNN[srcId](self.ents(t))

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


class FeedForward(TransE):
    """
    Feed forward model that does not use translational assumption
    """
    def __init__(self, nRels, nEnts, dim=100, hidden_dim=128):
        super(FeedForward, self).__init__(nRels, nEnts, dim=dim, norm=1, relPretrained=None, entPretrained=None)

#         self.ents = nn.Embedding(nEnts, dim, padding_idx=0)
#         self.rels = nn.Embedding(nRels, dim, padding_idx=0)

        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim,hidden_dim)
        self.fc3 = nn.Linear(hidden_dim,1)
    
    def normalize_weights(self):
        pass
    
    # def forward(self, heads, rels, tails, sources, heads_bad, rels_bad, tails_bad, sources_bad):
    def fwd(self, hs, rs, ts, ss):
        # first hidden layer

        h = self.ents(LongTensor(hs).unsqueeze(-1).to(device))
        r = self.rels(LongTensor(rs).unsqueeze(-1).to(device))
        t = self.ents(LongTensor(ts).unsqueeze(-1).to(device))
        
        x = torch.cat((h,r,t),dim=1)
        # first hidden layer
        x = nn.ReLU()(nn.Dropout(p=0.05)(self.fc1(x)))
        # second hidden layer
        x = nn.ReLU()(nn.Dropout(p=0.05)(self.fc2(x)))
        # average over head, rel, tail
        x = torch.mean(x, dim=1)
        # second hidden layer
        # x = nn.ReLU()(nn.Dropout(p=0.05)(self.fc2(x)))
        # final layer
        result = nn.Sigmoid()(self.fc3(x))
        return result.squeeze(-1)


class FeedForward_Source(TransE):
    """
    Feed forward model that does not use translational assumption
    """
    def __init__(self, nRels, nEnts, nSrc, dim=100, hidden_dim=128):
        super(FeedForward_Source, self).__init__(nRels, nEnts, dim=dim, norm=1, relPretrained=None, entPretrained=None)

        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim,hidden_dim)
        self.fc3 = nn.Linear(hidden_dim,nSrc)
    
    def normalize_weights(self):
        pass
    
    # def forward(self, heads, rels, tails, sources, heads_bad, rels_bad, tails_bad, sources_bad):
    def fwd(self, hs, rs, ts, ss):
        # first hidden layer

        h = self.ents(LongTensor(hs).unsqueeze(-1).to(device))
        r = self.rels(LongTensor(rs).unsqueeze(-1).to(device))
        t = self.ents(LongTensor(ts).unsqueeze(-1).to(device))
        
        x = torch.cat((h,r,t),dim=1)
        # first layer
        x = nn.ReLU()(nn.Dropout(p=0.05)(self.fc1(x)))
        # second layer
        x = nn.ReLU()(nn.Dropout(p=0.05)(self.fc2(x)))
        # average over head, rel, tail
        x = torch.mean(x, dim=1)
        # second hidden layer
        # x = nn.ReLU()(nn.Dropout(p=0.05)(self.fc2(x)))
        # final layer
        result = nn.Sigmoid()(self.fc3(x)) # probs for each source

        # grab values for correct sources
        ss = torch.Tensor(ss).long().to(device)
        result = result.gather(1, ss.view(-1,1))

        return result.squeeze(-1)









