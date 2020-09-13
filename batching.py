"""
Batch Loader for SubjKB model

USAGE
-------------------------------------------------------------
loader = BatchLoader(train, bernoulli_p, 
                     goldens, all_heads, all_rels, all_tails, 
                     batch_size=128)
for i in range(epochs):
    loader.reset()
    while loader.has_next():
        print(i)
        pos, neg = loader.next_batch()
-------------------------------------------------------------

@mashabelyi
"""
import numpy as np
from collections import defaultdict


def list2tup(arr):
    return tuple(x for x in arr)

class BatchLoader(object):
    def __init__(self, train, bernoulli_p, 
                 goldens, ents, sources,
                 batch_size=128, neg_ratio=1.0):
        
        self.train = train
        self.bernoulli_p = bernoulli_p
        self.goldens = goldens
        self.ents = list(ents)
        self.nEnts = len(self.ents)
        # self.tails = list(tails)
        # self.nTails = len(self.tails)
        self.sources = sources
        self.nSources = len(sources)

        self.batch_size = batch_size
        self.neg_ratio = neg_ratio
        self.curr_idx = 0
        self.N = len(train)

    def reset(self):
        # start new epoch
        self.curr_idx = 0
        self.permuted = np.random.permutation(self.train)
        
    def has_next(self):
        return self.curr_idx < self.N

    def corrupt_head(self, head, rel, tail, source):
        new = self.ents[np.random.randint(0, self.nEnts)] # sample head
        while (new, rel, tail) in self.goldens: 
            new = self.ents[np.random.randint(0, self.nEnts)] # sample head
        return (new, rel, tail, source)

    def corrupt_tail(self, head, rel, tail, source):        
        new = self.ents[np.random.randint(0, self.nEnts)] # sample tail
        while (head, rel, new) in self.goldens: 
            new = self.ents[np.random.randint(0, self.nEnts)] # sample tail
        return (head, rel, new, source)

    def corrupt_source(self, head, rel, tail, source):
        candidates = [s for s in self.sources if s != source]
        
        for newsrc in candidates:
            if (head, rel, tail, newsrc) in self.goldens: 
                continue
        
        if (head, rel, tail, newsrc) in self.goldens:
            return None
        else:
            return (head, rel, tail, newsrc)

        
    def next_batch(self):
        # return batch of positive and negative samples
        if self.curr_idx >= self.N:
            return []
        
        batch = self.permuted[self.curr_idx:min(self.curr_idx + self.batch_size, self.N)]
        batch_dbl = []
        self.curr_idx += self.batch_size
        corrupted = []
        for h,r,t,s in batch:
            # corrupt the head or the tail based on precalculated probability

            pr = self.bernoulli_p[r]
            if np.random.uniform() > pr:
                corrupted.append(self.corrupt_tail(h,r,t,s)) # replace tail
            else:
                corrupted.append(self.corrupt_head(h,r,t,s)) # reaplace head

            # UPDATE: Corrput both head and tail to train the model for both ehad-prediction and tail-prediction task.
            # this doesnt actually make any difference
            # batch_dbl.append((h,r,t,s))
            # batch_dbl.append((h,r,t,s))
            # corrupted.append(self.corrupt_tail(h,r,t,s)) # replace tail
            # corrupted.append(self.corrupt_head(h,r,t,s)) # reaplace head

        # return np.array(batch_dbl), np.array(corrupted)
        return np.array(batch), np.array(corrupted)

    def validation_triples(self, val):
        # return all possible corrupt triples pairs per validation sample
        # should contain all possible triples that don't exist in training
        print("setting up evaluation triples")
        batch = {}
        for h,r,t,s in val:
            corrupted = []

            pr = self.bernoulli_p[r]
            if np.random.uniform() > pr:
            # if True:
                # replace tails
                corrupted = {(h,r,x) for x in self.ents}
                corrputed = corrupted - (corrupted & self.goldens)
                corrupted = [(a,b,c,s) for a,b,c, in list(corrupted)]
                # 
                # for e in self.ents:
                #     if (h, r, e) not in self.goldens:
                #         corrupted.append((h,r,e,s))

            else:
                # replace heads
                corrupted = {(x,r,t) for x in self.ents}
                corrputed = corrupted - (corrupted & self.goldens)
                corrupted = [(a,b,c,s) for a,b,c, in list(corrupted)]

                # for e in self.ents:
                #     if (e, r, t) not in self.goldens:
                #         corrupted.append((e,r,t,s))


            batch[(h,r,t,s)] = np.array(corrupted)
        return batch

    
    def get_pred_tails(sefl, val, train, nents):
    
        hrs_t = defaultdict(set)
        for h,r,t,s in train:
            # hrs_t[(h,r,s)].add(t) # hard task
            hrs_t[(h,r)].add(t) # standard task
        all_tails = set(np.arange(2, nents))
        
        out = {}
        
        for h,r,t,s in val:
            # candidates = list(all_tails-hrs_t[(h,r,s)]) # hard task
            candidates = list(all_tails-hrs_t[(h,r)]) # standard task
            
            n = len(candidates)
            hh = np.expand_dims(np.repeat(h, n), axis=1)
            rr = np.expand_dims(np.repeat(r, n), axis=1)
            tt = np.expand_dims(candidates, axis=1)
            ss = np.expand_dims(np.repeat(s, n), axis=1)
            
            out[(h,r,t,s)] = np.concatenate((hh,rr,tt,ss), axis=-1)
        
        return out


    def get_pred_heads(self, val, train, nents):
    
        trs_h = defaultdict(set)
        for h,r,t,s in train:
            # trs_h[(t,r,s)].add(h) # hard task (easier, but confusing)
            trs_h[(t,r)].add(h) # standard task
        all_heads = set(np.arange(2, nents))
        
        out = {}
        
        for h,r,t,s in val:
            # candidates = list(all_heads-trs_h[(t,r,s)]) # hard task
            candidates = list(all_heads-trs_h[(t,r)]) # standard task
            
            n = len(candidates)
            hh = np.expand_dims(candidates, axis=1)
            rr = np.expand_dims(np.repeat(r, n), axis=1)
            tt = np.expand_dims(np.repeat(t, n), axis=1)
            ss = np.expand_dims(np.repeat(s, n), axis=1)
            
            out[(h,r,t,s)] = np.concatenate((hh,rr,tt,ss), axis=-1)
        
        return out

    def get_eval_samples(self, val, train, nents, type):
    
        intrain = defaultdict(set)
        for h,r,t,s in train:
            if type == 'head':
                # trs_h[(t,r,s)].add(h) # hard task (easier, but confusing)
                intrain[(t,r)].add(h) # standard task
            else:
                # intrain[(h,r,s)].add(t) # hard task
                intrain[(h,r)].add(t) # standard task

        all_candidates = set(np.arange(2, nents)) # 0=PAD, 1=UNK
        
        for h,r,t,s in val:
            if type == 'head':
                # candidates = list(all_candidates-intrain[(t,r,s)]) # hard task
                candidates = list(all_candidates-intrain[(t,r)]) # standard task
               
                n = len(candidates)
                hh = np.concatenate(([h], candidates))
                tt = np.concatenate(([t], np.repeat(t, n)))

            else:
                candidates = list(all_candidates-intrain[(h,r)]) # standard task

                n = len(candidates)
                hh = np.concatenate(([h], np.repeat(h, n)))
                tt = np.concatenate(([t], candidates))
            
            
            rr = np.repeat(r, n+1)
            ss = np.repeat(s, n+1)

            
            yield hh, rr, tt, ss
            # yield ((h,r,t,s), np.concatenate((hh,rr,tt,ss), axis=-1))

    def validaton_batches(self, val):
        """
            Return a set of batches for quick validation
            - used to calculation validation loss at the end of each epoch
        """

        pos_batches = []
        neg_batches = []

        nval = len(val)
        idx = 0
        while idx < nval:
            batch = val[idx:min(idx + self.batch_size, nval)]
            batch_dbl = []
            idx += self.batch_size
            corrupted = []  
            for h,r,t,s in batch:
                # corrupt the head or the tail
                # pr = self.bernoulli_p[r] if r in self.bernoulli_p else 0.5
                # if np.random.uniform() > pr:
                #     corrupted.append(self.corrupt_tail(h,r,t,s)) # replace tail
                # else:
                #     corrupted.append(self.corrupt_head(h,r,t,s)) # reaplace head
                batch_dbl.append((h,r,t,s))
                batch_dbl.append((h,r,t,s))
                corrupted.append(self.corrupt_tail(h,r,t,s)) # replace tail
                corrupted.append(self.corrupt_head(h,r,t,s)) # reaplace head


            pos_batches.append(np.array(batch_dbl))
            neg_batches.append(np.array(corrupted))

        return pos_batches, neg_batches





