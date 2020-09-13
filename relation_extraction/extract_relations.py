"""
	Extract (subject, relation, object) triplets from file

	USAGE
	python extract_relations.py INPUT.tsv OUT.tsv
	python extract_relations.py ../data/news_current/cnn/00000.tsv test00000.tsv
"""

import sys, os 
import numpy as np

from allennlp.data.tokenizers.token import Token
from allennlp.predictors.predictor import Predictor
from allennlp.predictors import open_information_extraction as oie
from nltk import sent_tokenize
from collections import defaultdict, Counter

import time
print("loading allennlp models")

tic = time.clock()
extractor = Predictor.from_path('https://s3-us-west-2.amazonaws.com/allennlp/models/openie-model.2018-08-20.tar.gz')
coref = Predictor.from_path("https://s3-us-west-2.amazonaws.com/allennlp/models/coref-model-2018.02.05.tar.gz")
t_load = time.clock() - tic
print(t_load)


def resolve_coref_text(document, clusters):
    resolved = document
    for cluster in clusters:
        mention_start, mention_end = cluster[0][0], cluster[0][1] + 1
        mention_span = document[mention_start:mention_end]
        for coref in cluster[1:]:
            final_token = document[coref[1]]
            if final_token in ['his', 'hers']:
                resolved[coref[0]] = ' '.join(mention_span) + " 's"
            else:
                resolved[coref[0]] = ' '.join(mention_span)
        for i in range(coref[0] + 1, coref[1] + 1):
                resolved[i] = ""
    return resolved
            

def tag_to_idx(tags, offset):
    idxs = defaultdict(list)
    for i in range(len(tags)):
        t = tags[i]
        if t == 'O':
            continue
        # tags will have the form 'B-V', 'B-ARG1', 'I-ARG1', etc
        t = t.split('-')[1]
        idxs[t].append(i+offset)
    
    # merge arg2 and arg1
    # for key in ['ARG2', 'ARG3']:
    #     idxs['ARG1'] += idxs[key]
    return idxs

def consolidate(ie_res):
    # consolidate ie results

    ie_slim = []
    for ie in ie_res:
        outputs = [v['tags'] for v in ie['verbs']]
        sent_tokens = [Token(w) for w in ie['words']]

        consolidated = oie.consolidate_predictions(outputs, sent_tokens)
        tags = [t for v,t in consolidated.items()]
        # tags = [t for v,t in ie.items()]

        ie_slim.append({'words':sent_tokens, 'tags':tags})
    return ie_slim

def get_relations(text):
    spacy_doc = coref._spacy(text)
    ie_res = extractor.predict_batch_json([{'sentence':s.text} for s in spacy_doc.sents])
    
    # sometimes there are extra whitespace tokens in the spacy_doc
    # but not in the ie_result (they get removed)
    # So here i create a list of the exact tokens that the ie extractor returs
    # for each sentence, to then pass it to the coref predictor
    doc_tokenized = []
    for sent in ie_res:
        doc_tokenized += sent['words']
    
#     coref_res = coref.predict_tokenized(list(tok.text for tok in spacy_doc))
    coref_res = coref.predict_tokenized(doc_tokenized)

    resolved_doc = resolve_coref_text(doc_tokenized, coref_res['clusters'])
    
    ie_slim = consolidate(ie_res)
    # ie_slim = ie_res

    
    relations = []
    curr = 0
    for sent in ie_slim:
        
        for tags in sent['tags']:
            # get the corect tokens from resolved_doc
            tag_idx = tag_to_idx(tags, curr)

            head = ' '.join([resolved_doc[i] for i in tag_idx['ARG0']]).strip()
            rel = ' '.join([resolved_doc[i] for i in tag_idx['V']]).strip()
            tail = ' '.join([resolved_doc[i] for i in tag_idx['ARG1']]).strip()
            relations.append((head, rel, tail))
    #         print("ARG0[{}], V[{}], ARG1[{}]".format(head, rel, tail))
    #         print()
        curr += len(sent['words'])
    
    return relations


def extract(fin, fout):
	# read in file line by line
	# expect cols: docId, docStr

	todo_parse = []
	with open(fin, 'r') as f:
		for line in f:
			row = line.split('\t')
			todo_parse.append(row)

	for p in todo_parse:
		print(p[0])
		relations = get_relations(p[1])

		if len(relations) == 0:
			continue

		# f = open(fout, 'a')
		with open(fout, 'a') as f:
			for head, rel, tail in relations:
				f.write("{}\t{}\t{}\t{}\n".format(p[0], head, rel, tail))


# if __name__ == '__main__':
# 	if len(sys.argv) < 3:
# 		print("pass in 2 argumens: fin fout")
# 		exit(0)
# 	fin = sys.argv[1]
# 	fout = sys.argv[2]
# 	extract(fin, fout)

