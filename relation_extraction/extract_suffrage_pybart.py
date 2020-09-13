from joblib import Parallel, delayed
from joblib import wrap_non_picklable_objects
from tqdm import tqdm
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import json
import os, re
from math import floor
from collections import Counter
import spacy
from spacy.matcher import Matcher 
from spacy.tokens import Span 
# Add pyBART Converter to Spacy pipeline
from pybart.api import Converter
from collections import defaultdict


# Parameters
# ==================================================
parser = ArgumentParser("ExtractYelpTriples", formatter_class=ArgumentDefaultsHelpFormatter, conflict_handler='resolve')

parser.add_argument("--fin",  help="Input file", required=True)
parser.add_argument("--fout",  help="Output file", required=True)
parser.add_argument("--err",  help="error log file", required=True)
parser.add_argument("--parallel", default=1, type=int, help="Num parallel processes to run")

args = parser.parse_args()
print(args)
# ==================================================


def load_reviews(fin):
	data = []
	with open(fin, 'r') as f:
		for line in f:
			data.append(json.loads(line))
	return data

def load_suffrage(fin):
	data = []
	with open(fin, 'r', encoding="ISO-8859-1") as f:
		for line in f:
			
			tmp = line.strip().split('\t')
			if len(tmp) < 2:
				continue
			data.append({'id': tmp[0], 'text':tmp[1]})
	return data

def get_obj_phrase(tokenId, children, doc):
	
	modifiers = {'amod', 'compound', 'attr', 'advmod', 'ccomp', 'acomp', 'xcomp', 'case', 'det'}
	
	phrase = ['' for i in range(len(doc))]
	phrase[tokenId] = doc[tokenId].text
	
	for child in children[tokenId]:
#         print(child)
		if child[1]['rel'] in modifiers:
			phrase[child[0].i] = child[0].text
	
	## only keep contiguous tokens
	## I was super excited -> "I", even though excited modifies "I"
	lo = tokenId
	hi = tokenId
	while lo > 0 and phrase[lo] != '':
		lo -= 1
	while hi < len(doc) and phrase[hi] != '':
		hi += 1
		
		
	return ' '.join([x for x in phrase[lo:hi+1] if len(x) > 0])


def get_subj_phrase(tokenId, children, doc):
	
	
	modifiers = {'amod', 'compound', 'attr', 'advmod', 'ccomp', 'acomp', 'xcomp'}
	
	phrase = ['' for i in range(len(doc))]
	phrase[tokenId] = doc[tokenId].text
	
	for child in children[tokenId]:
#         print(child)
		if child[1]['rel'] in modifiers:
			phrase[child[0].i] = child[0].text
	
	## only keep contiguous tokens
	## I was super excited -> "I", even though excited modifies "I"
	lo = tokenId
	hi = tokenId
	while lo > 0 and phrase[lo] != '':
		lo -= 1
	while hi < len(doc) and phrase[hi] != '':
		hi += 1
		
	
	return ' '.join([x for x in phrase[lo:hi+1] if len(x) > 0])


def get_pred(tokenId, children, doc, pred_obj):
	modifiers = {'mark', 'neg', 'aux'}
	rels = {'advmod', 'xcomp'}
	
	phrase = ['' for i in range(len(doc))]
	phrase[tokenId] = doc[tokenId].text
	
	obj_list = pred_obj[tokenId].copy()
	
	## attach negations, aux, mark
	## (do n't want), (to go)
	for child in children[tokenId]:
		# dont include adverbs liek
		# 'really want', 'was really excited'
		if child[0].pos_ == 'ADV':
			continue
		
		if child[1]['rel'] in modifiers:
			phrase[child[0].i] = child[0].text
			
		elif child[1]['rel'] in {'advmod', 'xcomp', 'ev'}:
			if 'CONJ' in child[0].pos_:
				continue
			
			if child[1]['head'].text == 'STATE' and  child[1]['rel'] == 'xcomp':
				continue
				
			# e.g. 
			# - seems to like
			# - wants to go
			_phrase, _obj_list = get_pred(child[0].i, children, doc, pred_obj)
			phrase[child[0].i] = _phrase
	
			##merge pred_obj relations
			obj_list.extend(_obj_list)
	
	## attach modifiers and xcomp clauses
#     rels = {'advmod', 'xcomp'}
#     for parent in doc[tokenId]._.parent_list:
#         if parent['rel'] in rels:
#             phrase[parent['head'].i] = parent['head'].text

	pred_text = ' '.join([x for x in phrase if len(x) > 0 and x != 'STATE'])
	return pred_text, obj_list


def extract_relations(doc):
	subj_pred = defaultdict(list)
	pred_obj = defaultdict(list)

	triples = []

	## Save pointers from tokens to their child dependencies
	## Merge multi-word predicates and subjets, object
	children = [[] for i in range(len(doc))] #keep track of pyBART children
	for tokenId in range(len(doc)):
		token = doc[tokenId]
	# for token in doc:
		
		for parent in token._.parent_list:
			if parent['rel'] == 'punct':
				continue
			
			predId = parent['head'].i
			
			# save nsubj, nsubjpass as relation subjects
	#         if parent['rel'].startswith('nsubj'):
			if parent['rel'] == 'nsubj':
				# Catch modifiers without a predicate
				# Red roses -> (roses STATE red)
				if 'mod' in parent['src']:
					triples.append((token.text, 'STATE', parent['head'].text))
				# all others - save idx pointers
				else:					
					subj_pred[tokenId].append(predId)
					
			if parent['rel'].endswith('obj') or parent['rel'].startswith('nmod'):
				pred_obj[predId].append(tokenId)
				
			if parent['head'].text == 'STATE' and parent['rel'] == 'xcomp':
				pred_obj[predId].append(tokenId)
				
				
			children[parent['head'].i].append((token, parent))


	for subjId in subj_pred:
		subj_text = get_subj_phrase(subjId, children, doc)
		
		for predId in subj_pred[subjId]:
			pred_text, obj_list = get_pred(predId, children, doc, pred_obj)
			# print(subjId, predId, obj_list)
			
			for objId in obj_list: 
				if objId == subjId:
					continue
	#             print(subj_text,'-', pred_text,'-', get_obj_phrase(objId, children, doc))
				triples.append((subj_text, pred_text, get_obj_phrase(objId, children, doc)))

	return triples

# non-packable spacy object fix from 
# https://joblib.readthedocs.io/en/latest/auto_examples/serialization_and_wrappers.html
@delayed
@wrap_non_picklable_objects
def process(data, fout):
	# nlp = spacy.load('en_core_web_sm', disable=['textcat'])
	nlp = spacy.load('en_ud_model_sm', disable=['textcat'])
	converter = Converter(remove_unc=True) #remove_unc=True
	nlp.add_pipe(converter, name="BART")

	## REMOVE SPECIAL CHARACTERS. Else nlp(text) will throw an error
	text = re.sub('[^A-Za-z0-9 .,?!-]+', '', data['text'])

	try:
		relations = extract_relations(nlp(text))
		
		with open(fout, 'a') as f:
			for head, rel, tail in relations:
				f.write("{}\t{}\t{}\t{}\n".format(data['id'], head, rel, tail))
	except:
		with open(args.err, 'a') as f:
			f.write(json.dumps(data) + '\n')

print("LOADING DATA")
articles = load_suffrage(args.fin)
print("loaded {} reviews".format(len(articles)))

if args.parallel > 1:
	# Parallel(n_jobs=args.parallel)(delayed(process)(i, args.fout) for i in tqdm(reviews))
	Parallel(n_jobs=args.parallel)(process(i, args.fout) for i in tqdm(articles))
else:
	print("here")
	print("loaded {} reviews".format(len(articles)))
	for data in tqdm(articles):
		process(data, args.fout)



