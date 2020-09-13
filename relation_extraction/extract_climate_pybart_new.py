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
import csv

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


def load_climate(fin):
    data = []
    with open(fin, 'r') as f:
        reader = csv.reader(f)
        for tmp in reader:
            if tmp[0] == 'URL':
                continue

            try:
                data.append({'text': tmp[-1], 'station':tmp[2], 'show':tmp[3], 'datetime':tmp[1]})
            except:
                pass
    return data

def merge_span(span, tokens, offset):
    if len(span) < 2:
        return tokens
    
    span_text = []
    span_parents = []
    idx = set()
    for token in span:
        idx.add(token.i)
        if token.pos_ not in {'DET'}: # ignore a/the/another?
            span_text.append(token.text)
#         span_text.append(token.text)
        
        for parent in token._.parent_list:
            if (parent['head'].i < span.start or
               parent['head'].i  > span.end):
                span_parents.append(parent)
    
    newdoc = ([tokens[i] for i in range(0,span.start-offset)] 
              + [{'i':span.start-offset, 'span':idx, 'text': ' '.join(span_text), 'parents':span_parents, 'children':[]}] 
              + [tokens[i] for i in range(span.end-offset,len(tokens))])
    
    return newdoc

def state_pred_obj(token):
    """
    Get the predicate and tail of a STATE token
    """
    pred = None
    tails = []

    if 'children' not in token:
        return pred, tails

    for child in token['children']:
        if child['rel'] == 'ev':
            pred = child['text']
        # xcmop: "is STATE s hoax" -> pred=is, tail=hoax
        # nmod: "seems STATE like a hoax" -> pred=is, tail=hoax
        elif child['rel'] == 'xcomp' or child['rel'].startswith('nmod'):
            tails.append(child['text'])

    return pred, tails
            
    
    
def compound_predicate(tokens, predId):
    """
    merge compound predicates.
    Return predicate text, list of children?
    """
    pred_phrase = ['' for i in range(len(tokens))]
    pred_phrase[predId] = tokens[predId]['text']
    tails = []

    if 'children' not in tokens[predId]:
        return ' '.join(pred_phrase).strip(), tails

    # fill in pred_phrase with compound predicate tokens
    for child in tokens[predId]['children']:
        if child['rel'] == 'xcomp':
            phrase, _tails = compound_predicate(tokens, child['id'])
            pred_phrase[child['id']] = phrase
            tails.extend(_tails)
        elif child['rel'] in {'aux', 'mark'}:
            pred_phrase[child['id']] = child['text']
        elif child['rel'] == 'neg':
            pred_phrase[child['id']] = 'NOT'
            
        elif child['rel'] == 'dobj' or child['rel'] == 'iobj' or child['rel'].startswith('nmod'):
            tails.append(child['text'])

        
    return ' '.join([x for x in pred_phrase if x!='']), tails

def extract_relations(doc):
    tokens = []
    for i in range(len(doc)):
        tokens.append({'i':i, 'span':{i}, 'text':doc[i].text, 'parents': doc[i]._.parent_list, 'pos':doc[i].pos_, 'children':[]})

    spans = list(doc.ents) + list(doc.noun_chunks)
    # sort in order of ocurrence in the sentence
    spans = sorted(spans, key=lambda x: x.start)

    offset = 0
    _curr = 0 # keep track of location in the sentence
    for span in spans:
        if span.start < _curr:
            # this is a repeated span (came up in both entity and noun chunk)
            continue
    #     print(span, "{}-{} ({})".format(span.start, span.end ,len(span)))
        _curr = span.end
        
        if len(span) > 1:
            tokens = merge_span(span, tokens, offset)
            offset += len(span) - 1
        
    # remap the indices
    idxorig = {}
    for i in range(len(tokens)):
        token = tokens[i]
        for idx in list(token['span']):
            idxorig[idx] = i
        token['i'] = i

    # Loop through the tokens again and save arrays of token children
    children = [[] for i in range(len(tokens))]
    # nsubj_ids = []
    for token in tokens:
        # iterate through the token's parents
        for parent in token['parents']:
            # find the new location of the head (after merging spans)
            parent_dict = tokens[idxorig[parent['head'].i]]
            # if this is not  circular (root) dependency, add it to the head's children array
            if token['i'] != idxorig[parent['head'].i]:
                parent_dict['children'].append({'id': token['i'], 'text':token['text'], 'rel': parent['rel']})


    # finally iterate through each token again 
    # and pull out propositional tuples
    tuples = []
    for token in tokens:
        head = token['text']
        
        # iterate through the token's parents to find nsbuj
        # ignoring any compund nsubj types (e.g. nsubj:xcomp(INF)) or nsubjpass
        for parent in token['parents']:
            if parent['rel'] == 'nsubj':
                # locate the predicate token in the merged list of tokens
                parent_dict = tokens[idxorig[parent['head'].i]]
                # get the predicate and its object(s)
                if parent['head'].text == 'STATE':
                    pred, tails = state_pred_obj(parent_dict)
                    
                    if pred is not None and len(tails) > 0:
                        for tail in tails:
                            tuples.append((head, pred, tail))
                        
                else:
                    predId = idxorig[parent['head'].i]
                    pred, tails = compound_predicate(tokens, predId)
                    if pred is not None and len(tails) > 0:
                        for tail in tails:
                            tuples.append((head, pred, tail))

    return tuples

# @delayed
def process(data, fout):

    nlp = spacy.load('en_ud_model_sm', disable=['textcat'])
    converter = Converter(remove_unc=True) #remove_unc=True
    nlp.add_pipe(converter, name="BART")

    ## REMOVE SPECIAL CHARACTERS. Else nlp(text) will throw an error
    # text = re.sub('[^A-Za-z0-9 .,?!-]+', '', data['text'])
    

    try:
        relations = extract_relations(nlp(data['text']))
        
        with open(fout, 'a') as f:
            for head, rel, tail in relations:
                f.write("{}\t{}\t{}\t{}\t{}\t{}\n".format(data['station'], data['datetime'], data['show'], head, rel, tail))
    except:
        print("ERROR")
        with open(args.err, 'a') as f:
            f.write(json.dumps(data) + '\n')

@delayed
def process_file(fin):
    articles = load_climate(fin)
    for data in articles:
        process(data, args.fout)

# process each file in folder
files_to_process = []
for file in os.listdir(args.fin):
    if file.endswith('.csv'):
        files_to_process.append(os.path.join(args.fin, file))

print("Loaded {} files".format(len(files_to_process)))

Parallel(n_jobs=args.parallel)(process_file(fpath) for fpath in tqdm(files_to_process))



# print("LOADING DATA")
# articles = load_climate(args.fin)
# print("loaded {} reviews".format(len(articles)))

# if args.parallel > 1:
#     # Parallel(n_jobs=args.parallel)(delayed(process)(i, args.fout) for i in tqdm(reviews))
#     Parallel(n_jobs=args.parallel)(process(i, args.fout) for i in tqdm(articles))
# else:
#     print("here")
#     print("loaded {} reviews".format(len(articles)))
#     for data in tqdm(articles):
#         process(data, args.fout)
