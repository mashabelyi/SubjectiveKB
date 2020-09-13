from joblib import Parallel, delayed
from joblib import wrap_non_picklable_objects
from tqdm import tqdm
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import json
import os
from math import floor
from collections import Counter
import spacy
from spacy.matcher import Matcher 
from spacy.tokens import Span 

# Parameters
# ==================================================
parser = ArgumentParser("ExtractYelpTriples", formatter_class=ArgumentDefaultsHelpFormatter, conflict_handler='resolve')

parser.add_argument("--fin",  help="Input file", required=True)
parser.add_argument("--fout",  help="Output file", required=True)
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


def find_neg(vb):
    for tok in vb.lefts:
        if tok.dep_ == 'neg':
            return "{} {}".format(tok.text, vb.text)

    for tok in vb.rights:
        if tok.dep_ == 'neg':
            return "{} {}".format(vb.text, tok.text)
    return vb.text

def process_head(head, subj_text):
    tmp = []
#     negtok = None 
    # find negations
#     head_text = find_neg(head)
    
    for tok in head.lefts:
        if 'subj' in tok.dep_ and tok.text != subj_text:
            return []
        
    for tok in head.rights:
        
        if tok.dep_ == "prep":
            for child in tok.children:
                tmp.append((subj_text, "{} {}".format(head.text, tok.text), child.text))

        elif tok.dep_.endswith("obj"):
            tmp.append((subj_text, head.text, tok.text))

            # TODO check for 'conj' coming out of tok
            # eg. I like cats and dogs -> (i, like, cats), (i, like, dogs)
            
        elif tok.dep_ =="advmod" or tok.dep_ =="ccomp" or tok.dep_ == "acomp" or tok.dep_ == "xcomp" or tok.dep_ == "attr":
#             tmp.append((subj_text, head_text, ' '.join([t.text for t in tok.subtree])))
            if list(tok.subtree)[-1].dep_.endswith('obj'):
                tmp.append((subj_text, head.text, ' '.join([t.text for t in tok.subtree])))
            else:
                tmp.append((subj_text, head.text, tok.text))
        
        elif tok.dep_=="conj":
#             # unless its the root of something else..
            tmp += process_head(tok, subj_text)


    return tmp
def process_verbs(doc, matcher):
    #define the pattern 
    verb_particles = [{'POS':'VERB'}, #, 'OP':"?"
                      {'DEP':'prt'}]
    verb_negation_l = [{'DEP':'neg'},
                     {'POS':'VERB'}]
    verb_negation_r = [{'POS':'VERB'},
                      {'DEP':'neg'}]
    aux_verb = [{'DEP':'aux'}, {'POS':'VERB'}]
    verb_prep = [{'POS':'VERB'}, {'DEP':'prep'}]
    # auxpass_verb = [{'DEP':'auxpass', 'POS':'VERB'}, {'POS':'VERB'}]

    # Matcher class object 
    matcher.add("verb_phrases", None, verb_particles) 
    matcher.add("verb_negation_l", None, verb_negation_l)
    matcher.add("verb_negation_r", None, verb_negation_r)
    matcher.add("aux_verb", None, aux_verb)
    matcher.add("aux_verb", None, verb_prep)
#     matcher.add("auxpass_verb", None, auxpass_verb)

    matches = matcher(doc)
    spans = []
    for i in range(len(matches)):
        span = doc[matches[i][1]-i:matches[i][2]-i] 
        span.merge()
    return doc

def extract_relations(doc, matcher):
    
    # combine noun phrases
    spans = list(doc.ents) + list(doc.noun_chunks)
    for span in spans:
        span.merge()
    # combine verb phrases
    doc = process_verbs(doc, matcher)
    
    # Visualize
    # for sent in doc.sents:
    #     displacy.render(sent, style="dep")
    
    triples = []
    
    sent_subjects = [] # list of subjects in each sentence
    for sent in doc.sents:
        sent_subjects.append([w for w in sent if w.dep_ == 'nsubj' or w.dep_=='nsubjpass'])
    
#     print()
#     [print(s, '-->', s.dep_) for s in subjects]
#     print()
    
    for sent in sent_subjects:
        for ent in sent:
#             print(ent)
            triples += process_head(ent.head, ent.text)
            
            
    return triples

# non-packable spacy object fix from 
# https://joblib.readthedocs.io/en/latest/auto_examples/serialization_and_wrappers.html
@delayed
@wrap_non_picklable_objects
def process(review, fout):

    nlp = spacy.load('en_core_web_sm', disable=['textcat'])
    matcher = Matcher(nlp.vocab) 

    relations = extract_relations(nlp(review['text']), matcher)
    
    with open(fout, 'a') as f:
        for head, rel, tail in relations:
            f.write("{}\t{}\t{}\t{}\t{}\t{}\n".format(head, rel, tail, review['stars'], review['review_id'], review['business_id']))

print("LOADING REVIEWS")
reviews = load_reviews(args.fin)

if args.parallel > 1:
    # Parallel(n_jobs=args.parallel)(delayed(process)(i, args.fout) for i in tqdm(reviews))
	Parallel(n_jobs=args.parallel)(process(i, args.fout) for i in tqdm(reviews))
else:
	for review in reviews:
		process(review, args.fout)



