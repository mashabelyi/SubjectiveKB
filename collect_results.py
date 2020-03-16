"""
Usage

python3 collect_results.py ./ tuning_results.tsv

"""

import sys, os, json

root = sys.argv[1]
outfile = sys.argv[2]

results = []

for dirname in os.listdir(root):
	if dirname.startswith('YELP_d') or dirname.startswith('YELP_src_d') or dirname.startswith('YELP_src_mat_d'):
		valF = os.path.join(root, dirname, ('val_eval1.tsv' if dirname.startswith('YELP_src_d') else 'val_eval.tsv'))
		
		model = 'baseline'
		if dirname.startswith('YELP_src_d'):
			model = 'src'
		elif dirname.startswith('YELP_src_mat_d'):
			model = 'src_mat'

		if os.path.exists(valF):

			# get parameters
			with open(os.path.join(root, dirname, 'config.json'), 'r') as f:
				config = json.load(f)

			with open(valF, 'r') as f:
				i = 0
				for line in f:
					if i==1: 
						data = line.split('\t')
						results.append([
							model,
							config['embedding_dim'],
							config['margin'], 
							config['learning_rate'],
							config['norm'],
							data[0],
							data[1],
							data[2]])
					i+=1



with open(outfile, 'w') as f:
	for res in results:
		f.write('\t'.join([str(x) for x in res]))

