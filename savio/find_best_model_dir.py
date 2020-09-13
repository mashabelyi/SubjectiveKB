"""
Usage

python3 collect_results.py ./ tuning_results.tsv

"""

import sys, os, json

root = sys.argv[1]


def get_model_metrics(dir):
	# get parameters
	with open(os.path.join(root, dirname, 'config.json'), 'r') as f:
		config = json.load(f)

	heads_h10 = 0 
	tails_h10 = 0 

	with open(os.path.join(dir,'val_heads.tsv'), 'r') as f:
		for line in f:
			if line.startswith('total'): 
				data = line.strip().split('\t')
				heads_h10 = float(data[3])

	
	if os.path.exists(os.path.join(dir,'val_tails.tsv')):
		with open(os.path.join(dir,'val_tails.tsv'), 'r') as f:
			for line in f:
				if line.startswith('total'): 
					data = line.strip().split('\t')
					tails_h10 = float(data[3])


	return heads_h10, tails_h10

best_heads = -1.0
best_heads_dir = ''
best_tails = -1.0
best_tails_dir = ''

for dirname in os.listdir(root):

	if os.path.exists(os.path.join(root, dirname,'val_heads.tsv')):
		heads_h10, tails_h10 = get_model_metrics(os.path.join(root, dirname))

		# print(dirname, heads_h10, tails_h10, best_heads, best_tails)

		if heads_h10 > best_heads:
			best_heads = heads_h10
			best_heads_dir = dirname

		if tails_h10 > best_tails:
			best_tails = tails_h10
			best_tails_dir = dirname


print("\nBest heads H@10 = ", best_heads)
print(best_heads_dir)

print("\nBest tails H@10 = ", best_tails)
print(best_tails_dir)



