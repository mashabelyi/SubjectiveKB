"""
Usage

python3 collect_results.py ./ tuning_results.tsv

"""

import sys, os, json

root = sys.argv[1]
outfile = sys.argv[2]


def get_model_metrics(dir):
	# get parameters
	with open(os.path.join(root, dirname, 'config.json'), 'r') as f:
		config = json.load(f)

	val_loss = float("inf")	
	train_loss = float("inf")	
	with open(os.path.join(dir,'training_log.csv'), 'r') as f:
		i=0
		for line in f:
			if i==0:
				i+=1
				continue
			data = line.strip().split(',')
			lossV = float(data[2])
			lossT = float(data[1])
			if lossV < val_loss:
				val_loss = lossV
			if lossT < train_loss:
				train_loss = lossT

	heads = []
	with open(os.path.join(dir,'val_heads.tsv'), 'r') as f:
		for line in f:
			if line.startswith('total'): 
				data = line.strip().split('\t')
				heads = [
					config['model'],
					config['embedding_dim'],
					config['margin'], 
					config['learning_rate'],
					config['norm'],
					data[1],
					data[2],
					data[3],
					train_loss,
					val_loss]

		with open(outfile+'.heads', 'a') as f:
			f.write('{}\n'.format('\t'.join([str(x) for x in heads])))

	tails = []
	if os.path.exists(os.path.join(dir,'val_tails.tsv')):
		with open(os.path.join(dir,'val_tails.tsv'), 'r') as f:
			for line in f:
				if line.startswith('total'): 
					data = line.strip().split('\t')
					tails = [
						config['model'],
						config['embedding_dim'],
						config['margin'], 
						config['learning_rate'],
						config['norm'],
						data[1],
						data[2],
						data[3],
						train_loss,
						val_loss]

		with open(outfile+'.tails', 'a') as f:
			f.write('{}\n'.format('\t'.join([str(x) for x in tails])))


	return heads, tails


results_heads = []
results_tails = []

with open(outfile+'.heads', 'w') as f:
	f.write('model\tdim\tmargin\tlr\tnorm\tMR\tMRR\tH@10\ttrain_loss\tval_loss\n')
with open(outfile+'.tails', 'w') as f:
	f.write('model\tdim\tmargin\tlr\tnorm\tMR\tMRR\tH@10\ttrain_loss\tval_loss\n')

for dirname in os.listdir(root):
	# if dirname.startswith('YELP2'):
	if os.path.exists(os.path.join(root, dirname,'val_heads.tsv')):
		h, t = get_model_metrics(os.path.join(root, dirname))


