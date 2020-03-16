import os

class Logger():
	def __init__(self, model_dir, metrics):
		self.dir = model_dir
		self.file = os.path.join(self.dir, 'training_log.csv')
		self.metrics = metrics

		with open(self.file, 'w') as f:
			f.write('epoch,{}\n'.format(','.join(self.metrics)))

	def log(self, epoch, data):
		towrite = [str(epoch)]
		for m in self.metrics:
			towrite.append(str(data[m]) if m in data else '')

		with open(self.file, 'a') as f:
			f.write('{}\n'.format(','.join(towrite)))



