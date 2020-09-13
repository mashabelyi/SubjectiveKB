import os, pickle
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau
from torch.nn import MarginRankingLoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Trainer():
	def __init__(self, model, train, val, test, optimizer, criterion, logger, loader, config):
		""" Init

		Inputs

		MODEL (obj)		instantiated model object
		TRAIN (array)	training samples
		VAL (array)		validation samples
		TEST (array)	test samples
		OPTIMIZER 		instantiated torch optimizer object
		CRITERION 		instantiated torch loss object. Should return 'sum' of loss in batch
		LOGGER 			instance of Logger class
		LOADER 			instance of BatchLoader class
		CONFIG 			

		"""
		self.model = model.to(device)
		self.traindata = train
		self.valdata = val
		self.testdata = test
		self.optimizer = optimizer
		self.criterion = criterion.to(device)
		self.logger = logger
		self.loader = loader
		self.config = config


	def load_checkpoint(self, checkpoint):
		""" Load model weights from checkpoint
		
		Inputs

		CHECKPOINT (string) 	path to pcikle file with saved weights

		"""
		self.model.load_state_dict(torch.load(checkpoint))

	def train(self, epochs, tolerance=1e-8, patience=5):
		""" Train model
		
		Inputs

		EPOCHS (int) 		max number of epochs to train
		TOLERANCE (float) 	minimal decrease in validation loss tolerated
		PATIENCE (int)		if validation loss does not decrease by at least TOLERANCE in PATENCE epochs, training will stop early

		"""
		
		# grad_accum_step = 5
		# _grad_accum = grad_accum_step

		_patience = patience
		_epoch = 0
		best_val_loss = float("inf")

		scheduler=ReduceLROnPlateau(self.optimizer, factor=0.5, patience=5, threshold=1e-4, verbose=True)

		while _epoch < epochs and _patience > 0:
			
			self.model.train()

			epoch_loss = 0
			batch = 0
			self.loader.reset()

			# iterate through batches
			while self.loader.has_next():
				batch += 1
				print("epoch {} batch {}".format(_epoch, batch), end='\r')
				
				# fetch next batch of training samples 
				pos, neg = self.loader.next_batch()

				# do any normalization before each mini-batch
				self.model.norm_step()

				# run trainins samples through network
				# posScore, negScore = self.model(pos[:,0], pos[:,1], pos[:,2], pos[:,3], 
				# 					neg[:,0], neg[:,1], neg[:,2], neg[:,3])
				# calculate batch loss
				# tmpTensor = torch.tensor([-1], dtype=torch.float).to(device)
				# batch_loss = self.criterion(posScore, negScore, tmpTensor)
				# epoch_loss += batch_loss


				# new implementation - forward pass returns batch loss
				batch_loss = self.model(pos[:,0], pos[:,1], pos[:,2], pos[:,3], 
									neg[:,0], neg[:,1], neg[:,2], neg[:,3])
				epoch_loss += batch_loss


				# backpropagate
				batch_loss.backward()
				self.optimizer.step()

				# reset gradients 
				#   (we do gradient accumulation within each batch, but need 
				#   to reset the gradients at the beginning of each batch)
				self.optimizer.zero_grad() 

				


			# Average epoch loss over all training samples 
			#     -- assume self.criterion uses reduction='sum'
			epoch_loss = epoch_loss.item()#/len(self.traindata)
			
			# Calculate Validation loss at the end of each epoch
			val_loss = self._val_loss()

			# update learning rate if needed
			scheduler.step(val_loss) 

			# if improvement is not large enough:
			if (best_val_loss - val_loss) < tolerance:
				_patience -= 1


			# Save model checkpoint if best validation loss is achieved
			if val_loss < best_val_loss:
				torch.save(self.model.state_dict(), os.path.join(self.config['name'], 'best_val_loss_state_dict.pt'))
				best_val_loss = val_loss
				_patience = patience
			

			# Print feedback
			print("epoch {} - loss: {:.8}, val_loss: {:.8}, patience: {}".format(_epoch, epoch_loss, val_loss, _patience))
			self.logger.log(_epoch, {'loss':epoch_loss, 'val_loss':val_loss})

			_epoch += 1


	def _val_loss(self):
		"""Calculate loss on validation set"""
		self.model.eval()

		# with torch.no_grad():
		val_loss = 0
		pos_batches, neg_batches = self.loader.validaton_batches(self.valdata)
		for pos, neg in zip(pos_batches, neg_batches):
			# run through network
			_loss = self.model(pos[:,0], pos[:,1], pos[:,2], pos[:,3], 
										neg[:,0], neg[:,1], neg[:,2], neg[:,3])
			val_loss += _loss

			# tmpTensor = torch.tensor([-1], dtype=torch.float).to(device)
			# val_loss += self.criterion(posScore, negScore, tmpTensor)

		# return mean loss over all validation samples
		return val_loss.item()#/len(self.valdata)


	def _report_metrics(self, MR, MRR, h10):
		print("\n\nEVALUATION RESULTS")
		print("====================================")
		print("MR = {}, MRR = {:.4}, H@10 = {:.4}\n".format(round(MR), MRR, h10))

	def _log_metrics(self, MR, MRR, h10, sources, ranks, file):
		with open(file, 'w') as f:
			f.write("id\tMR\tMRR\tH@10\n")
			f.write("total\t{}\t{}\t{}\n".format(MR, MRR, h10))

			for srcId, metrics in sources.items():
				f.write("{}\t{}\t{}\t{}\n".format(srcId, metrics['MR'], metrics['MRR'], metrics['H10']))

		with open(file+'.ranks', 'w') as f:
			f.write("\n".join([str(x)[1:-1] for x in ranks]))


	def eval_raw(self, samples, logfile=None):
		self.load_checkpoint(os.path.join(self.config['name'], 'best_val_loss_state_dict.pt'))
		self.model.eval()

		
		MR, MRR, h10 = self.model.eval_raw(samples)

		self._report_metrics(MR, MRR, h10)
		if logfile:
			self._log_metrics(MR, MRR, h10, logfile)
		

	def eval_filt(self, samples_dict, logfile=None):

		self.load_checkpoint(os.path.join(self.config['name'], 'best_val_loss_state_dict.pt'))
		self.model.eval()
		
		# with torch.no_grad():
		MR, MRR, h10, source_metrics, ranks = self.model.eval_filt(samples_dict)

		self._report_metrics(MR, MRR, h10)
		
		if logfile:
			self._log_metrics(MR, MRR, h10, source_metrics, ranks, logfile)

	def eval(self, iterator, n, logfile=None):
		self.load_checkpoint(os.path.join(self.config['name'], 'best_val_loss_state_dict.pt'))
		self.model.eval()

		MR, MRR, h10, source_metrics, ranks = self.model.eval_run(iterator, n)

		if logfile:
			self._log_metrics(MR, MRR, h10, source_metrics, ranks, logfile)

