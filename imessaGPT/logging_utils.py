'''
File to house logging-related utility functions used throughout this project.
See cli/train.py for example usage.
'''
import sys
import logging
import pprint
import wandb

from imessaGPT.cli_utils import create_parser, parse_train_args


class TrainLogger:
	'''Class to log training-related data to stdout (console/file) and WandB
	'''
	AVAILABLE_LOG_LEVELS = ('quiet', 'normal', 'verbose')
	def __init__(self, args=None, level='normal'):
		# Create our logger and default training arguments
		self.logger = logging.getLogger()
		self.set_train_args(train_args=args)
		self.set_log_level(level=level)
		self.step = 0
		self.epoch = 0

	def setup(self, log_file='train_log.log'):
		'''Sets up the logger used for stdout logging (to console/file)

		Args:
			log_file (str, optional): File to log to. Defaults to 'train_log.log'.
		'''
		f_handler = logging.FileHandler(filename=log_file, mode='a')
		formatter = logging.Formatter(
			fmt='%(asctime)s - %(levelname)s - %(message)s',
			datefmt='%m/%d/%Y %H:%M:%S',
		)
		f_handler.setFormatter(formatter)
		self.logger.addHandler(f_handler)
		self.logger.addHandler(logging.StreamHandler(sys.stdout))
		self.logger.setLevel(logging.INFO)

	def set_train_args(self, train_args=None):
		'''Sets the training arguments. If None, uses defaults in cli_utils.py

		Args:
			train_args (Namespace, optional): Training CLI args. Defaults to None.
		'''
		if train_args is None:
			parent_parser = create_parser()
			self.args = parse_train_args(parents=[parent_parser])
		else:
			self.args = train_args

	def set_log_level(self, level='normal'):
		self.level = level if level in self.AVAILABLE_LOG_LEVELS else 'normal'

	def start(self, train_args=None):
		'''Begins logging and sets up WandB

		Args:
			args (Namespace): Training arguments to use for this logger
		'''
		self.set_train_args(train_args)
		pretty_args = pprint.pformat(vars(self.args), compact=True)
		self.logger.info(f'\n> Starting with arguments: \n{pretty_args}\n')
		if self.args.remote_logging:
			wandb.init(
				project=self.args.wandb_project,
				config=self.args,
				anonymous='allow',
			)

	def watch_model(self, model_to_watch):
		'''Watch a model on WandB (to track model gradients)

		Args:
			model_to_watch (torchvision model): Model to watch on WandB
		'''
		if self.args.remote_logging:
			self.logger.info('\n> Watching model on WandB')
			wandb.watch(model_to_watch)
		else:
			self.logger.info('\n> Model is NOT being watched/tracked on WandB')

	def training_overview(self, steps_per_epoch):
		'''Logs an overview of training, typically before it begins

		Args:
			steps_per_epoch (int): # of steps/epoch (train dataloader's length)
		'''
		self.logger.info('\n***** Beginning Training *****')
		self.logger.info(f'> Epochs: {self.args.num_epochs}')
		self.logger.info(f'> Steps per epoch: {steps_per_epoch}')
		self.logger.info(f'> Total steps: {self.args.max_steps}\n')

	def progress(self, step, epoch):
		'''Update our current training step and epoch

		Args:
			step (int): The current training step
			epoch (int): The current training epoch (starting at 0)
		'''
		self.step = step
		self.epoch = epoch + 1

	def log_train_metrics(self, metrics):
		'''Logs training metrics (such as loss and accuracy) to stdout/WandB

		Args:
			step (int): The current training step
			epoch (int): The current training epoch
			metrics (dict): Dictionary of metrics to log
		'''
		if self.args.remote_logging:
			wandb.log(metrics, step=self.step)
		if self.step % self.args.log_every == 0:
			self.logger.info(f'\n> (Epoch={self.epoch}, Step={self.step}) - {metrics}')

	def log_val_metrics(self, metrics):
		'''Logs validation metrics (such as loss and accuracy) to stdout/WandB

		Args:
			step (int): The current training step
			epoch (int): The current training epoch
			metrics (dict): Dictionary of metrics to log
		'''
		if self.args.remote_logging:
			wandb.log(metrics, step=self.step)
		if self.step % self.args.eval_every == 0 or self.step == self.args.max_steps:
			self.logger.info(f'\n> (Epoch={self.epoch}, Step={self.step}) - {metrics}')

	def log_model_checkpoint(self):
		'''Logs notification of model checkpoint being saved during training

		Args:
			step (int): The current training step
			epoch (int): The current training epoch
		'''
		if self.step % self.args.checkpoint_every == 0:
			self.logger.info(
				f'\n> (Epoch={self.epoch}, Step={self.step})'
				f' - Saving model checkpoint to: {self.args.output_dir}'
			)

	def training_summary(self, best_loss):
		'''Logs a summary of training, typically after training finishes

		Args:
			best_loss (float): Best (i.e., lowest) loss achieved during training
		'''
		self.logger.info('\n***** Training Complete *****')
		self.logger.info(f'> Best (lowest) loss: {best_loss}')
		self.logger.info(f'> Epochs: {self.epoch}')
		self.logger.info(f'> Total steps: {self.step}')
		self.logger.info(f'> Saving best model to: {self.args.output_dir}')

	def upload_model(self):
		'''Uploads a model to WandB

		Args:
			model_to_upload (torchvision model): Model to upload to WandB
		'''
		if self.remote_logging:
			self.logger.info('\n> Uploading model to WandB')
			wandb.save(self.args.model_file)
		else:
			self.logger.info('\n> Model is NOT being uploaded to WandB')

	def finish(self):
		'''Logs that the training script has finished successfully w/out errors
		'''
		self.logger.info('\n> Script finished successfully without errors')
		self.logger.info(f'> Final model saved in {self.args.output_dir}')
