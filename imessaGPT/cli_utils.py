import argparse

from pathlib import Path


# The CWD of the script importing this module
CWD = Path().resolve()

# The "root" data dir, containing all project resource files (by default)
CHAT_DB_FILE = CWD / 'chat.db'
DATASET_FILE = CWD / 'dataset.csv'


def create_parser():
	'''
	This function creates a default, general argument parser to act as a 
	parent for scripts in this project that are run from the CLI.
	This function is used in preprocess.py, train.py, and evaluate.py.
	Default arguments have the meaning of being a reasonable default value.
	'''

	# General-purpose parser to use for different scripts
	# Not intended to be used as a standalone parser in this project
	parser = argparse.ArgumentParser(
		description='General-use parser for CLI scripts in this project',
		add_help=False # Don't add help, only because this is a parent parser
	)

	# File path arguments used throughout CLI scripts
	parser.add_argument(
		'--chat_file',
		type=Path,
		default=CHAT_DB_FILE,
		help='Path to the the chat.db file that contains the iMessages.',
	)
	parser.add_argument(
		'--dataset_file',
		type=Path,
		default=DATASET_FILE,
		help='Path to the (.csv) file that contains the processed dataset.',
	)
	# Return parser to use as a parent, so we DON'T want to call parse_args()
	return parser


def parse_preprocessing_args(parents=[]):
	'''
	This function creates a preprocessing-related argument parser and parses a
	(preprocessing) script's input arguments.

	Default arguments have the meaning of being a reasonable default value.
	To change the parameters, pass them to the script. For example, assuming:
	we have a parent parser with a dataset_dir argument:

	python3 preprocess.py --dataset_dir=dataset --phone_number=+18009134187
	'''

	# Create parser based on the parent parser(s)
	# An empty list ([]) is equivalent to no parents
	parser = argparse.ArgumentParser(
		parents=parents,
		description='Pre-process a set of iMessages from a chat.db file'
	)

	# Data preprocessing-related parameters
	parser.add_argument(
		'--phone_number',
		type=str,
		default='+18889374895',
		help='The phone number to use for iMessage conversations.'
	)

	args = parser.parse_args()
	return args


def parse_train_args(parents=[]):
	'''
	This function creates a training-related argument parser and parses a
	(training) script's input arguments.

	Default arguments have the meaning of being a reasonable default value.
	To change the parameters, pass them to the script. For example, assuming:
	we have a parent parser with a processed_dir argument:

	python3 train.py --dataset_dir=dataset --learning_rate=2e-3
	'''

	# Create parser based on the parent parser(s)
	# An empty list ([]) is equivalent to no parents
	parser = argparse.ArgumentParser(
		parents=parents,
		description='Train a model on a network/graph of reddit comment data'
	)

	parser.add_argument(
		'--train_val_size',
		type=int,
		default=None,
		help='Combined size (# samples) of the training and validation set.',
	)
	parser.add_argument(
		'--test_size',
		type=int,
		default=None, # None implies a separate test set will be used
		help='Size (# samples) of the test set.',
	)
	parser.add_argument(
		'--percent_train',
		type=float,
		default=0.8,
		help='Percentage of the data to use for training (train_val_size * percent_train).',
	)
	parser.add_argument(
		'--shuffle_generator',
		type=bool,
		default=True,
		help='Whether to shuffle training data.',
	)
	parser.add_argument(
		'--shuffle_train_data',
		type=bool,
		default=True,
		help='Whether to shuffle training data.',
	)

	# Training arguments
	parser.add_argument(
		'--model_location',
		type=str,
		default='trained_model',
		help='The model name to use for saving and loading a model.'
	)
	parser.add_argument(
		'--batch_size',
		type=int,
		default=256,
		help='Batch size (per device) for the training dataloader.',
	)
	parser.add_argument(
		'--learning_rate',
		type=float,
		default=5e-4,
		help='Initial learning rate (after the potential warmup period) to use.',
	)
	parser.add_argument(
		'--num_train_epochs',
		type=int,
		default=5,
		help='Total number of training epochs to perform.',
	)
	parser.add_argument(
		'--steps_per_epoch',
		type=int,
		default=540,
		help='Number of (update) steps to take during each training epoch.',
	)
	parser.add_argument(
		'--eval_every_steps',
		type=int,
		default=40,
		help='Perform evaluation every n network updates.',
	)
	parser.add_argument(
		'--dropout',
		type=float,
		default=0.5,
		help='Dropout rate to use during training.',
	)
	parser.add_argument(
		'--num_layers',
		type=int,
		default=2,
		help='Number of hidden layers to use during training.',
	)
	parser.add_argument(
		'--layer_size',
		type=int,
		default=16,
		help='Size of hidden layers to use during training.',
	)
	parser.add_argument(
		'--hidden_activation',
		type=str,
		default='relu',
		help='Activation function to use for hidden layers.',
	)
	parser.add_argument(
		'--final_activation',
		type=str,
		default='softmax',
		help='Final activation function to use at final layer.',
	)
	parser.add_argument(
		'--num_attn_heads',
		type=int,
		default=8,
		help='Number of attention heads to use during training.',
	)
	parser.add_argument(
		'--attn_dropout',
		type=float,
		default=0.5,
		help='Dropout rate to use for attention during training.',
	)
	parser.add_argument(
		'--use_bias',
		type=bool,
		default=True,
		help='Whether to use bias during training or not.',
	)
	parser.add_argument(
		'--layer_num_samples',
		'--list',
		type=lambda s: [int(num_samples) for num_samples in s.split(',')],
		default=[30, 10],
		help='List of comma separated number of samples to use at each layer.'
	)

	args = parser.parse_args()
	return args

