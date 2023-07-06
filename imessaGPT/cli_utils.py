import os
import copy
import argparse
import torch

from pathlib import Path
from os.path import join, splitext


# The CWD of the script importing this module
CWD = Path().resolve()
OUTPUT_DIR = CWD / 'outputs'

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
		add_help=False, # Don't add help, only because this is a parent parser
	)

	# File path arguments used throughout CLI scripts
	parser.add_argument(
		'--chat_file',
		type=str,
		default=str(CHAT_DB_FILE),
		help='Path to the the chat.db file that contains the iMessages.',
	)
	parser.add_argument(
		'--dataset_file',
		type=str,
		default=str(DATASET_FILE),
		help='Path to the (.csv) file that contains the processed dataset.',
	)
	parser.add_argument(
		'--output_dir',
		type=str,
		default=str(OUTPUT_DIR),
		help='Path to the directory where script outputs will be saved.',
	)
	parser.add_argument(
		'--seed',
		type=int,
		default=7,
		help='Seed to use for randomness.',
	)
	parser.add_argument(
		'--phone_number',
		type=str,
		default='+18889374895',
		help='The phone number to use for iMessage conversations.'
	)
	parser.add_argument(
		'--sender',
		type=str,
		default='SENDER',
		help='Name of person sending the messages (i.e., your name).',
	)
	parser.add_argument(
		'--receiver',
		type=str,
		default='RECEIVER',
		help='Name of person receiving the messages (i.e., person with phone_number).',
	)
	# Return parser to use as a parent, so we DON'T want to call parse_args()
	return parser


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
		description='Train a model on a network/graph of reddit comment data',
	)

	parser.add_argument(
		'--model_file',
		type=str,
		default='conversational_model.pt',
		help='The name of the model (.pt) file to save to/load from',
	)
	parser.add_argument(
		'--percent_val',
		type=float,
		default=0.15,
		help='Percentage of the data to use for validation (train_val_size * percent_train).',
	)
	parser.add_argument(
		'--batch_size',
		type=int,
		default=1,
		help='Batch size (per device) for each DataLoader.',
	)
	parser.add_argument(
		'--num_workers',
		type=int,
		default=0,
		help='Number of workers to use for each DataLoader.',
	)
	parser.add_argument(
		'--pin_memory',
		default=False,
		action='store_true',
		help='Controls if each DataLoader uses pinned memory.',
	)

	# Training arguments
	parser.add_argument(
		'--device',
		default='cuda' if torch.cuda.is_available() else 'cpu',
		help='Device (cuda or cpu) on which the code should run',
	)
	parser.add_argument(
		'--non_blocking',
		default=False,
		action='store_true',
		help='Controls if non-blocking transfers are used in .to(device) for batch data.',
	)
	parser.add_argument(
		'--model_location',
		type=str,
		default='trained_model',
		help='The model name to use for saving and loading a model.'
	)
	parser.add_argument(
		'--learning_rate',
		type=float,
		default=5e-4,
		help='Initial learning rate (after the potential warmup period) to use.',
	)
	parser.add_argument(
		'--weight_decay',
		type=float,
		default=0.0,
		help='Weight decay to use.',
	)
	parser.add_argument(
		'--num_epochs',
		type=int,
		default=5,
		help='Total number of training epochs to perform.',
	)
	parser.add_argument(
		'--eval_every_steps',
		type=int,
		default=40,
		help='Perform evaluation every n network updates.',
	)
	parser.add_argument(
		'--lr_scheduler_type',
		type=str,
		default='linear',
		choices=[
			'linear', 'cosine', 'cosine_with_restarts',
			'polynomial', 'constant', 'constant_with_warmup',
		],
		help='The learning rate scheduler type to use.',
	)
	parser.add_argument(
		'--max_steps',
		type=int,
		default=None,
		help='Number of training steps to perform. If provided, overrides num_epochs.',
	)
	parser.add_argument(
		'--num_warmup_steps',
		type=int,
		default=0,
		help='Number of steps for the warmup in the lr scheduler.'
	)
	parser.add_argument(
		'--log_every',
		type=int,
		default=10,
		help='How often, in # of training steps, to log training metrics.'
	)
	parser.add_argument(
		'--eval_every',
		type=int,
		default=20,
		help='How often, in # of training steps, to evaluate the model.',
	)
	parser.add_argument(
		'--checkpoint_every',
		type=int,
		default=50,
		help='How often, in # of training steps, save a model checkpoint.',
	)
	parser.add_argument(
		'--remote_logging',
		default=False,
		action='store_true',
		help='Whether to log data remotely (to WandB) or not.',
	)
	parser.add_argument(
		'--wandb_project',
		type=str,
		default='imessaGPT',
		help='The name of the wandb project to remotely log metrics to',
	)
	parser.add_argument(
		'--upload_model',
		default=False,
		action='store_true',
		help='Whether to remotely upload the trained model (to WandB) or not.',
	)

	args = parser.parse_args()
	valid_args = validate_train_args(args)
	return valid_args


def validate_train_args(args):
	'''Validates/updates the CLI arguments parsed in the parse_args function,
	defined in this file.

	Args:
		args (Namespace): The CLI arguments to validate
	'''
	valid = copy.deepcopy(args)
	# Check if output dir (where models are saved) exists, if not create it
	os.makedirs(valid.output_dir, exist_ok=True)
	# Update file names to use the valid paths
	# Also, strip the last extension (if it exists) and add a valid extension
	valid.model_file = join(
		valid.output_dir,
		splitext(valid.model_file)[0] + '.pt',
	)
	return valid
