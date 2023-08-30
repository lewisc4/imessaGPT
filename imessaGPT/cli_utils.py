import os
import copy
import argparse
import torch

from pathlib import Path
from os.path import join, splitext


# The CWD of the script importing this module
CWD = Path().resolve()
# Default directory that houses all raw/processed training data
DATA_DIR = CWD / 'data'
# Default directory that houses all model outputs
MODEL_DIR = CWD / 'model_outputs'

# Use iMessage DB as the default source of raw, unprocessed message data
RAW_FILE = DATA_DIR / 'chat.db'
MESSAGE_COLUMN = 'text'
# Default file to save preprocessed/cleaned messages to
PROCESSED_FILE = DATA_DIR / 'dataset.csv'
# Default file to use as a dataset for training
DATASET_FILE = DATA_DIR / 'dataset.csv'


def parse_preprocess_args():
	'''
	This function creates a preprocessing-related argument parser and parses a
	(preprocessing) script's input arguments.

	Default arguments have the meaning of being a reasonable default value.
	To change the parameters, pass them to the script. For example:

	python3 train.py --data_dir=data --phone_number=+18889990000
	'''
	parser = argparse.ArgumentParser('Preprocess conversation data for GPT-2.')
	parser.add_argument(
		'--data_dir',
		type=str,
		default=str(DATA_DIR),
		help='Path to the directory that houses all raw/processed data files.',
	)
	parser.add_argument(
		'--raw_file',
		type=str,
		default=str(RAW_FILE),
		help='Path to the file with the initial, unprocessed message data.',
	)
	parser.add_argument(
		'--message_col',
		type=str,
		default=MESSAGE_COLUMN,
		help='Name of the column in raw_file that stores message text.',
	)
	parser.add_argument(
		'--processed_file',
		type=str,
		default=str(PROCESSED_FILE),
		help='Path to the .csv file with the processed data (from --raw_file).',
	)
	parser.add_argument(
		'--phone_number',
		type=str,
		default='+18889990000',
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
	parser.add_argument(
		'--keep_reactions',
		default=False,
		action='store_true',
		help='Whether to keep "reaction" iMessages or not when cleaning.',
	)
	parser.add_argument(
		'--keep_urls',
		default=False,
		action='store_true',
		help='Whether to keep urls in messages or not when cleaning.',
	)
	parser.add_argument(
		'--min_len',
		type=int,
		default=1,
		help='Minimum length (in chars) of messages to keep when cleaning.',
	)
	# Parse and validate the provided arguments
	args = parser.parse_args()
	valid_args = validate_preprocess_args(args)
	return valid_args


def validate_preprocess_args(args):
	'''Validates/updates the CLI arguments parsed in the parse_args function,
	defined in this file.

	Args:
		args (Namespace): The CLI arguments to validate
	'''
	valid = copy.deepcopy(args)
	# Create data dir if it doesn't exist (where raw/processed data is stored)
	os.makedirs(valid.data_dir, exist_ok=True)
	# Update file names to use the valid paths
	valid.raw_file = join(valid.data_dir, valid.raw_file)
	valid.processed_file = join(valid.data_dir, valid.processed_file)
	return valid


def parse_train_args():
	'''
	This function creates a training-related argument parser and parses a
	(training) script's input arguments.

	Default arguments have the meaning of being a reasonable default value.
	To change the parameters, pass them to the script. For example:

	python3 train.py --data_dir=data --learning_rate=2e-3
	'''
	parser = argparse.ArgumentParser('Train a GPT-2 model on conversation data.')
	parser.add_argument(
		'--data_dir',
		type=str,
		default=str(DATA_DIR),
		help='Path to the directory that houses all raw/processed data files.',
	)
	parser.add_argument(
		'--dataset',
		type=str,
		default=str(DATASET_FILE),
		help='Path to the dataset file to use for training.',
	)
	parser.add_argument(
		'--model_dir',
		type=str,
		default=str(MODEL_DIR),
		help='Path to the directory where model outputs will be saved.',
	)
	parser.add_argument(
		'--model_file',
		type=str,
		default='conversational_model.pt',
		help='The name of the model (.pt) file to save to/load from.',
	)
	parser.add_argument(
		'--num_examples',
		type=int,
		default=None,
		help='Number of dataset examples to use. None means use ALL examples.',
	)
	parser.add_argument(
		'--percent_val',
		type=float,
		default=0.15,
		help='Percentage of the data to use for validation (train_val_size * percent_train).',
	)
	parser.add_argument(
		'--seed',
		type=int,
		default=7,
		help='Seed to use for randomness (in train/val/test splits).',
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
		help='Device (cuda or cpu) on which the code should run.',
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
	# Parse and validate the provided arguments
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
	valid.dataset = join(valid.data_dir, valid.dataset)
	# Create model directory if it doesn't exist (where model outputs are saved)
	os.makedirs(valid.model_dir, exist_ok=True)
	# Update file names to use the valid paths
	# Also, strip the last extension (if it exists) and add a valid extension
	valid.model_file = join(
		valid.model_dir,
		splitext(valid.model_file)[0] + '.pt',
	)
	return valid
