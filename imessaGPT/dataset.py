import random
import torch
import numpy as np
import pandas as pd

from transformers import GPT2Tokenizer
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, Subset


class ConversationDataset(Dataset):
	'''Dataset (torch.utils.data.Dataset) from an iMessage conversation.
	'''
	# Special tokens to use in examples
	UNK_TOKEN = '<|unknowntext|>'
	BOS_TOKEN = '<|startoftext|>'
	EOS_TOKEN = '<|endoftext|>'
	PAD_TOKEN = '<|paddingoftext|>'

	def __init__(self, messages):
		self.messages = messages
		self.samples = []
		self.input_ids = []
		self.attention_masks = []
		self.tokenizer = GPT2Tokenizer.from_pretrained(
			'gpt2',
			unk_token=self.UNK_TOKEN,
			bos_token=self.BOS_TOKEN,
			eos_token=self.EOS_TOKEN,
			pad_token=self.PAD_TOKEN,
		)

	def __len__(self):
		'''Returns the length of the dataset (# of sets of input ids there are)
		'''
		return len(self.input_ids)

	def __getitem__(self, idx):
		'''Retuns the input ids and attention mask at a given index
		'''
		return self.input_ids[idx], self.attention_masks[idx]

	def build_samples(self):
		'''Build samples for the dataset. Each sample is built sequentially, by
		getting all messages sent by one author and all messages sent after them
		by the other author.
		'''
		curr_sample = ''
		for row in tqdm(self.messages.itertuples(), desc='Building Samples'):
			curr_sample += f'\n{row.author}: {row.text}\n'
			if row.author != row.next_author:
				self.samples.append(f'{self.BOS_TOKEN}{curr_sample}')
				if len(self.samples) > 1:
					self.samples[-2] += curr_sample + self.EOS_TOKEN
				if pd.isna(row.next_author):
					self.samples[-1] += self.EOS_TOKEN
				curr_sample = ''

	def show_examples(self, n_examples=2, sep='-'):
		'''Show example samples from the dataset.

		Args:
			n_examples (int, optional): Num samples to show. Defaults to 2.
			sep (str, optional): Char to separate samples. Defaults to '-'.
		'''
		sep *= 100
		for i, ex in enumerate(random.sample(self.samples, n_examples)):
			print(f'{sep}\nExample #{i + 1}:\n{ex}\n{sep}\n')

	def tokenize_samples(self, truncate=True, max_len=512, pad_type='max_length'):
		'''Tokenizes the dataset samples using a GPT2 tokenizer.

		Args:
			truncate (bool, optional): Truncate samples or not. Defaults to True.
			max_len (int, optional): Max sample len. Defaults to 512.
			pad_type (str, optional): Type of padding. Defaults to 'max_length'.
		'''
		# For each sample, tokenize it and get its input ids and attention mask
		for sample in tqdm(self.samples, desc='Tokenizing'):
			tokens = self.tokenizer(
				sample,
				truncation=truncate,
				max_length=max_len,
				padding=pad_type,
			)
			self.input_ids.append(torch.tensor(tokens['input_ids']))
			self.attention_masks.append(torch.tensor(tokens['attention_mask']))

	def train_val_split(self, percent_val=0.2, seed=None):
		'''Splits self into training and validation subsets.

		Args:
			percent_val (float, optional): Percentage of data for validation.
			seed (int, optional): Seed for random split reproducibility.

		Returns:
			tuple: The (train, val) subsets.
		'''
		indices = list(range(len(self)))
		train_indices, val_indices = train_test_split(
			indices,
			test_size=percent_val,
			random_state=seed,
		)
		return Subset(self, train_indices), Subset(self, val_indices)

	def pad_batch(self, batch):
		'''Pads a batch of data (e.g., in a DataLoader for collation).

		Args:
			batch (list): List of (input_ids, attention_mask) tensor pairs.

		Returns:
			dict: Padded versions of input_ids and attention_mask.
		'''
		unzipped = list(map(list, zip(*batch)))
		input_ids, attention_mask = unzipped[0], unzipped[1]
		return self.pad(input_ids), self.pad(attention_mask)

	def pad(self, sequence_list):
		'''Pads a batch of sequences with our tokenizer's pad_token_id so that
		all sequences are the same length as the longest sequence in the batch.

		Args:
			sequence_list (list): Batch of sequences (e.g., token ids) to pad.

		Returns:
			torch.LongTensor: Tensor of sequences padded with pad_token_id.
		'''
		sequence_lens = [len(sequence) for sequence in sequence_list]
		max_sequence_len = max(sequence_lens)
		pad_id = self.tokenizer.pad_token_id
		padded_sequence_list = []
		for sequence, sequence_len in zip(sequence_list, sequence_lens):
			if isinstance(sequence, (torch.Tensor, np.ndarray)):
				sequence = sequence.tolist()
			padding = [pad_id] * (max_sequence_len - sequence_len)
			padded_sequence_list.append(sequence + padding)
		return torch.LongTensor(padded_sequence_list)
