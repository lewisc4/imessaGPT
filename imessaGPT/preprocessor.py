import numpy as np

from imessaGPT.database import ConversationDB


class ConversationPreprocessor:
	# List of possible iMessage "reactions"
	REACTIONS = (
		'Liked', 'Loved', 'Disliked',
		'Questioned', 'Emphasized', 'Laughed at',
	)
	# Strings that signify the beginning and end of the text being reacted to
	# If a message is of the form: reaction_name "text", it's a reaction
	# Ex: Liked "Cool!", means someone "liked" a message with the text "Cool!"
	REACTION_BEGINNINGS = tuple(reaction + ' “' for reaction in REACTIONS)
	REACTION_ENDING = '”'

	def __init__(self, data, sender_name, receiver_name):
		self.set_data(data)
		self.sender_name = sender_name
		self.receiver_name = receiver_name

	def set_data(self, data):
		'''Sets data to preprocess, checking if required columns are present.

		Args:
			data (pd.DataFrame): The dataframe to preprocesses.
		'''
		missing_columns = ConversationDB.BASE_SCHEMA.keys() - data.columns
		if len(missing_columns) > 0:
			raise ValueError(f'Columns missing in data: {missing_columns}')
		self.data = data

	def preprocess(self, keep_reactions=False, keep_urls=False, min_len=1):
		'''First sets the message author names, then cleans the messages.

		Args:
			keep_reactions (bool): Whether to keep iMessage "reaction" messages.
			keep_urls (bool): Whether to keep urls in messages.
			min_len (int): Minimum message length requirement (in chars).
		'''
		self.set_message_authors()
		self.clean_messages(keep_reactions, keep_urls, min_len)

	def set_message_authors(self):
		'''Sets the name of the author for each message, based on the "is_sent"
		data column. If is_sent is True, the author name is sender_name and if
		is_sent is False, the author name is receiver_name. Also sets the author
		of the next message (next row).
		'''
		author_names = np.where(
			self.data['is_sent'],
			self.sender_name,
			self.receiver_name,
		)
		# Set the author of each message and the message following it
		self.data['author'] = author_names
		self.data['next_author'] = self.data['author'].shift(-1)

	def clean_messages(self, keep_reactions, keep_urls, min_len):
		'''Cleans messages: removes empty messages, audio messages, corrupt
		messages, "reaction" messages (if specified), urls within messages
		(if specified), leading/training whitespace within messages, and
		messages that don't meet certain length requirements.
		
		Args:
			keep_reactions (bool): Whether to keep iMessage "reaction" messages.
			keep_urls (bool): Whether to keep urls in messages.
			min_len (int): Minimum message length requirement (in chars).
		'''
		self.remove_empty_messages()
		self.remove_audio_messages()
		self.remove_corrupt_messages()
		if not keep_reactions:
			self.remove_reactions()
		if not keep_urls:
			self.remove_urls()
		self.strip_messages()
		self.remove_lens_less_than(min_len)

	def remove_empty_messages(self):
		'''Remove messages flagged as being empty.
		'''
		self.data = self.data[self.data['is_empty'] == 0]

	def remove_audio_messages(self):
		'''Remove messages flagged as an audio message.
		'''
		self.data = self.data[self.data['is_audio_message'] == 0]

	def remove_corrupt_messages(self):
		'''Remove messages flagged as being corrupt.
		'''
		self.data = self.data[self.data['is_corrupt'] == 0]

	def remove_reactions(self):
		'''Remove all "reaction" messages.
		'''
		self.data = self.data[
			~(self.data.text.str.startswith(self.REACTION_BEGINNINGS)
			& self.data.text.str.endswith(self.REACTION_ENDING))
		]

	def remove_urls(self):
		'''Remove URL strings from all messages that match an 'http' pattern.
		'''
		no_urls = self.data['text'].str.replace(r'http\S+', '', regex=True)
		self.data['text'] = no_urls

	def strip_messages(self):
		'''Strip whitespace from the beginning and end of each message.
		'''
		self.data['text'] = self.data['text'].str.strip()

	def remove_lens_less_than(self, min_len=1):
		'''Remove messages that do not meet a minimum char length requirement.

		Args:
			min_len (int, optional): Min message char length. Defaults to 1.
		'''
		valid_lens = self.data['text'].str.len() >= min_len
		self.data = self.data[valid_lens]

	def save_data(self, save_file):
		'''Saves self.data (DataFrame) to a .csv file.

		Args:
			save_file (str): Name of the .csv file to save data to.
		'''
		self.data.to_csv(save_file, encoding='utf-8', index=False)
