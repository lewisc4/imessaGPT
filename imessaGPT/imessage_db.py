import sqlite3
import pandas as pd


class ChatDB:
	'''Chat database for iMessages. Has queries for message metadata, such as
	a message's text, date sent, sender/receiver phone #, etc. Also has queries
	to retrieve entire message conversations/threads (via the "handles" table)
	and to join the message threads with the message metadata. The chat.db file
	is typically found (on Macs) at: ~/Library/Messages/chat.db
	'''
	# Chat database SQL query strings
	GET_ALL_MESSAGES = "SELECT * FROM message"
	GET_ALL_HANDLES = "SELECT * FROM handle"
	GET_MESSAGE_JOINS = "SELECT * FROM chat_message_join"
	GET_ALL_TABLES = "SELECT name FROM sqlite_master WHERE type = 'table'"
	# Chat database table mappings to rename columns: {original_name: new_name}
	MESSAGE_COL_MAP = {'ROWID': 'message_id'}
	HANDLE_COL_MAP = {'id': 'phone_number', 'ROWID': 'handle_id'}
	# Chat database SQL table column names/headers
	MESSAGE_COLS = [
		'text', 'handle_id', 'date', 'is_sent', 'message_id', 'is_empty',
		'is_spam', 'is_corrupt', 'is_audio_message', 'guid', 'reply_to_guid',
	]
	HANDLE_COLS = ['handle_id', 'phone_number']

	def __init__(self, db_file):
		'''Create object to represent an iMessage chat database.

		Args:
			db_file (str): The path to the chat.db file
		'''
		# Connect to the database and establish a cursor to fetch data from it
		self.db_file = db_file
		self.conn = sqlite3.connect(self.db_file)
		self.cur = self.conn.cursor()
		# Query the database to get all relevant message/conversation data
		self.all_messages = self.get_all_messages()
		self.all_handles = self.get_all_handles()
		self.all_conversations = self.get_all_conversations()
		self.all_table_names = self.get_all_tables()

	def get_all_messages(self):
		'''Get all messages as a pandas DF, with MESSAGE_COLS as the columns.

		Returns:
			pandas.DataFrame: DataFrame (i.e., table) of message data 
		'''
		messages = pd.read_sql_query(self.GET_ALL_MESSAGES, self.conn)
		return messages.rename(columns=self.MESSAGE_COL_MAP)

	def get_all_handles(self):
		'''Get all handles (conversation/message threads) as a pandas DF.

		Returns:
			pandas.DataFrame: DataFrame (i.e., table) of handle data 
		'''
		handles = pd.read_sql_query(self.GET_ALL_HANDLES, self.conn)
		return handles.rename(columns=self.HANDLE_COL_MAP)

	def get_all_conversations(self):
		'''Get all conversations (i.e., message data joined with each handle).

		Returns:
			pandas.DataFrame: DataFrame (i.e., table) of all conversations
		'''
		# Merge the message and handle DFs so they can be joined
		message_handles = pd.merge(
			self.all_messages[self.MESSAGE_COLS],
			self.all_handles[self.HANDLE_COLS],
			on='handle_id',
			how='left',
		)
		# Join the messages with the handles
		message_joins = pd.read_sql_query(self.GET_MESSAGE_JOINS, self.conn)
		conversations = pd.merge(
			message_handles,
			message_joins[['chat_id', 'message_id']],
			on='message_id',
			how='left',
		)
		return conversations

	def get_all_tables(self):
		'''Get a list of all available tables in the chat database.

		Returns:
			list: The list of available table names as strings
		'''
		self.cur.execute(self.GET_ALL_TABLES)
		return [table_name for table_name in self.cur.fetchall()]


class Conversation(ChatDB):
	'''A single message conversation/thread in the iMessage chat database.
	This is a conversation between two people, with one of them being yourself.
	Has functions to apply to each message's text in the conversation, such as
	removing empty messages, removing URLs, setting sender/receiver names, etc.
	'''
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

	def __init__(self, db_file, phone_number, sender, receiver):
		super().__init__(db_file)
		# Phone # of the person you are conversing with (i.e., the receiver)
		self.phone_number = phone_number
		# Desired names for the sender (yourself) and the receiver
		self.sender = sender
		self.receiver = receiver
		# Get messages only for this conversation and set sender/receiver names
		self.messages = self.get_conversation_messages()
		self.set_authors()

	def get_conversation_messages(self):
		'''Get all conversation messages between yourself and phone_number.

		Returns:
			pd.DataFrame: Messages between yourself and phone_number
		'''
		matches = self.all_conversations['phone_number'] == self.phone_number
		return self.all_conversations[matches]

	def set_authors(self):
		'''Set current and next message author names based on sender/receiver.
		'''
		is_sent = self.messages['is_sent'].tolist()
		authors = [self.sender if sent else self.receiver for sent in is_sent]
		self.messages['author'] = authors
		self.messages['next_author'] = self.messages['author'].shift(-1)

	def clean_messages(self, min_len=1, keep_urls=True, keep_reactions=False):
		'''Clean the text of each message in the conversation.

		Args:
			min_len (int, optional): Min message length to keep. Defaults to 1.
			keep_urls (bool, optional): Keep URLs or not. Defaults to True.
			keep_emojis (bool, optional): Keep emojis or not. Defaults to True.
			keep_reactions (bool, optional): Keep reactions or not. Defaults to False.
		'''
		self.remove_empty_messages()
		self.remove_audio_messages()
		self.remove_corrupt_messages()
		if not keep_urls:
			self.remove_urls()
		if not keep_reactions:
			self.remove_reactions()
		self.strip_messages()
		self.remove_len_less_than(min_len)

	def remove_empty_messages(self):
		'''Remove messages flagged as being empty.
		'''
		self.messages = self.messages[self.messages['is_empty'] == 0]

	def remove_audio_messages(self):
		'''Remove messages flagged as an audio message.
		'''
		self.messages = self.messages[self.messages['is_audio_message'] == 0]

	def remove_corrupt_messages(self):
		'''Remove messages flagged as being corrupt.
		'''
		self.messages = self.messages[self.messages['is_corrupt'] == 0]

	def strip_messages(self):
		'''Strip whitespace from the beginning and end of each message.
		'''
		self.messages['text'] = self.messages['text'].str.strip()

	def remove_urls(self):
		'''Remove URL strings from all messages that match an 'http' pattern.
		'''
		no_urls = self.messages['text'].str.replace(r'http\S+', '', regex=True)
		self.messages['text'] = no_urls
	
	def remove_reactions(self):
		'''Remove all "reaction" messages.
		'''
		self.messages = self.messages[
			~(self.messages.text.str.startswith(self.REACTION_BEGINNINGS)
			& self.messages.text.str.endswith(self.REACTION_ENDING))
		]

	def remove_len_less_than(self, min_len=1):
		'''Remove messages that do not meet a minimum char length requirement.

		Args:
			min_len (int, optional): Min message char length. Defaults to 1.
		'''
		valid_lens = self.messages['text'].str.len() >= min_len
		self.messages = self.messages[valid_lens]
