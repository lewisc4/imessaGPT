import sqlite3
import pandas as pd

from pathlib import Path
from abc import ABC, abstractmethod


class ConversationDB(ABC):
	'''Boilerplate class to act as a parent for different types
	(.csv, .db, etc.) of conversation databases
	'''
	DB_FILE_EXTENSION = None
	# SCHEMA_MESSAGE_COL is the column in the schema that stores message text.
	# All other columns in the schema describe SCHEMA_MESSAGE_COL.
	SCHEMA_MESSAGE_COL = 'text'
	# Default database column names and the default functions to populate them:
	BASE_SCHEMA = {
		# This column is always just the messages themselves
		SCHEMA_MESSAGE_COL: lambda messages: messages,
		# If message is from sender or receiver. Alternates b/w both by default.
		'is_sent': lambda messages: messages.index % 2 == 0,
		# If message is empty or not. Checks if message length is 0 by default.
		'is_empty': lambda messages: messages.astype(str).str.len() == 0,
		# If message is an audio message. False by default.
		'is_audio_message': lambda messages: False,
		# If message is corrupt. False by default.
		'is_corrupt': lambda messages: False,
	}

	def __init__(self, db_file, message_col):
		self.set_db_file(db_file)
		# Column in the database file that stores the conversation's messages
		self.message_col = message_col
		# Initial data is an empty DataFrame, adapted to BASE_SCHEMA
		self.data = self.adapt_to_schema(pd.DataFrame({self.message_col: []}))

	def set_db_file(self, db_file):
		'''Sets source file to read from, ensuring it's of the correct type.
		'''
		if Path(db_file).suffix != self.DB_FILE_EXTENSION:
			raise ValueError(f'DB file must be a {self.DB_FILE_EXTENSION} file')
		self.db_file = db_file

	def populate_data(self):
		'''Reads db_file into a DataFrame and adapts it to BASE_SCHEMA.

		Returns:
			pd.DataFrame: DataFrame from db_file, adapted to BASE_SCHEMA.
		'''
		df_from_file = self.read_file()
		self.data = self.adapt_to_schema(df_from_file)

	@abstractmethod
	def read_file(self):
		'''Reads db_file as a DataFrame and returns it (implement in children).
		'''
		pass

	def adapt_to_schema(self, df):
		'''Adapts a DataFrame to conform to BASE_SCHEMA. Any columns in
		BASE_SCHEMA that aren't in df are added to df with default values.

		Args:
			df (pd.DataFrame): The DataFrame to adapt.

		Returns:
			pd.DataFrame: The adapted DataFrame.
		'''
		# Ensure the DataFrame has message_col as a column
		if self.message_col not in df.columns:
			raise ValueError(f'{self.db_file} missing: {self.message_col}')
		# Rename the message column in the file to match our schema
		df = df.rename(columns={self.message_col: self.SCHEMA_MESSAGE_COL})
		# Add any missing columns using their default population function
		for column, default_func in self.BASE_SCHEMA.items():
			if column not in df.columns:
				df[column] = default_func(df[self.SCHEMA_MESSAGE_COL])
		return df


class CSVConversationDB(ConversationDB):
	'''Class to read/store conversations stored in a .csv file.
	'''
	# Files read by this class MUST be .csv files (obviously)
	DB_FILE_EXTENSION = '.csv'

	def __init__(self, db_file, message_col):
		super().__init__(db_file, message_col)

	def read_file(self):
		'''Read conversations from a .csv file into a DataFrame.
		'''
		return pd.read_csv(self.db_file).dropna(axis=1, how='all')


class iMessageConversationDB(ConversationDB):
	'''Class to read/store conversations stored in an iMessage .db file.
	'''
	# Files read by this class MUST be .db files (e.g., "chat.db")
	DB_FILE_EXTENSION = '.db'
	# Query to get all messages in the chat.db file and their relevant fields
	MESSAGES_QUERY = '''SELECT ROWID as message_id, text, handle_id, date,
	is_sent, is_empty, is_spam, is_corrupt, is_audio_message, guid, reply_to_guid
	FROM message'''
	# Query to get each handle/thread/conversation between the "owner" of
	# chat.db (presumably you) and other people, identified by their phone #
	HANDLES_QUERY = '''SELECT ROWID as handle_id, id as phone_number
	FROM handle'''
	# Query to get the chat/thread/conversation that each message belongs to
	CHAT_MESSAGES_QUERY = '''SELECT chat_id, message_id
	FROM chat_message_join'''

	def __init__(self, db_file, message_col, phone_number):
		super().__init__(db_file, message_col)
		self.phone_number = phone_number
		# Connect to the database and establish a cursor to fetch data from it
		self.conn = sqlite3.connect(self.db_file)
		self.cur = self.conn.cursor()

	def read_file(self):
		'''Read conversations from a .db iMessage file into a DataFrame.
		'''
		# Read the messages and handles and join them based on handle ID
		all_messages = pd.read_sql_query(self.MESSAGES_QUERY, self.conn)
		all_handles = pd.read_sql_query(self.HANDLES_QUERY, self.conn)
		message_handles = pd.merge(
			all_messages, all_handles,
			on='handle_id', how='left',
		)
		# Join each message and handle with the associated chat/conversation
		chat_messages = pd.read_sql_query(self.CHAT_MESSAGES_QUERY, self.conn)
		all_conversations = pd.merge(
			message_handles, chat_messages,
			on='message_id', how='left',
		)
		# Get only conversations with self.phone_number
		phone_matches = all_conversations['phone_number'] == self.phone_number
		return all_conversations[phone_matches]
