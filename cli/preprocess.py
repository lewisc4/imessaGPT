from pathlib import Path
from imessaGPT.cli_utils import parse_preprocess_args
from imessaGPT.database import CSVConversationDB, iMessageConversationDB
from imessaGPT.preprocessor import ConversationPreprocessor


def get_raw_data(args):
	'''Gets raw file data (from args.raw_file). Data is stored in a DataFrame
	and has been adapted to a database schema (see imessaGPT.database).

	Args:
		args (Namespace): The CLI script arguments (see cli_utils.py).

	Returns:
		pd.DataFrame: Pandas DataFrame that holds the raw data.
	'''
	# Create the database based on raw_file's extension (must be .csv or .db).
	file_extension = Path(args.raw_file).suffix
	if file_extension == CSVConversationDB.DB_FILE_EXTENSION:
		database = CSVConversationDB(
			db_file=args.raw_file,
			message_col=args.message_col,
		)
	elif file_extension == iMessageConversationDB.DB_FILE_EXTENSION:
		database = iMessageConversationDB(
			db_file=args.raw_file,
			message_col=args.message_col,
			phone_number=args.phone_number
		)
	else:
		raise ValueError('Raw data file must be either a .csv or .db file')
	# Populate the database with data from raw_file and return its data
	database.populate_data()
	return database.data


def main(args):
	# Get the raw data
	raw_data = get_raw_data(args)
	# Create a preprocessor
	preprocessor = ConversationPreprocessor(
		data=raw_data,
		sender_name=args.sender,
		receiver_name=args.receiver,
	)
	# Preprocess the raw data and save it 
	preprocessor.preprocess(
		keep_reactions=args.keep_reactions,
		keep_urls=args.keep_urls,
		min_len=args.min_len,
	)
	preprocessor.save_data(args.processed_file)


if __name__ == '__main__':
	# Parse the CLI arguments and run the script
	args = parse_preprocess_args()
	main(args)
