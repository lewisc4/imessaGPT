from transformers import GPT2LMHeadModel
from imessaGPT.cli_utils import create_parser, parse_preprocessing_args
from imessaGPT.imessage_data import ConversationDataset


def model_example(conversation_dataset):
	'''Basic example to show how a conversation dataset can be used with GPT2.
	'''
	# Create the model and resize token embeddings to match our dataset
	model = GPT2LMHeadModel.from_pretrained('gpt2')
	model.resize_token_embeddings(len(conversation_dataset.tokenizer))
	# Create example prompt in a conversational format and tokenize it
	prompt = f'{conversation_dataset.BOS_TOKEN}Chris: '
	generated = conversation_dataset.tokenizer(f'{prompt}', return_tensors='pt')
	# Generate example outputs from our prompt
	sample_outputs = model.generate(
		generated.input_ids,
		attention_mask=generated.attention_mask,
		do_sample=False,
		top_k=50,
		max_length=512,
		top_p=0.90,
		temperature=0,
		num_return_sequences=0,
	)
	# Decode the predictions into readable text
	predicted = conversation_dataset.tokenizer.decode(
		sample_outputs[0],
		skip_special_tokens=True,
	)
	print(predicted)


def main(args):
	'''Runs the script

	Args:
		args (Namespace): The script's CLI arguments
	'''
	# Create the conversation dataset
	conversation_dataset = ConversationDataset(
		db_file=args.chat_file,
		phone_number=args.phone_number,
		sender='Sender',
		receiver='Receiver',
	)
	# Clean the conversation's messages
	conversation_dataset.clean_messages(
		min_len=1,
		keep_urls=False,
		keep_reactions=False,
	)
	# Build and tokenize the dataset's samples and show some examples
	conversation_dataset.build_samples()
	conversation_dataset.tokenize_samples()
	conversation_dataset.show_examples(n_examples=3, sep='*')
	# Show how the dataset can be used with a GPT2LMHeadModel
	# model_example(conversation_dataset)


if __name__ == '__main__':
	# Parse the CLI arguments and run the scrip
	parent_parser = create_parser()
	args = parse_preprocessing_args(parents=[parent_parser])
	main(args)
