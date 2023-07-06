import math
import copy
import torch
import transformers

from tqdm import tqdm
from transformers import GPT2LMHeadModel, Adafactor
from torch.utils.data import DataLoader
from imessaGPT.cli_utils import create_parser, parse_train_args
from imessaGPT.imessage_dataset import ConversationDataset
from imessaGPT.logging_utils import TrainLogger


# Setup logging immediately
logger = TrainLogger()
logger.setup()


def generate_model_examples(model, tokenizer, bos_token, prompt):
	'''Generate GPT-2 example outputs, using a BOS token + prompt

	Args:
		model (transformers.GPT2LMHeadModel): The GPT-2 model for generations
		tokenizer (transformers.GPT2Tokenizer): The GPT-2 tokenizer
		bos_token (str): Beginning-of-sentence token to prepend to the prompt
		prompt (str): The prompt to pass to the model

	Returns:
		list: The (decoded) GPT-2 generations
	'''
	encoding = tokenizer(f'{bos_token}{prompt}: ', return_tensors='pt')
	generated = model.generate(
		encoding.input_ids,
		attention_mask=encoding.attention_mask,
		do_sample=False,
		top_k=50,
		max_length=512,
		top_p=0.90,
		temperature=0,
		num_return_sequences=0,
	)
	return tokenizer.decode(generated[0], skip_special_tokens=True)


def evaluate_model(model, val_data, device, non_blocking=False):
	'''Evaluates a model on validation data, computing avg. loss and perplexity

	Args:
		model (transformers.GPT2LMHeadModel): The GPT-2 model to evaluate
		val_data (torch.utils.data.DataLoader): Validation DataLoader
		device (str or torch.device): The device to use for evaluations
		non_blocking (bool, optional): Whether to use non-blocking data transfers

	Returns:
		dict: Dictionary with avg. 'val_loss' and 'val_perplexity' metrics
	'''
	# Set the model to evaluation mode, so as not to update gradients
	model.eval()
	running_loss = 0.0
	for input_ids, attention_mask in tqdm(val_data, desc='Evaluation'):
		with torch.inference_mode():
			# Get model output using the input ids and attention masks
			input_ids = input_ids.to(device, non_blocking=non_blocking)
			attention_mask = attention_mask.to(device, non_blocking=non_blocking)
			model_output = model(input_ids, attention_mask=attention_mask, labels=input_ids)
			# Update the running loss based on the model output
			logits = model_output.logits.to(device, non_blocking=non_blocking)
			loss = model_output.loss.to(device, non_blocking=non_blocking)
			running_loss += loss.item()
	# Set the model back to training mode, calculate the avg. loss/perplexity
	model.train()
	val_loss = running_loss / len(val_data.dataset)
	val_perplexity = torch.exp(val_loss)
	return {'val_loss': val_loss, 'val_perplexity': val_perplexity}


def train_model(model, train_data, val_data, optimizer, scheduler, args):
	'''Trains (finetunes) a base model and returns the trained version 

	Args:
		model (transformers.GPT2LMHeadModel): The untrained/base model
		train_data (torch.utils.data.DataLoader): Training DataLoader
		val_data (torch.utils.data.DataLoader): Validation DataLoader
		optimizer (torch.optim.Optimizer): Optimizer to use during training
		scheduler (torch.optim.lr_scheduler.LRScheduler): Learning rate scheduler
		args (Namespace): Training (CLI) arguments (see cli_utils.py)

	Returns:
		transformers.GPT2LMHeadModel: The trained/finetuned model
	'''
	# Track weights that yield the best val loss
	best_weights = copy.deepcopy(model.state_dict())
	best_loss = float('inf')
	# Progress bar and global step increment by one after each training batch
	progress_bar = tqdm(range(args.max_steps))
	step = 0
	for epoch in range(args.num_epochs):
		# Set the model to training mode, so as to perform gradient updates
		model.train()
		for input_ids, attention_mask in train_data:
			# Get the model's output from the input ids and attention masks
			input_ids = input_ids.to(
				args.device,
				non_blocking=args.non_blocking
			)
			attention_mask = attention_mask.to(
				args.device,
				non_blocking=args.non_blocking
			)
			output = model(
				input_ids,
				attention_mask=attention_mask,
				labels=input_ids,
			)
			logits = output.logits.to(args.device, non_blocking=args.non_blocking)
			loss = output.loss.to(args.device, non_blocking=args.non_blocking)
			# Perform backward pass and update/advance optimizer and scheduler
			loss.backward()
			optimizer.step()
			scheduler.step()
			optimizer.zero_grad()
			# Log training progress/metrics
			progress_bar.update(1)
			step += 1
			logger.progress(step=step, epoch=epoch)
			logger.log_train_metrics({'Training Loss': loss.item()})
			# Perform model evaluation, depending on the step
			if step % args.eval_every == 0 or step == args.max_steps:
				val_metrics = evaluate_model(
					model=model,
					val_data=val_data,
					device=args.device,
					non_blocking=args.non_blocking
				)
			if val_metrics['val_loss'] < best_loss:
				best_loss = val_metrics['val_loss']
				best_weights = copy.deepcopy(model.state_dict())
			logger.log_val_metrics(val_metrics)
			# Save a model checkpoint, depending on the step
			if step % args.checkpoint_every == 0:
				logger.log_model_checkpoint()
				torch.save({'model_state_dict': model.state_dict()}, args.model_file)
			# Stop training if we've reached the maximum number of steps
			if step >= args.max_steps:
				break
	# Load and return the model that achieved the lowest validation loss
	model.load_state_dict(best_weights)
	return model, best_loss


def main(train_args):
	'''Runs the script

	Args:
		train_args (Namespace): Training-related CLI arguments
	'''

	# Create the conversation dataset
	conversation_dataset = ConversationDataset(
		db_file=train_args.chat_file,
		phone_number=train_args.phone_number,
		sender=train_args.sender,
		receiver=train_args.receiver,
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

	# Get train/val dataloaders from the dataset
	train_dataset, val_dataset = conversation_dataset.train_val_split(
		percent_val=train_args.percent_val,
		seed=train_args.seed,
	)
	train_dataloader = DataLoader(
		train_dataset,
		batch_size=train_args.batch_size,
		shuffle=True,
		num_workers=train_args.num_workers,
		collate_fn=conversation_dataset.pad_batch,
		pin_memory=train_args.pin_memory,
	)
	val_dataloader = DataLoader(
		val_dataset,
		batch_size=train_args.batch_size,
		num_workers=train_args.num_workers,
		collate_fn=conversation_dataset.pad_batch,
		pin_memory=train_args.pin_memory,
	)

	# Initialize our GPT-2 model, making sure to resize it to our dataset
	model = GPT2LMHeadModel.from_pretrained('gpt2')
	model.resize_token_embeddings(len(conversation_dataset.tokenizer))
	model.to(train_args.device)
	logger.watch_model(model)

	# Correctly set the # of epochs and max # of training steps
	steps_per_epoch = len(train_dataloader)
	if train_args.max_steps is None:
		train_args.max_steps = train_args.num_epochs * steps_per_epoch
	else:
		train_args.num_epochs = math.ceil(train_args.max_steps / steps_per_epoch)

	optimizer = Adafactor(
		model.parameters(),
		lr=train_args.learning_rate,
		weight_decay=train_args.weight_decay,
		scale_parameter=False,
		relative_step=False,
	)

	scheduler = transformers.get_scheduler(
		name=train_args.lr_scheduler_type,
		optimizer=optimizer,
		num_warmup_steps=train_args.num_warmup_steps,
		num_training_steps=train_args.max_steps,
	)

	logger.training_overview(steps_per_epoch)
	# Show how the dataset can be used with a GPT2LMHeadModel
	# model_example(conversation_dataset)
	trained_model, best_loss = train_model(
		model=model,
		train_data=train_dataloader,
		val_data=val_dataloader,
		optimizer=optimizer,
		scheduler=scheduler,
		args=train_args,
	)
	# Save the trained model and upload it, if specified 
	logger.training_summary(best_loss=best_loss)
	torch.save({'model_state_dict': trained_model.state_dict()}, train_args.model_file)
	if train_args.upload_model:
		logger.upload_model()
	logger.finish()

if __name__ == '__main__':
	# Parse the CLI arguments and run the script
	parent_parser = create_parser()
	train_args = parse_train_args(parents=[parent_parser])
	logger.start(train_args=train_args)
	main(train_args)
