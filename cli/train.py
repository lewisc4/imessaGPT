import math
import copy
import torch
import transformers
import pandas as pd

from tqdm import tqdm
from transformers import GPT2LMHeadModel, Adafactor
from torch.utils.data import DataLoader

from imessaGPT.cli_utils import parse_train_args
from imessaGPT.dataset import ConversationDataset
from imessaGPT.logging_utils import TrainLogger


# Setup logging immediately
logger = TrainLogger()
logger.setup()


def evaluate_model(model, val_data, args):
	'''Evaluates a model on validation data, computing avg. loss and perplexity

	Args:
		model (transformers.GPT2LMHeadModel): The GPT-2 model to evaluate
		val_data (torch.utils.data.DataLoader): Validation DataLoader
		args (Namespace): Training (CLI) arguments (see cli_utils.py)

	Returns:
		dict: Dictionary with avg. 'val_loss' and 'val_perplexity' metrics
	'''
	# Set the model to evaluation mode, so as not to update gradients
	model.eval()
	running_loss = 0.0
	for inputs, attn_mask in tqdm(val_data, desc='Evaluation'):
		with torch.inference_mode():
			# Get model output using the input ids and attention masks
			inputs = inputs.to(args.device, non_blocking=args.non_blocking)
			attn_mask = attn_mask.to(args.device, non_blocking=args.non_blocking)
			output = model(inputs, attention_mask=attn_mask, labels=inputs)
			# Update the running loss based on the model output
			loss = output.loss.to(args.device, non_blocking=args.non_blocking)
			running_loss += loss.item()
	# Set the model back to training mode, calculate the avg. loss/perplexity
	model.train()
	val_loss = running_loss / len(val_data.dataset)
	val_perplexity = torch.exp(torch.Tensor([val_loss])).item()
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
		for inputs, attn_mask in train_data:
			# Set gradients to 0 so as not to accumulate them
			optimizer.zero_grad()
			# Forward pass
			inputs = inputs.to(args.device, non_blocking=args.non_blocking)
			attn_mask = attn_mask.to(args.device, non_blocking=args.non_blocking)
			output = model(inputs, attention_mask=attn_mask, labels=inputs)
			loss = output.loss.to(args.device, non_blocking=args.non_blocking)
			# Backward pass and updating/advancing optimizer and scheduler
			loss.backward()
			optimizer.step()
			scheduler.step()
			# Log training progress/metrics
			progress_bar.update(1)
			step += 1
			logger.progress(step=step, epoch=epoch)
			logger.log_train_metrics({'Training Loss': loss.item()})
			# Perform model evaluation, depending on the step
			if step % args.eval_every == 0 or step == args.max_steps:
				val_metrics = evaluate_model(model, val_data, args)
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
	dataset_messages = pd.read_csv(train_args.dataset)[:train_args.num_examples]
	conversation_dataset = ConversationDataset(dataset_messages)
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

	logger.training_overview()
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
	train_args = parse_train_args()
	logger.start(train_args=train_args)
	main(train_args)
