# iMessage GPT

## Overview
This project provides the functionality to train a custom GPT model based on a user's iMessages. This involves fine-tuning a [GPT-2 model](https://huggingface.co/docs/transformers/model_doc/gpt2) from OpenAI (specifically [GPT2LMHeadModel](https://huggingface.co/docs/transformers/model_doc/gpt2#transformers.GPT2LMHeadModel)) using the `chat.db` file found on Apple devices, which houses iMessage-related metadata. On Mac devices, the `chat.db` file is typically found here: `~/Library/Messages/chat.db`.

## Environment Setup
### Package Installation
It is necessary to have python >= 3.7 installed in order to run the code for this project. In order to install the necessary libraries and modules, follow the below instructions.

1. Clone or download this project to your local computer.
2. Navigate to the [root directory](https://github.com/lewisc4/imessaGPT), where the [`setup.py`](/setup.py) file is located.
3. Install the [`imessagGPT`](/imessaGPT) module and all dependencies by running: `pip install -e .` (required python modules are in [`requirements.txt`](/requirements.txt)).

### Dataset Setup
As mentioned earlier, the `chat.db` file found on Apple devices stores iMessage-related metadata and it will be used as the dataset for fine-tuning. On Mac devices, the `chat.db` file is typically found here: `~/Library/Messages/chat.db`. It is recommended to make a copy of this file and move it elsewhere, as you may encounter permission errors if you attempt to access the original file. By default, this project assumes it will be copied to the [`cli/`](/cli) directory, but it can be copied anywhere as long as the new location is specified before preprocessing.

### GPU-related Requirements/Installations
Follow the steps below to ensure your GPU and all relevant libraries are up to date and in good standing.

1. If you are on a GPU machine, you need to install a GPU version of pytorch. To do that, first check what CUDA version your server has with `nvidia-smi`.
2. If your CUDA version is below 10.2, don't use this server
3. If your CUDA version is below 11, run `pip install torch`
4. Else, `pip install torch==1.10.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html`
5. Check that pytorch-GPU works via `python -c "import torch; print(torch.cuda.is_available())"`. If it returns False, reinstall pytorch via one of the above commands (usually this helps).
6. If you are using 30XX, A100 or A6000 GPU, you have to use CUDA 11.3 and above.

## Training
The [`train.py`](/cli/train.py) script is used to train a model via CLI arguments.

### Hyperparameters
All available script arguments can be found in [cli_utils.py](/imessaGPT/cli_utils.py). Some useful parameters to change/test with are:

| Argument/Parameter     | Description                                                                                                                                                               |
|------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `--chat_file`          | Path to chat.db, `cli/chat.db` by default (see [**Dataset Setup**](https://github.com/lewisc4/imessaGPT/blob/main/README.md#dataset-setup))                               |
| `--output_dir`         | Directory to save the trained model to (created if it doesn't exist)                                                                                                      |
| `--phone_number`       | Phone number of the person to use for the message thread for training                                                                                                     |
| `--sender`             | Name of the person sending messages to phone_number (i.e., you)                                                                                                           |
| `--receiver`           | Name of the person receiving messages (i.e., the person with phone_number)                                                                                                |
| `--model_file`         | The name of the `.pt` model file to save in `output_dir`                                                                                                                  |
| `--percent_val`        | % of data to use for validation                                                                                                                                           |
| `--num_workers`        | Number of workers to use in DataLoader(s)                                                                                                                                 |
| `--learning_rate`      | External learning rate used by the optimizer                                                                                                                              |
| `--device`             | Device to train on, defaults to `cuda` if available, otherwise `cpu`                                                                                                      |
| `--batch_size`         | Batch size to use in DataLoader(s)                                                                                                                                        |
| `--weight_decay`       | External weight decay used by the optimizer                                                                                                                               |
| `--num_epochs`         | Number of training epochs                                                                                                                                                 |
| `--eval_every`         | How often, in number of training steps, to evaluate the model                                                                                                             |
| `--wandb_project`      | Weights & Biases project name (account not required)                                                                                                                      |
| `--upload_model`       | Whether to upload the model to Weights & Biases or not                                                                                                                    |

### Example Usage
For the below examples, assume we have saved a copy of the `chat.db` file to: `cli/chat.db`, as described in the [**Dataset Setup**](https://github.com/lewisc4/imessaGPT/blob/main/README.md#dataset-setup) section. Also, assume the commands are run from the [`cli/`](/cli) directory.

```bash
# To train a model on a message thread between you and someone with the phone number +1(888)-777-6666:
python3 train.py --phone_number=+1888777666

# To the same as above, but using 20% of the message thread as validation data:
python3 train.py --phone_number=+1888777666 --percent_val=0.2

```

