import warnings     
import datasets     
import os           
import requests
import heapq
import re
import urllib
import numpy as np
import torch
import transformers
from torch.utils.data import Dataset
from fasttext.FastText import _FastText
from dataclasses import dataclass, field
from transformers import AutoTokenizer
from transformers import LlamaConfig
from transformers import LlamaForCausalLM
from transformers import AutoModelForCausalLM
from transformers import TextStreamer
from copy import deepcopy

# Suppress the warnings
warnings.filterwarnings("ignore") 

# Load the pretraining dataset and Select only the text column
pretraining_dataset = datasets.load_dataset( "upstage/Pretraining_Dataset", split="train")
pretraining_dataset = pretraining_dataset.select_columns(["text"]) 

# Scrape python code from Github 
urls = [
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/searches/double_linear_search_recursion.py",
    "https://raw.githubusercontent.com/KosingZhu/tensorflow/master/tensorflow/python/tools/module_util.py",
    "https://raw.githubusercontent.com/EricRemmerswaal/tensorflow/master/tensorflow/python/distribute/distribute_coordinator_context.py",
    "https://raw.githubusercontent.com/computationalartist/tensorflow/master/tensorflow/python/ops/numpy_ops/integration_test/benchmarks/numpy_mlp.py",
    "https://raw.githubusercontent.com/Van-an/tensorflow/master/tensorflow/python/distribute/coordinator/values.py",
    "https://raw.githubusercontent.com/nkgwer/tensorflow/master/tensorflow/lite/tools/visualize.py",
    "https://raw.githubusercontent.com/gitblazer/youtube-dl/master/youtube_dl/version.py",
    "https://raw.githubusercontent.com/Joshua-Barawa/My-Photos/master/venv/lib/python3.8/site-packages/django/contrib/messages/__init__.py",
    "https://raw.githubusercontent.com/PaliC/pytorch/master/test/fx/test_subgraph_rewriter.py"
]

# Download the code files and save them to the code directory
code_dir = "./code"
os.makedirs(code_dir, exist_ok=True)    # Create the code directory if it does not exist
print("Downloading code files...")
for url in urls:
    print(f"Working on url: {url}")
    response = requests.get(url)
    file_name = os.path.basename(url)
    file_path = os.path.join(code_dir, file_name)
    
    with open(file_path, "wb") as file:
        file.write(response.content)

# Load the code files into a dataset
print("Loading code files into a dataset...")
code_dataset = []   
for file in os.listdir(code_dir):
    file_path = os.path.join(code_dir, file)
    if os.path.isfile(file_path):  # Ensure the entry is a file
        with open(file_path, 'r') as f:
            code_dataset.append({'text': f.read()})
code_dataset = datasets.Dataset.from_list(code_dataset)     # Convert the list of dictionaries to a HF dataset object

# Concatenate the pretraining and code datasets to get the final dataset
dataset = datasets.concatenate_datasets([pretraining_dataset, code_dataset])

# Define a filter to remove paragraphs with too few lines or lines that are too short
def paragraph_length_filter(x):        
    """Returns False iff a page has too few lines or lines are too short."""
    lines = x['text'].split('\n')
    if (
        len(lines) < 3
        or min(heapq.nlargest(3, [len(line) for line in lines])) < 3
    ):
        return False
    return True

# Apply the filter
print("Filtering out paragraphs with too few lines or lines that are too short...")
dataset = dataset.filter(
    paragraph_length_filter,
    load_from_cache_file=False
)

# Define a function to find duplicates
def find_duplicates(paragraphs):
    """
    Use this function to find the number of repetitions 
    in the paragraphs.
    """
    unique_x = set()
    duplicate_chars = 0
    duplicate_elements = 0
    for element in paragraphs:
        if element in unique_x:
            duplicate_chars += len(element)
            duplicate_elements += 1
        else:
            unique_x.add(element)
    return duplicate_elements, duplicate_chars

# Define a filter to remove paragraphs with too many repetitions
def paragraph_repetition_filter(x):
    """
    Returns False iff a page has too many repetitions.
    """
    text = x['text']
    paragraphs = re.compile(r"\n{2,}").split(text.strip())                # Split by paragraphs (2 or more newlines)
    paragraphs_duplicates, char_duplicates = find_duplicates(paragraphs)  # Find number of duplicates in paragraphs
    if paragraphs_duplicates / len(paragraphs) > 0.3:
        return False
    if char_duplicates / len(text) > 0.2:
        return False
    return True

# Apply the filter
print("Filtering out paragraphs with too many repetitions...")
dataset = dataset.filter(
    paragraph_repetition_filter,
    load_from_cache_file=False
)

# Define a function to remove duplicate entries
def deduplication(ds):
    def dedup_func(x):
        """Use this function to remove duplicate entries"""
        if x['text'] in unique_text:
            return False
        else:
            unique_text.add(x['text'])
            return True

    unique_text = set()

    ds = ds.filter(dedup_func, load_from_cache_file=False, num_proc=1)
    return ds

# Apply the filter
print("Removing duplicate entries...")
dataset = deduplication(dataset)

# Define a function to remove non-English paragraphs
def english_language_filter(ds):
    # load language detection model
    model = _FastText('./model/L2_language_model.bin')
    
    def is_english(x):
        # Predict language of the text and probability
        language, score = model.predict(x['text'].replace("\n", ""))

        language = language[0].split("__")[2]
        return score > 0.4 and language == "en" # change code here if building a model in another language

    ds = ds.filter(is_english, load_from_cache_file=False, num_proc=1)
    return ds

# Apply the filter
print("Filtering out non-English paragraphs...")
dataset = english_language_filter(dataset)

# Save the dataset to a parquet file
print("Saving the dataset to a parquet file...")
file_path = "./data/preprocessed_dataset.parquet"
dataset.to_parquet(file_path)

print("Preprocessing completed successfully!")

# split the dataset into 10 shards and select the first shard
print("Splitting the dataset into 10 shards and selecting the first shard...")
dataset = dataset.shard(num_shards=10, index=0)

# Load the tokenizer
print("Loading the tokenizer...")
model_path_or_name = "upstage/SOLAR-10.7B-v1.0"
tokenizer = AutoTokenizer.from_pretrained(
    model_path_or_name, 
    use_fast=False
)

# Define a function to tokenize the dataset
def tokenization(example):
    # Tokenize
    tokens = tokenizer.tokenize(example["text"])

    # Convert tokens to ids
    token_ids = tokenizer.convert_tokens_to_ids(tokens)

    # Add <bos>, <eos> tokens to the front and back of tokens_ids 
    # bos: begin of sequence, eos: end of sequence
    token_ids = [
        tokenizer.bos_token_id] \
        + token_ids \
        + [tokenizer.eos_token_id
    ]
    example["input_ids"] = token_ids

    # We will be using this column to count the total number of tokens 
    # in the final dataset
    example["num_tokens"] = len(token_ids)
    return example

# Apply the tokenization function
print("Tokenizing the dataset...")
dataset = dataset.map(tokenization, load_from_cache_file=False)

# Print the total number of tokens in the dataset (we should know the numbers of tokens so that we can turn them into (B,T))
print("Total number of tokens in the dataset: ", np.sum(dataset["num_tokens"]))

# Concatenate the input_ids
print("Concatenating the input_ids into a single tensor...")
input_ids = np.concatenate(dataset["input_ids"])

# Define the max sequence length
max_seq_length = 32

# Discard extra tokens from end of the list so number of tokens is exactly divisible by max_seq_length
print("Truncating the input_ids to the nearest multiple of max_seq_length...")
total_length = len(input_ids) - len(input_ids) % max_seq_length
input_ids = input_ids[:total_length]

# Reshape the input_ids to (B, T)
input_ids_reshaped = input_ids.reshape(-1, max_seq_length).astype(np.int32)

# Convert the input_ids to a HF dataset
print("Converting the input_ids to a HF dataset...")
input_ids_list = input_ids_reshaped.tolist()
packaged_pretrain_dataset = datasets.Dataset.from_dict(
    {"input_ids": input_ids_list}
)

# Save the dataset to a parquet file
print("Saving the dataset to a parquet file...")
packaged_pretrain_dataset.to_parquet("./data/packaged_pretrain_dataset.parquet")

# Create a LlamaConfig object to configure the architecture of the model
print("Creating a LlamaConfig object...")
config = LlamaConfig()

# Update parameters to change the model architecture
print("Updating parameters to change the model architecture...")
config.num_hidden_layers = 12      # reduced from 32 to 12
config.hidden_size = 1024          # reduced 1/4 from 4096 to 1024
config.intermediate_size = 4096    # reduced 1/3 from 11008 to 4096 (dimension of MLP representations)
config.num_key_value_heads = 8     # reduced 1/4 from 32 to 8 (defaults to num_attention_heads=32)
config.torch_dtype = "bfloat16"    # for half-precision training
config.use_cache = False           # `True` is incompatible w/ gradient checkpointing

# Print the config
print("Printing the config...")
print(config)

# Random Weight Initialization 
# model = LlamaForCausalLM(config) 

# weight initialization using an existing model
# # If you load an existing model, you can use it as is to continue pretraining on new data
# model_name_or_path = "./models/upstage/TinySolar-248m-4k"
# model = AutoModelForCausalLM.from_pretrained(
#     model_name_or_path,
#     device_map="cpu",
#     torch_dtype=torch.bfloat16,
# )

# downscaling from a general pretrained model
# model_name_or_path = "upstage/TinySolar-248m-4k"
# model = AutoModelForCausalLM.from_pretrained(
#     model_name_or_path,
#     device_map="cpu",
#     torch_dtype=torch.bfloat16,
# )
# tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
# print_nparams(model)  # 248013824 => 248M

# layers = model.model.layers
# model.model.layers = layers[:5] + layers[-5:]     # remove the middle two layers (layers 5 and 6)

# config = AutoConfig.from_pretrained(              # update the config to reflect the new number of layers
#     model_name_or_path,    
#     num_hidden_layers=len(model.model.layers),
# )
# model.config = config                             # update the model config

# print_nparams(model)  # 217601024 => 217M


# Configure a 16 layer model and initialize it with random weights
print("Configuring a 16 layer model...")
config = LlamaConfig(
    num_hidden_layers=16,  # We want our model to have 16 final layers
    hidden_size=1024,
    intermediate_size=4096,
    num_attention_heads=32,
    num_key_value_heads=8,
    torch_dtype="bfloat16",
    use_cache=False 
)
print("Initializing the model with random weights...")
model = LlamaForCausalLM(config)
model = model.to(dtype=torch.bfloat16)  # convert to bfloat16

# Print the number of parameters in the model
def print_nparams(model):
    """Calculate the total number of model parameters"""
    nparams = sum(p.numel() for p in model.parameters())
    print(f"The total number of parameters is: {nparams}")
print_nparams(model)  # 248013824 => 248M

# Load the pretrained model and tokenizer
print("Loading the pretrained 12 layer model and tokenizer...")
model_name_or_path = "upstage/TinySolar-248m-4k"
pretrained_model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    device_map="cpu",
    torch_dtype=torch.bfloat16,    
)
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

# Copy the bottom 8 and top 8 layers from the pretrained 12 layer model and use them to overwrite the random weights of the 16 layer model
print("Copying the bottom 8 and top 8 layers from the 12 layer model...")
model.model.layers = deepcopy(pretrained_model.model.layers[:-4]) \
    + deepcopy(pretrained_model.model.layers[4:])

# Copy the embed_tokens and lm_head layers from the pretrained 12 layer model
print("Copying the embed_tokens and lm_head layers from the pretrained model...")
model.model.embed_tokens = deepcopy(pretrained_model.model.embed_tokens)
model.lm_head = deepcopy(pretrained_model.lm_head)

# Print the config of the updated model
print("Printing the config of the updated model...")
print(model.config)

# Save the model to a directory
model.save_pretrained('./model/TinySolar-308m-4k-init') # new model name here reflects the 308 million parameters of the new, upscaled model

# Load the pretrained model
print("Loading the pretrained model...")
pretrained_model = AutoModelForCausalLM.from_pretrained(
    "./model/TinySolar-308m-4k-init",
    device_map="cpu", 
    torch_dtype=torch.bfloat16,
    use_cache=False,
)


# Load the dataset
class CustomDataset(Dataset):                   # Inherit from Pytorch's Dataset class
    def __init__(self, args, split="train"):
        """Initializes the custom dataset object."""
        self.args = args
        self.dataset = datasets.load_dataset(
            "parquet",
            data_files=args.dataset_name,
            split=split
        )

    def __len__(self):
        """Returns the number of samples in the dataset."""
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        Retrieves a single data sample from the dataset 
        at the specified index
        """
        # Convert the lists to a LongTensor for PyTorch
        input_ids = torch.LongTensor(self.dataset[idx]["input_ids"])
        labels = torch.LongTensor(self.dataset[idx]["input_ids"])

        # Return the sample as a dictionary
        return {"input_ids": input_ids, "labels": labels}
    
 
# Configure Training Arguments
@dataclass
class CustomArguments(transformers.TrainingArguments):
    dataset_name: str = field(                           # Dataset configuration
        default="./data/packaged_pretrain_dataset.parquet")
    num_proc: int = field(default=1)                     # Number of subprocesses for data preprocessing
    max_seq_length: int = field(default=32)              # Maximum sequence length

    # Core training configurations
    seed: int = field(default=0)                         # Random seed for initialization, ensuring reproducibility
    optim: str = field(default="adamw_torch")            # Optimizer, here it's AdamW implemented in PyTorch
    max_steps: int = field(default=30)                   # Number of maximum training steps
    per_device_train_batch_size: int = field(default=2)  # Batch size per device during training

    # Other training configurations
    learning_rate: float = field(default=5e-5)           # Initial learning rate for the optimizer
    weight_decay: float = field(default=0)               # Weight decay
    warmup_steps: int = field(default=10)                # Number of steps for the learning rate warmup phase
    lr_scheduler_type: str = field(default="linear")     # Type of learning rate scheduler
    gradient_checkpointing: bool = field(default=True)   # Enable gradient checkpointing to save memory
    dataloader_num_workers: int = field(default=2)       # Number of subprocesses for data loading
    bf16: bool = field(default=False)                    # Use bfloat16 precision for training on supported hardware
    gradient_accumulation_steps: int = field(default=1)  # Number of steps to accumulate gradients before updating model weights
    
    # Logging configuration
    logging_steps: int = field(default=3)                # Frequency of logging training information
    report_to: str = field(default="none")               # Destination for logging (e.g., WandB, TensorBoard)

    # Saving configuration
    # save_strategy: str = field(default="steps")          # Can be replaced with "epoch"
    # save_steps: int = field(default=3)                   # Frequency of saving training checkpoint
    # save_total_limit: int = field(default=2)             # The total number of checkpoints to be saved    

# Parse the custom arguments
parser = transformers.HfArgumentParser(CustomArguments)