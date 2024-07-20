import warnings     
import datasets     
import os           
import requests
import heapq
import urllib
from fasttext.FastText import _FastText

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
    code_dataset.append({'text': open(os.path.join(code_dir, file), 'r').read()})
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
    model = _FastText('./models/L2_language_model.bin')
    
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