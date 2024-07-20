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