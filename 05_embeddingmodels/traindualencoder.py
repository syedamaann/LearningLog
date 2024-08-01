# Warning control
import warnings
warnings.filterwarnings('ignore')

import torch
import numpy as np
import pandas as pd

from transformers import AutoTokenizer

# Create a sample dataframe 
df = pd.DataFrame(
    [
        [4.3, 1.2, 0.05, 1.07],
        [0.18, 3.2, 0.09, 0.05],
        [0.85, 0.27, 2.2, 1.03],
        [0.23, 0.57, 0.12, 5.1]
    ]
)

# convert the dataframe to a pytorch tensor
data = torch.tensor(df.values, dtype=torch.float32)

# Define the contrastive loss function
def contrastive_loss(data):     
    target = torch.arange(data.size(0))
    loss = torch.nn.CrossEntropyLoss()(data, target)
    return loss

# Define the structure of encoder model
class Encoder(torch.nn.Module):
    def __init__(self, vocab_size, embed_dim, output_embed_dim):
        super().__init__()
        self.embedding_layer = torch.nn.Embedding(vocab_size, embed_dim)
        self.encoder = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(embed_dim, nhead=8, batch_first=True),
            num_layers=3,
            norm=torch.nn.LayerNorm([embed_dim]),
            enable_nested_tensor=False
        )
        self.projection = torch.nn.Linear(embed_dim, output_embed_dim)
    
    def forward(self, tokenizer_output):
        x = self.embedding_layer(tokenizer_output['input_ids'])
        x = self.encoder(x, src_key_padding_mask=tokenizer_output['attention_mask'].logical_not())
        cls_embed = x[:,0,:]
        return self.projection(cls_embed)