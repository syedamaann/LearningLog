# Warning control
import warnings
warnings.filterwarnings('ignore')

import torch
import numpy as np
import pandas as pd

from transformers import AutoTokenizer

# Create a sample dataframe and convert to a pytorch tensor
df = pd.DataFrame(
    [
        [4.3, 1.2, 0.05, 1.07],
        [0.18, 3.2, 0.09, 0.05],
        [0.85, 0.27, 2.2, 1.03],
        [0.23, 0.57, 0.12, 5.1]
    ]
)
data = torch.tensor(df.values, dtype=torch.float32)