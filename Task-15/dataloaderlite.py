import os
import math
import time
import inspect
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F

import tiktoken
from transformers import AutoTokenizer
import torch
from transformers import AutoTokenizer
from datasets import load_dataset

class DataLoaderLite:
    def __init__(self, B, T):
        self.B = B
        self.T = T

        # Load the Hugging Face tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/cosmo2-tokenizer")

        # Load dataset in streaming mode
        self.dataset = load_dataset(
            "HuggingFaceTB/smollm-corpus",
            name="cosmopedia-v2",
            split="train",
            streaming=True
        )

        # Create an iterator
        self.iterator = iter(self.dataset)

        print("Streaming dataset loaded.")

    def next_batch(self):
        B, T = self.B, self.T
        token_batches = []

        while len(token_batches) < B * T + 1:  # Collect enough tokens for a full batch
            try:
                sample = next(self.iterator)  # Get next sample from dataset
                text = sample['text']  # Adjust key if necessary
                tokens = self.tokenizer.encode(text, add_special_tokens=False)
                token_batches.extend(tokens)
            except StopIteration:
                # Reset iterator if we run out of data
                self.iterator = iter(self.dataset)

        # Convert list to tensor
        token_tensor = torch.tensor(token_batches[:B * T + 1])

        x = token_tensor[:-1].view(B, T)  # Inputs
        y = token_tensor[1:].view(B, T)   # Targets

        return x, y

# CHANGES IN CURRENT CODE
torch.set_float32_matmul_precision('high')
