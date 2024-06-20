import torch
from torch.utils.data import Dataset
from tokenizer import BPETokenizer
from datasets import load_dataset
from typing import Literal


class TokenPredictionDataset(Dataset):

    def __init__(
        self,
        section: Literal['train', 'test', 'val'],
        context_size: int,
        tokenizer: BPETokenizer,
        random_seed: int = 42,
        max_string_len: int = 5_000,
    ):
        """
        The Carolina dataset's wiki subset for token prediction tasks using BPE tokenization.

        This dataset class loads text data, splits it into training, validation, 
        and test sections, tokenizes the text using a Byte Pair Encoding (BPE) 
        tokenizer, and prepares context-size sequences for model training.

        Attributes:
            dataset (list): A list of tokenized sequences with specified context size.

        Args:
            section (str): The section of the dataset to use, one of `train`, `val`, or `test`.
            context_size (int): The number of tokens in each input sequence.
            tokenizer (BPETokenizer): An instance of a BPE tokenizer for encoding the text.
            random_seed (int, optional): The seed for random operations, default is 42.
            max_string_len (int, optional): The maximum length of text strings to consider, 
                default is 5000.

        Raises:
            ValueError: If `context_size` is greater than `max_string_len`.
            ValueError: If `section` is not one of `train`, `val`, or `test`.

        Example:
            ```python
            tokenizer = BPETokenizer()
            dataset = TokenPredictionDataset(section='train', context_size=128, tokenizer=tokenizer)
            ```
        """
        if context_size > max_string_len:
            raise ValueError(
                f"{context_size=} must be smaller or equal to {max_string_len=}"
            )

        # Get data from huggingface
        strings = load_dataset("carolina-c4ai/corpus-carolina", taxonomy="wik").shuffle(
            random_seed
        )["corpus"]["text"]
        strings = [s for s in strings if len(s) <= max_string_len]

        # Separate according to section
        val_start_idx = int(0.6 * len(strings))
        test_start_idx = int(0.8 * len(strings))

        if section == "train":
            strings = strings[:val_start_idx]
        elif section == "val":
            strings = strings[val_start_idx:test_start_idx]
        elif section == "test":
            strings = strings[test_start_idx:]
        else:
            raise ValueError(
                f"section must be `train`, `val` or `test`, but got `{section}`"
            )

        # Encode with tokenizer
        tokenized_strings = [
            tokenizer.encode(s) for s in strings if len(s) <= max_string_len
        ]

        # Build dataset
        self.dataset = []
        for tokenized_string in tokenized_strings:
            for i in range(len(tokenized_string) - context_size + 1):
                self.dataset.append(tokenized_string[i : i + context_size])

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        x = self.dataset[index][:-1]
        y = self.dataset[index][1:]
        return x, y
