import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


def preprocess(df):
    # no preprocessing needs to be done on sample-pt.csv

    print("Initial dataset size:", len(df))

    methods = [str(method).rstrip().split() for method in df]

    return methods


def generate_vocab_with_completion_n(methods):
	"""
	Generates a vocabulary dictionary from a list of tokenized methods,
	ensuring that '[COMPLETION_N]' appears exactly once.

	Args:
		methods (list of list of str): Tokenized Java methods.

	Returns:
		dict: Vocabulary mapping each unique token to a unique index.
	"""
	# Extract unique tokens from the methods
	unique_tokens = set(token for seq in methods for token in seq)

	# Add special tokens
	unique_tokens.add("[COMPLETION_N]")  # Ensure one occurrence of [COMPLETION_N]
	unique_tokens.add("<PAD>")  # Ensure padding token is included

	# Assign each token a unique index
	vocab = {token: idx for idx, token in enumerate(unique_tokens)}

	return vocab


class CodeCompletionDataset(Dataset):
    def __init__(self, input_data, target_data, vocab, seq_length):
        """
        Custom dataset for code completion.

        Args:
            input_data (list of list of str): Tokenized input sequences.
            target_data (list of list of str): Corresponding tokenized target sequences.
            vocab (dict): Token-to-index mapping.
            seq_length (int): Fixed sequence length for padding.
        """
        self.input_data = input_data
        self.target_data = target_data
        self.vocab = vocab
        self.seq_length = seq_length

        # Ensure <PAD> token exists in vocabulary
        if '<PAD>' not in vocab:
            raise ValueError("Vocabulary must contain '<PAD>' token for padding.")

        self.pad_token_id = vocab['<PAD>']

    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, idx):
        input_seq = self.input_data[idx]
        target_seq = self.target_data[idx]

        # Convert tokens to indices
        input_seq = [self.vocab.get(token, self.pad_token_id) for token in input_seq]
        target_seq = [self.vocab.get(token, self.pad_token_id) for token in target_seq]

        # Pad sequences to the fixed length
        input_seq = self.pad_sequence(input_seq, self.seq_length)
        target_seq = self.pad_sequence(target_seq, self.seq_length)

        return torch.tensor(input_seq, dtype=torch.long), torch.tensor(target_seq, dtype=torch.long)

    def pad_sequence(self, sequence, max_len):
        """
        Pads or truncates a sequence to a fixed length.

        Args:
            sequence (list of int): List of token indices.
            max_len (int): Desired sequence length.

        Returns:
            list of int: Padded/truncated sequence.
        """
        return sequence[:max_len] + [self.pad_token_id] * max(0, max_len - len(sequence))


if __name__ == "__main__":

    df = pd.concat([pd.read_csv("sample-pt_1.csv"), pd.read_csv("sample-pt_2.csv")], ignore_index=True)
    overall_dataset = df["original_input"]
    input_dataset = df["prepared_input"]
    target_dataset = df["output"]

    print("Dataset read. Preprocessing... ")

    overall_methods = preprocess(overall_dataset)
    input_methods = preprocess(input_dataset)
    target_methods = preprocess(target_dataset)

    train_input_methods, test_input_methods = train_test_split(input_methods, test_size=0.1, random_state=42)
    train_input_methods, val_input_methods = train_test_split(train_input_methods, test_size=0.1111, random_state=42)

    train_target_methods, test_target_methods = train_test_split(target_methods, test_size=0.1, random_state=42)
    train_target_methods, val_target_methods = train_test_split(train_target_methods, test_size=0.1111, random_state=42)

    vocab = generate_vocab_with_completion_n(overall_methods)
    vocab_size = len(vocab)
    print(f"Vocab Size: {vocab_size}")
