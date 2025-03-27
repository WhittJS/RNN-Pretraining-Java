import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from torch.utils.data import Dataset, DataLoader
from collections import Counter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def preprocess(df):
    # no preprocessing needs to be done on sample-pt.csv

    print("Initial dataset size:", len(df))

    methods = [str(method).rstrip().split() for method in df]

    return methods


def generate_vocab_with_completion_n(methods, max_vocab_size=10000):
    """
    Generates a vocabulary dictionary from a list of tokenized methods,
    ensuring that '[COMPLETION_N]' appears exactly once.
    The vocabulary is capped at the most common `max_vocab_size` tokens.

    Args:
        methods (list of list of str): Tokenized Java methods.
        max_vocab_size (int): Maximum number of tokens to keep.

    Returns:
        dict: Vocabulary mapping each unique token to a unique index.
    """

    token_counts = Counter(token for seq in methods for token in seq)
    most_common_tokens = [token for token, _ in token_counts.most_common(max_vocab_size - 2)]  # Reserve 2 slots

    special_tokens = {"[COMPLETION_N]", "<PAD>"}
    unique_tokens = list(special_tokens.union(most_common_tokens))

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


class VanillaRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, dropout=0):
        super(VanillaRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):

        if isinstance(x, int):  # If x is an integer, convert it
            x = torch.tensor([x], dtype=torch.long)  # Convert to tensor

        embedded = self.embedding(x)
        output, hidden = self.rnn(embedded)
        out = self.fc(output)
        return out, hidden


def train_model(model, dataloader, optimizer, criterion, epochs):
    """Train the model and display step-wise training details."""
    model.train()
    display_interval = 100
    logs = []

    for epoch in range(epochs):
        epoch_loss = 0
        total_correct = 0
        total_samples = 0

        progress_bar = tqdm.tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch+1}")

        for batch_idx, (inputs, targets) in progress_bar:
            optimizer.zero_grad()

            inputs, targets = inputs.to(device), targets.to(device)

            outputs, _ = model(inputs)

            # Compute loss
            loss = criterion(outputs.reshape(-1, outputs.shape[-1]), targets.reshape(-1))
            loss.backward()
            optimizer.step()

            # Compute accuracy (if applicable)
            predictions = outputs.argmax(dim=-1)
            correct = (predictions == targets).sum().item()
            total = targets.numel()

            # Occasionally print input, target, and prediction for interpretation
            if batch_idx % display_interval == 0:
                print(f"Input: {inputs[0].cpu().numpy()}")
                print(f"Target: {targets[0].cpu().numpy()}")

            batch_accuracy = correct / total
            total_correct += correct
            total_samples += total

            epoch_loss += loss.item()

            # Update progress bar with loss and accuracy
            progress_bar.set_postfix(loss=f"{loss.item():.4f}", accuracy=f"{batch_accuracy:.2%}")

        # Compute epoch-level metrics
        avg_loss = epoch_loss / len(dataloader)
        avg_accuracy = total_correct / total_samples

        print(f"Epoch {epoch+1} - Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.2%}")
        logs.append({"Epoch": epoch + 1, "Loss": avg_loss, "Accuracy": avg_accuracy})

    df = pd.DataFrame(logs)
    df.to_csv("training_logs.csv", index=False)

    torch.save(model.state_dict(), "models/dl_code_completion.pth")
    print("Model saved successfully as rnn_code_completion.pth")


if __name__ == "__main__":

    df = pd.concat([pd.read_csv("sample-pt_1.csv"), pd.read_csv("sample-pt_2.csv")], ignore_index=True)
    overall_dataset = df["original_input"]
    input_dataset = df["prepared_input"]
    target_dataset = df["output"]

    print("Dataset read. Preprocessing... ")

    overall_methods = preprocess(overall_dataset)
    input_methods = preprocess(input_dataset)
    target_methods = preprocess(target_dataset)

    vocab = generate_vocab_with_completion_n(overall_methods)
    vocab_size = len(vocab)
    print(f"Vocab Size: {vocab_size}")

    datasetTrain = CodeCompletionDataset(input_methods, target_methods, vocab, seq_length=100)

    dataloader_train = DataLoader(datasetTrain, batch_size=32, shuffle=True)

    embedding_dim = 64
    hidden_dim = 256
    output_dim = vocab_size
    n_layers = 16
    learning_rate = 0.001
    epochs = 10
    dropout = 0.2

    model = VanillaRNN(vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, dropout).to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab["<PAD>"]).to(device)

    train_model(model, dataloader_train, optimizer, criterion, epochs)
