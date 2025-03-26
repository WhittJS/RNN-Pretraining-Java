import pandas as pd


def preprocess(df):
    # no preprocessing needs to be done on sample-pt.csv

    print("Initial dataset size:", len(df))

    # print(df["Method Code"])
    methods = df["original_input"]

    methods = [method.rstrip().split() for method in methods]

    return methods


def tokenizaton_vocab(methods):
    # TOKENIZATION AND CREATING VOCABULARY
    # Extract unique tokens from the methods
    unique_tokens = set(token for seq in methods for token in seq)
    # Add special tokens
    unique_tokens.add("[COMPLETION_N]")  # Ensure one occurrence of [COMPLETION_N]
    unique_tokens.add("<PAD>")  # Ensure padding token is included

    # Assign each token a unique index
    vocab = {token: idx for idx, token in enumerate(unique_tokens)}

    # Create bidirectional mappings
    token_to_id = {token: idx for idx, token in enumerate(unique_tokens)}
    id_to_token = {idx: token for token, idx in token_to_id.items()}

    # Exploration Point
    print("\nVocabulary size:", len(token_to_id))
    print("'<PAD>' token ID:", token_to_id['<PAD>'])
    print("5 random vocabulary items:", list(token_to_id.items())[:5])
    return vocab, id_to_token


if __name__ == "__main__":

    df = pd.concat([pd.read_csv("sample-pt_1.csv"), pd.read_csv("sample-pt_2.csv")], ignore_index=True)

    print("Dataset read. Preprocessing... ")

    methods = preprocess(df)

    vocab, id_to_token = tokenizaton_vocab(methods)
