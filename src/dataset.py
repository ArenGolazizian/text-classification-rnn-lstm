import pandas as pd
from collections import Counter
from utils import tokenize, remove_rare_words
from nltk.corpus import stopwords

def load_and_preprocess_data(data_path, max_vocab, max_len):
    df = pd.read_csv(data_path)
    stop_words = set(stopwords.words('english'))

    df['tokens'] = df['review'].apply(lambda x: tokenize(x, stop_words))

    all_tokens = [token for tokens in df['tokens'] for token in tokens]
    common_tokens = set(list(zip(*Counter(all_tokens).most_common(max_vocab)))[0])
    df['tokens'] = df['tokens'].apply(lambda x: remove_rare_words(x, common_tokens, max_len))

    df = df[df['tokens'].apply(lambda tokens: any(token != '<UNK>' for token in tokens))]

    vocab = sorted(set([token for tokens in df['tokens'] for token in tokens]))
    token2idx = {token: idx for idx, token in enumerate(vocab)}
    token2idx['<PAD>'] = len(token2idx)

    df['indexed_tokens'] = df['tokens'].apply(lambda tokens: [token2idx[token] for token in tokens])

    return df['indexed_tokens'].tolist(), df['label'].tolist(), token2idx

def split_data(sequences, targets, valid_ratio=0.05, test_ratio=0.05):
    total_size = len(sequences)
    test_size = int(total_size * test_ratio)
    valid_size = int(total_size * valid_ratio)
    train_size = total_size - valid_size - test_size

    train_sequences, train_targets = sequences[:train_size], targets[:train_size]
    valid_sequences, valid_targets = sequences[train_size:train_size+valid_size], targets[train_size:train_size+valid_size]
    test_sequences, test_targets = sequences[train_size+valid_size:], targets[train_size+valid_size:]

    return train_sequences, train_targets, valid_sequences, valid_targets, test_sequences, test_targets
