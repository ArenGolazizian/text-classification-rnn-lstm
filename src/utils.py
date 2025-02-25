import re
from nltk import wordpunct_tokenize
import torch

def tokenize(text, stop_words):
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower()
    tokens = wordpunct_tokenize(text)
    tokens = [token for token in tokens if token not in stop_words]
    return tokens

def remove_rare_words(tokens, common_tokens, max_len):
    return [token if token in common_tokens else '<UNK>' for token in tokens][-max_len:]

def pad_sequences(sequences, padding_val=0, pad_left=False):
    sequence_length = max(len(sequence) for sequence in sequences)
    if not pad_left:
        return [sequence + [padding_val] * (sequence_length - len(sequence)) for sequence in sequences]
    return [[padding_val] * (sequence_length - len(sequence)) + sequence for sequence in sequences]

def collate_fn(batch, padding_val):
    inputs, targets = zip(*batch)
    inputs_padded = pad_sequences(inputs, padding_val=padding_val)
    return torch.LongTensor(inputs_padded), torch.LongTensor(targets)
