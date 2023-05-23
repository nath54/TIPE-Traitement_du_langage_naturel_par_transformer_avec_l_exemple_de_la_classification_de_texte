import math
import torch

from transformers import BertTokenizer

# Load BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Sentence to encode
sentence = "Neural Networks are so cool!"

# Step 1: Tokenization
tokens = tokenizer.tokenize(sentence)
tokens = ['[CLS]'] + tokens + ['[SEP]']
token_ids = tokenizer.convert_tokens_to_ids(tokens)

# Step 2: Segment Embedding
segment_ids = [0] * len(token_ids)

# Step 3: Positional Encoding
max_len = len(token_ids)
d_model = 32  # Dimension of BERT hidden states

positional_encodings = torch.zeros((1, max_len, d_model))

for pos in range(max_len):
    for i in range(d_model):
        if i % 2 == 0:
            positional_encodings[0, pos, i] = math.sin(pos / (10000 ** (2 * i / d_model)))
        else:
            positional_encodings[0, pos, i] = math.cos(pos / (10000 ** ((2 * i - 1) / d_model)))



# Print intermediate results
print("Tokens:", tokens)
print("Token IDs:", token_ids)
print("Segment IDs:", segment_ids)
print("Positional Encodings Shape:", positional_encodings.shape)
print("Positional Encoding : ", positional_encodings)