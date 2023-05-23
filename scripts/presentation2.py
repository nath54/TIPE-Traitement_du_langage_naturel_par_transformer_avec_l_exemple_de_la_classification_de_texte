import torch
import math
from transformers import BertTokenizer, BertModel

# Load BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Sentence to encode
sentence = "This is a sample sentence."

# Tokenization
tokens = tokenizer.tokenize(sentence)
tokens = ['[CLS]'] + tokens + ['[SEP]']
token_ids = tokenizer.convert_tokens_to_ids(tokens)

# Positional Encoding
max_len = len(token_ids)
d_model = model.config.hidden_size  # Dimension of BERT hidden states

# Create positional encodings
positional_encodings = torch.zeros((1, max_len, d_model))
for pos in range(max_len):
    for i in range(d_model):
        if i % 2 == 0:
            positional_encodings[0, pos, i] = math.sin(pos / (10000 ** (2 * i / d_model)))
        else:
            positional_encodings[0, pos, i] = math.cos(pos / (10000 ** ((2 * i - 1) / d_model)))

# Convert token IDs and positional encodings to tensors
token_tensor = torch.tensor([token_ids])
pos_enc_tensor = torch.tensor(positional_encodings)

# BERT Model
model.eval()
with torch.no_grad():
    token_embeddings = model.embeddings.word_embeddings(token_tensor)  # Token embeddings
    combined_embeddings = token_embeddings + pos_enc_tensor  # Combine tokens and positional encodings

# Print the combined embeddings
print("Combined Embeddings Shape:", combined_embeddings.shape)
print("Combined Embeddings:")
print(combined_embeddings)
