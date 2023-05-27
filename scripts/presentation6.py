import torch
from transformers import BertModel
from transformers import BertTokenizer
from tqdm import tqdm

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

model = BertModel.from_pretrained("bert-base-uncased").cuda()


def get_embedding(word):
    with torch.no_grad():
        tokens = tokenizer.tokenize(word)

        token_ids = tokenizer.convert_tokens_to_ids(tokens)

        input_ids = torch.tensor([token_ids]).cuda()
        outputs = model(input_ids)
        word_embedding = outputs.last_hidden_state[0][0]
        #
        del tokens
        del token_ids
        del input_ids
        del outputs
        torch.cuda.empty_cache()
        return word_embedding



themes = {
    "city": [],
    "cars": [],
    "house": [],
    "apparts": [],
    "stores": [],
    "police": [],
    "hospitals": []
}










