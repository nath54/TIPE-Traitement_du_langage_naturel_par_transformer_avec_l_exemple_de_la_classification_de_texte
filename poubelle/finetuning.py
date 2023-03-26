from lib import *

PATH = "../bert/sentiments_initial_v5.pt"
# DATASET_PATH = "../datasets/sentiment140/parsed.csv"
DATASET_PATH = "../datasets/sentiment_initial/train.tsv"
CSV_DELIMITER = "\t"
NUM_CLASSES = 1

tokenizer = load_tokenizer()
dataset, dataloader = load_datasets(tokenizer, DATASET_PATH, NUM_CLASSES, CSV_DELIMITER)

if not os.path.exists(PATH):
    model, loss_fn, optimizer = init_model(PATH, NUM_CLASSES)
else:
    model, loss_fn, optimizer = load_model(PATH)

model=finetune(5, dataloader, model, loss_fn, optimizer)
save_model(PATH, model, optimizer, NUM_CLASSES)

