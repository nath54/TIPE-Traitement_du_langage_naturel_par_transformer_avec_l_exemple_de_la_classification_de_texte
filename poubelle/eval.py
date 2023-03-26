from lib import *

PATH = "../bert/sentiments_initial_v4.pt"
DATASET_PATH = "../datasets/sentiment_initial/test.tsv"
CSV_DELIMITER = "\t"
NUM_CLASSES = 1

tokenizer = load_tokenizer()
dataset, dataloader = load_datasets(tokenizer, DATASET_PATH, NUM_CLASSES, CSV_DELIMITER)

if not os.path.exists(PATH):
    model, loss_fn, optimizer = init_model(PATH, NUM_CLASSES)
else:
    model, loss_fn, optimizer = load_model(PATH)


loop=enumerate(dataloader)

accuracies = []
losses = []

model = model.to(DEVICE)
for batch, dl in loop:
    ids=dl['ids'].to(DEVICE)
    token_type_ids=dl['token_type_ids'].to(DEVICE)
    mask=dl['mask'].to(DEVICE)
    label=dl['target'].to(DEVICE)
    label = label.unsqueeze(1)
    output=model(
        ids=ids,
        mask=mask,
        token_type_ids=token_type_ids).to(DEVICE)
    label = label.type_as(output)
    loss=loss_fn(output,label)
    pred = torch.where(output >= 0, 1, 0)
    num_correct = sum(1 for a, b in zip(pred, label) if a[0] == b[0])
    num_samples = pred.shape[0]
    accuracy = num_correct/num_samples
    accuracies.append(accuracy)
    losses.append(loss)
    print(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}')

A = sorted(accuracies)
print(f"Min accuracy : {A[0]}")
print(f"Max accuracy : {A[-1]}")
print(f"Mediane accuracy : {A[len(A)//2]}")
print(f"Moyenne accuracy : {sum(accuracies)/len(accuracies)}")

L = sorted(losses)
print(f"Min loss : {L[0]}")
print(f"Max loss : {L[-1]}")
print(f"Mediane loss : {L[len(L)//2]}")
print(f"Moyenne loss : {sum(losses)/len(losses)}")

