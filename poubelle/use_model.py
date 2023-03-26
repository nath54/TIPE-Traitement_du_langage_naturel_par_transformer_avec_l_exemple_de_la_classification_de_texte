from lib import *

PATH = "../bert/sentiments_initial_v4.pt"
NUM_CLASSES = 1

method = "input" # file

if not os.path.exists(PATH):
    raise UserWarning(f"The model checkpoints {PATH} doesn't exists !")

tokenizer = load_tokenizer()
model, loss_fn, optimizer = load_model(PATH)
model = model.to(DEVICE)

def main_inputs_loop():
    i = input("\n\nInput to classify : ")
    while i != "q":
        if i == "": continue
        #
        tt = tokenize_text(i, tokenizer)

        ids=torch.tensor([tt['input_ids']]).to(DEVICE)
        token_type_ids=torch.tensor([tt['token_type_ids']]).to(DEVICE)
        mask=torch.tensor([tt['attention_mask']]).to(DEVICE)

        output=model(
            ids=ids,
            mask=mask,
            token_type_ids=token_type_ids)
        pred = torch.where(output >= 0, 1, 0)
        print("Output : ", output, pred)
        #
        i = input("Input to classify : ")

if __name__ == "__main__":
    if method == "input":
        main_inputs_loop()
    else:
        print("Not implemented yet")
        pass
