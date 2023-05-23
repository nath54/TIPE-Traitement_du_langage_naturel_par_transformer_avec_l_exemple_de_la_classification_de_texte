import transformers

# Quelques exemples de tokenizer
tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased")

sentence = "I love neural networks"

while sentence != "q":
    sentence = input("\n\n : ")

    tokens = tokenizer.encode_plus(
                sentence,
                None,
                padding="max_length",
                add_special_tokens=True,
                return_attention_mask=True,
                max_length=16,
                truncation=True
            )


    print("SENTENCE : ", sentence)
    print("TOKENS : ", tokens)
    print("DETAILS : ", tokenizer.tokenize(sentence))

