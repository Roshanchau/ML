from transformers import AutoTokenizer;

tokenizer = AutoTokenizer.from_pretrained(
    "bert-base-uncased"
)

text= "I am learning NLP";

tokens= tokenizer.tokenize(text);

print(tokens);

ids = tokenizer.encode(text)

print(ids)

output = tokenizer(
    text,
    padding=True,
    truncation=True,
    return_tensors="pt"
)

print(output)