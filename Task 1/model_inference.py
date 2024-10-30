import torch

from torch.utils.data import Dataset
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
tokenizer.add_tokens(['B-Mountain', 'I-Mountain'])
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
ids_to_label = {1: 'B-Mountain',
                2: 'I-Mountain',
                0: 'O'}

model = torch.load('mountain_ner_model.pt', map_location=device)

class process_sentence_single(Dataset):

    def __init__(self, text):
        self.text = text
        print("dataloader initialized")

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        sentence = self.text.strip().split()

        tokenized_sentence = []

        for word in sentence:
            # Tokenize the word and count # of subwords the word is broken into
            tokenized_word = tokenizer.tokenize(word)
            tokenized_sentence.extend(tokenized_word)

        sen_code = tokenizer.encode_plus(tokenized_sentence,
                                         add_special_tokens=True,  # Add [CLS] and [SEP]
                                         return_attention_mask=True,  # Generate the attention mask
                                         #             return_tensors = 'pt'
                                         )

        # step 4: turn everything into PyTorch tensors
        item = {key: torch.as_tensor(val) for key, val in sen_code.items()}
        return item


def make_single_pred(sentence):
    # get the processed input_ids and mask
    # test_text = "Mark is the ceo of Facebook. located in California ."
    test_text = sentence
    pre_text = process_sentence_single(test_text)
    text = pre_text[0]

    ids = text['input_ids']
    mask = text['attention_mask']

    # make prediction

    test_pred = model(input_ids=torch.unsqueeze(ids, 0).to(device), attention_mask=torch.unsqueeze(mask, 0).to(device))

    ## flatten prediction
    active_logits = test_pred[0].view(-1, model.num_labels)  # shape (batch_size * seq_len, num_labels)
    flattened_predictions = torch.argmax(active_logits, axis=1)  # shape (batch_size * seq_len,)


    # convert ids to corresponding tokens
    text_tokens = tokenizer.convert_ids_to_tokens(ids)

    # convert predctions to labels
    text_labels = []
    for i in flattened_predictions.squeeze(0).cpu().numpy():
        text_labels.append(ids_to_label.get(i))

    # remove first and last tokens ([CLS] and [SEP])
    text_tokens = text_tokens[1:-1]
    text_labels = text_labels[1:-1]

    print("\n printing tokens with labels")
    print(text_tokens)
    print(text_labels)

    return text_tokens, text_labels

print('Enter sentence:')
sentence = input()
txt, lbl = make_single_pred(sentence)