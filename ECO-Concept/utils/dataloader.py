import json
import torch
from transformers import BertTokenizer
from transformers import RobertaTokenizer
from torch.utils.data import TensorDataset, DataLoader


def make_dataset(directory):
    file = open(directory, "r", encoding="utf-8")
    data = json.load(file)
    contents = []
    labels = []
    for item in data:
        contents.append(item['content'])
        labels.append(int(item['label']))
    return contents, labels


class TensorDataset(torch.utils.data.Dataset):
    def __init__(self, contents, labels, bert_path, base_model_name='bert', max_len=256):
        self.bert_path = bert_path
        self.max_len = max_len
        self.contents = contents
        self.labels = torch.tensor(labels)
        self.content_encs = word2input(contents, self.max_len, self.bert_path, base_model_name)

    def __len__(self):
        return len(self.contents)

    def __getitem__(self, idx):
        return self.content_encs[idx], self.labels[idx], self.contents[idx]


def word2input(texts, max_len, bert_path, base_model_name):
    if base_model_name == 'roberta':
        tokenizer = RobertaTokenizer.from_pretrained(bert_path)
    if base_model_name == 'bert':
        tokenizer = BertTokenizer.from_pretrained(bert_path)
    all_vecs = []
    for i, text in enumerate(texts):
        text_encs = tokenizer(
            text,
            add_special_tokens=True,
            max_length=max_len,
            truncation=True,
            padding='max_length',
            return_tensors="pt"
        )
        all_vecs.append(text_encs)
    return all_vecs
