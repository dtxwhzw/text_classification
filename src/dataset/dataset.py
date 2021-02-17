import pandas as pd
import numpy as np
import json
import torch
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self):
        super(MyDataset,self).__init__()

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self,index):
        if torch.is_tensor(index):
            index = index.to_list()
        if isinstance(index,slice):
            begin, end, step = index.indices(len(self))
            return [self.get_example(i) for i in range(begin, end ,step)]
        if isinstance((index,list)):
            return [self.get_example(i) for i in index]
        else:
            return self.get_example(index)

    def get_example(self,i):
        return NotImplementedError


class SingleLabel(MyDataset):
    def __init__(self,path,dictionary=None,max_length=128,tokenizer=None,word=False):
        super(SingleLabel).__init__()
        self.data = pd.read_csv(path,sep='\t').dropna()
        with open(config.root_path + '/data/label2id.json','r') as f:
            self.label_map = json.load(f)
        self.data['catrgory_id'] = self.data['label'].map(self.label2id)
        if not word:
            self.data['text'] = self.data['text'].apply(
                lambda x: " ".join("".join(x.split())))
        if tokenizer is not None:
            self.model_name = 'bert'
            self.tokenizer = tokenizer
        else:
            self.model_name = 'normal'
            self.tokenizer = dictionary
        self.max_length = max_length

    def __len__(self):
        return self.data.shape[0]

    def get_example(self,i):
        data = self.data.iloc[i]
        text = data['text']
        labels = int(data['category_id'])
        attention_mask, token_type_ids = [0],[0]
        if 'bert' in self.model_name:
            text_dict = self.tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=self.max_length,
                return_attention_mask=True
            )
            input_ids, attention_mask, token_type_ids = text_dict['input_ids'],text_dict['attention_mask'],text_dict['token_type_ids']
        else:
            text = text.split()
            text = text + [0] * max(0,self.max_length - len(text)) if len(text) < self.max_length else text[:self.max_length]
            input_ids = [self.tokenizer.indexer(x) for x in text]

        output = {
            'token_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
            'labels': labels
        }
        return output
