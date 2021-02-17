import torch
import numpy as np

class ClassifyCollator(object):
    def __init__(self,max_seq_len):
        self.max_seq_len = max_seq_len

    def __call__(self, batch):
        return self.dynamic_pad(batch,self.max_seq_len)

    def dynamic_pad(self,batch,upper_bound=512):
        '''

        :param batch:
        :param upper_bound:最多允许的sqe长度
        :return:
        '''
        tok_ids_arr,token_type_ids, attention_mask,labels = zip(*batch)
        max_len = max(list(len(t) for t in tok_ids_arr))
        max_len = min(max_len,upper_bound)
        return self._pad_and_convert(tok_ids_arr,token_type_ids,attention_mask,labels,max_len)

    def _pad(self,tok_ids_arr,max_len):
        padding_tok_arr = []
        for i, tok_ids in enumerate(tok_ids_arr):
            if len(tok_ids) >= max_len:
                padding_tok_arr.append(tok_ids[:max_len])
            else:
                tmp_arr = tok_ids + (max_len - len(tok_ids)) * [0]
                padding_tok_arr.append(tmp_arr)
        return torch.tensor(padding_tok_arr)


    def _pad_and_convert(self,tok_ids_arr,token_type_ids,attention_mask,labels,max_len):
        return self._pad(tok_ids_arr,max_len),self._pad(attention_mask),self._pad(token_type_ids), torch.tensor(labels),