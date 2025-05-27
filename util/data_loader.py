import os 
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    

import torch
from torchtext.datasets import Multi30k
from torch.utils.data import DataLoader as TorchDataLoader, Dataset
from datasets import load_dataset
from collections import Counter


class DataLoader:
    def __init__(self,ext, tokenize_en,tokenize_de, init_token, eos_token):
        self.ext = ext
        self.tokenize_en = tokenize_en
        self.tokenize_de = tokenize_de
        self.init_token = init_token
        self.eos_token = eos_token
        self.source_vocab = {}
        self.target_vocab = {}
        
    def make_dataset(self):
        train_data, valid_data, test_data = Multi30k(split=('train', 'valid', 'test'))
        return train_data,valid_data,test_data
    
    def build_vocab(self,train_data,min_freq):
        source_words = Counter()
        target_words = Counter()
        for example in train_data:
            src_sentence, trg_sentence = example

            if self.ext == ('.de', '.en'):
                src_tokens = self.tokenize_de(src_sentence)
                trg_tokens = self.tokenize_en(trg_sentence)
            else:
                src_tokens = self.tokenize_en(src_sentence)
                trg_tokens = self.tokenize_de(trg_sentence)

            source_words.update(src_tokens)
            target_words.update(trg_tokens)
        
        special = ['<pad>', '<unk>', self.init_token, self.eos_token]
        self.source_vocab = {word:i for i,word in enumerate(special + [w for w,c in source_words.items() if c >= min_freq])}
        self.target_vocab = {word:i for i,word in enumerate(special + [w for w,c in target_words.items() if c >= min_freq])}
        
    def make_iter(self,train,validate,test,batch_size,device):
        train_list = list(train)
        train_iter = TorchDataLoader(
            SimpleDataset(train_list,self), batch_size=batch_size,shuffle=True,
            collate_fn=lambda x: self._collate(x,device)
        )
        valid_iter = TorchDataLoader(
            SimpleDataset(validate,self),batch_size=batch_size, shuffle=False,
            collate_fn=lambda x:self._collate(x,device)
        )
        test_iter = TorchDataLoader(
            SimpleDataset(test,self), batch_size=batch_size,shuffle=False,
            collate_fn=lambda x:self._collate(x,device)
        )
        return train_iter,valid_iter,test_iter
    
    def _collate(self,batch,device):
        src_batch,trgt_batch = zip(*batch)
        src_batch = list(src_batch)
        trgt_batch = list(trgt_batch)
        max_src = max(len(s) for s in src_batch)
        max_trg = max(len(t) for t in trgt_batch)
        
        pad_id = 0
        
        src_padded = [s + [pad_id]* (max_src - len(s)) for s in src_batch ]
        trg_padded = [t + [pad_id] * (max_trg - len(t)) for t in trgt_batch]
        
        class Batch:
            def __init__(self,src,trg):
                self.src = torch.tensor(src).to(device)
                self.trg = torch.tensor(trg).to(device)
            
        return Batch(src_padded,trg_padded)

class SimpleDataset(Dataset):
    def __init__(self, data, dataloader):
        self.data = data
        self.dataloader = dataloader
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        example = self.data[idx]
    # Ensure it's a tuple of (de, en)
        if not isinstance(example, tuple) or len(example) != 2:
            raise ValueError(f"Expected tuple of 2 (de, en), but got: {example}")
        src_text = example[0].strip()  # 'de' sentence
        trg_text = example[1].strip()  # 'en' sentence

        src_tokens = self.dataloader.tokenize_de(src_text)
        trg_tokens = self.dataloader.tokenize_en(trg_text)
        
        src_ids = [self.dataloader.source_vocab.get(self.dataloader.init_token, 2)] + \
                [self.dataloader.source_vocab.get(t,1) for t in src_tokens] + \
                [self.dataloader.source_vocab.get(self.dataloader.eos_token, 3)]
        trg_ids = [self.dataloader.target_vocab.get(self.dataloader.init_token, 2)] + \
                [self.dataloader.target_vocab.get(t,1) for t in trg_tokens] + \
                [self.dataloader.target_vocab.get(self.dataloader.eos_token, 3)]

        return src_ids,trg_ids
            
        