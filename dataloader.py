import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import pandas as pd
import numpy as np
import spacy
from tqdm import tqdm
import re
import time
import pickle
import os
import gensim
import unicodedata

def get_data():
    
    # Creating custom training dataset
    class TrainDataset(Dataset):
        def __init__(self, train_queries, train_intents, train_slots, n_intents, int_dict, transform=None):
            self.tq = train_queries
            self.ti = train_intents
            self.ts = train_slots
            self.n_samples = len(self.tq)
            self.n_intents = n_intents
            self.int_dict = int_dict
            self.transform = transform
        def __getitem__(self, index):
            if self.transform:
                return self.transform(self.x[index]), self.y[index]
            else:
                arr = np.zeros(self.n_intents)
                arr[self.int_dict[self.ti[index]]] = 1.0
                arr = torch.from_numpy(arr)
                return self.tq[index], self.ts[index], arr
        def __len__(self):
            return self.n_samples

    # Creating custom validation dataset
    class ValDataset(Dataset):
        def __init__(self, val_queries, val_intents, val_slots, n_intents, int_dict, transform=None):
            self.vq = val_queries
            self.vi = val_intents
            self.vs = val_slots
            self.n_samples = len(self.vq)
            self.n_intents = n_intents
            self.int_dict = int_dict
            self.transform = transform
        def __getitem__(self, index):
            if self.transform:
                return self.transform(self.x[index]), self.y[index]
            else:
                arr = np.zeros(self.n_intents)
                arr[self.int_dict[self.vi[index]]] = 1.0
                arr = torch.from_numpy(arr)
                return self.vq[index], self.vs[index], arr
        def __len__(self):
            return self.n_samples

    # Creating custom testing dataset
    class TestDataset(Dataset):
        def __init__(self, test_queries, test_intents, test_slots, n_intents, int_dict, transform=None):
            self.tq = test_queries
            self.ti = test_intents
            self.ts = test_slots
            self.n_samples = len(self.tq)
            self.n_intents = n_intents
            self.int_dict = int_dict
            self.transform = transform
        def __getitem__(self, index):
            if self.transform:
                return self.transform(self.x[index]), self.y[index]
            else:
                arr = np.zeros(self.n_intents)
                arr[self.int_dict[self.ti[index]]] = 1.0
                arr = torch.from_numpy(arr)
                return self.tq[index], self.ts[index], arr
        def __len__(self):
            return self.n_samples
        
        
    def load_ds(fname):
        with open(fname, 'rb') as stream:
            ds,dicts = pickle.load(stream)
        return ds,dicts
    
    def basic_clean_word(text):
        text = (unicodedata.normalize('NFKD', text)
                .encode('ascii', 'ignore')
                .decode('utf-8', 'ignore')
                .lower())
        word = re.sub(r'[^\w\s]', '', text)
        return word
    
    # Reading training data
    def get_train_val_data():
        train_ds, dicts = load_ds(os.path.join(DATA_DIR,'atis.train.pkl'))
        t2i, s2i, in2i = map(dicts.get, ['token_ids', 'slot_ids','intent_ids'])
        i2t, i2s, i2in = map(lambda d: {d[k]:k for k in d.keys()}, [t2i,s2i,in2i])
        query, slots, intent =  map(train_ds.get, ['query', 'slot_labels', 'intent_labels'])
        train_queries = []
        train_intents = []
        train_slots = []
        for i in range(len(query)):
            q = []
            s = []
            for j in range(1, len(query[i])-1):
                q.append(basic_clean_word(i2t[query[i][j]]))
                s.append(i2s[slots[i][j]])
            train_intents.append(i2in[intent[i][0]])
            train_queries.append(q)
            train_slots.append(s)
        return train_queries[:-500], train_queries[-500:], train_intents[:-500], train_intents[-500:], train_slots[:-500], train_slots[-500:]

    # Reading testing data
    def get_test_data():
        test_ds, dicts = load_ds(os.path.join(DATA_DIR,'atis.test.pkl'))
        t2i, s2i, in2i = map(dicts.get, ['token_ids', 'slot_ids','intent_ids'])
        i2t, i2s, i2in = map(lambda d: {d[k]:k for k in d.keys()}, [t2i,s2i,in2i])
        query, slots, intent =  map(test_ds.get, ['query', 'slot_labels', 'intent_labels'])
        test_queries = []
        test_intents = []
        test_slots = []
        for i in range(len(query)):
            q = []
            s = []
            for j in range(1, len(query[i])-1):
                q.append(basic_clean_word(i2t[query[i][j]]))
                s.append(i2s[slots[i][j]])
            test_intents.append(i2in[intent[i][0]])
            test_queries.append(q)
            test_slots.append(s)
        return test_queries, test_intents, test_slots
    
    DATA_DIR="C:/Users/dhruv/Desktop/ASP sem 1/Capstone/Data/ATIS2/"
    train_queries, val_queries, train_intents, val_intents, train_slots, val_slots = get_train_val_data()
    test_queries, test_intents, test_slots = get_test_data()
    _, dicts = load_ds(os.path.join(DATA_DIR,'atis.train.pkl'))
    int_dict = dicts['intent_ids']
    slot_dict = dicts['slot_ids']
    n_intents = 26
    
    # Creating dataset objects
    train_dataset = TrainDataset(train_queries, train_intents, train_slots, n_intents, int_dict)
    val_dataset = ValDataset(val_queries, val_intents, val_slots, n_intents, int_dict)
    test_dataset = TestDataset(test_queries, test_intents, test_slots, n_intents, int_dict)
    batch_size = None

    # Implementing train loader to split the data into batches
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              shuffle=True, # data reshuffled at every epoch
                              num_workers=0) # Use several subprocesses to load the data

    # Implementing train loader to split the data into batches
    val_loader = DataLoader(dataset=val_dataset,
                              batch_size=batch_size,
                              shuffle=True, # data reshuffled at every epoch
                              num_workers=0) # Use several subprocesses to load the data

    # Implementing train loader to split the data into batches
    test_loader = DataLoader(dataset=test_dataset,
                              batch_size=batch_size,
                              shuffle=True, # data reshuffled at every epoch
                              num_workers=0) # Use several subprocesses to load the data
    
    return train_loader, val_loader, test_loader, int_dict, slot_dict, n_intents