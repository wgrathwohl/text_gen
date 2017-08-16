from __future__ import print_function
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import json

class TextDataset(Dataset):
    def __init__(self, vocab_file, text_file, transform=None):
        self.transform = transform
        self.vocab = {}
        ind = 0
        with open(vocab_file, 'r') as f:
            for line in f:
                line = line.strip()
                self.vocab[line] = ind
                ind += 1

        with open(text_file, 'r') as f:
            self.lines = [line.strip() for line in f]

        self.examples = []
        for line in self.lines:
            l_vec = []
            for word in line.split():
                if word in self.vocab:
                    l_vec.append(self.vocab[word])
                else:
                    l_vec.append(self.vocab['<unk>'])
            l_vec.append(self.vocab['<eos>'])
            self.examples.append(np.array(l_vec, dtype=np.int32))
        self.ind2word = {v: k for k, v in self.vocab.items()}

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        sample = self.examples[idx]
        if self.transform:
            return self.transform(sample)
        else:
            return sample



class YelpDataset(Dataset):

    def __init__(self, vocab_file, fields,
                 data_file='/ais/gobi5/roeder/datasets/yelp_reviews/all_json.txt',
                 transform=None):
        self.transform = transform
        self.fields = fields
        self._curr_file = self._load(data_file)
        self._json = []
        print("Loading dataset into memory, this may take a while...")
        for x in self._curr_file:
            if x:
                self._json.append(json.loads(x))
        print("Done loading dataset.")
        self.vocab = {}
        self._load_vocab(vocab_file)

    def _load_vocab(self, vocab_file):
        vocab = {}
        with open(vocab_file, 'rb') as f:
            for idx, line in enumerate(f):
                line = line.strip()
                self.vocab[line] = idx
        return vocab

    def _load(self, file_name):
        return open(file_name)

    def __len__(self):
        return len(self._json)

    def __getitem__(self, idx):
        json_line = self._json[idx]
        example = dict()

        for field in self.fields:
            example[field] = json_line[field]
        return example


def pad_batch(examples):
    # sort in decreasing order
    examples = [ex['text'] for ex in examples]
    examples.sort(key=lambda s: -1 * len(s))
    lens = np.array([len(s) for s in examples], dtype=np.int64)
    max_len = max(lens)
    batch_size = len(examples)
    # make batch
    batch = np.zeros((batch_size, max_len), dtype=np.int64)
    for j, ex in enumerate(examples):
        l = len(ex)
        batch[j, :l] = ex

    return torch.from_numpy(batch), lens


# sanity test
#test_ds = YelpDataset(vocab_file="yelp_data/vocab.txt", fields=['text', 'categories', 'city', 'cool'])
#test_loader = DataLoader(test_ds, batch_size=64, shuffle=True, num_workers=4, collate_fn=pad_batch)
#for batch in test_loader:
#    print(batch['categories'][0])

