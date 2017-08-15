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
    """
    each batch is contained in one pickled file
    transform is set to pickle.load by default
    a loaded pickle file is a list of dict where each dict is one example
    """

    def __init__(self, vocab_file, fields,
                 file_dir='/ais/gobi5/roeder/datasets/yelp_reviews/pickles',
                 transform=None):
        self.transform = transform
        self.fields = fields
        self._curr_file = self._load(file_dir)
        self._json = []
        print("Loading dataset into memory, this may take a while...")
        for x in self._curr_file:
            if x:
                self._json.append(json.loads(x))
        print("Done loading dataset.")
        self.vocab = self._load_vocab(vocab_file)

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
    examples_text = [s['text'] for s in examples]
    lens = np.array([len(t) for t in examples_text], dtype=np.int64)
    max_len = max(lens)
    batch_size = len(examples_text)
    # make batch
    batch = np.zeros((batch_size, max_len), dtype=np.int64)
    final_inds = np.zeros((batch_size, max_len), dtype=np.int64)
    mask = np.zeros((batch_size, max_len), dtype=np.int64)
    for j, ex in enumerate(examples_text):
        l = len(ex)
        batch[j, :l] = ex
        final_inds[j, l-1] = 1
        mask[j, :l] = 1
    # ::todo:: return labels for classification
    return torch.from_numpy(batch), torch.from_numpy(final_inds), torch.from_numpy(mask)


#test_ds = YelpDataset(vocab_file="yelp_data/vocab.txt", fields=['text', 'categories', 'city', 'cool'],
#                      file_dir="./yelp_data/txt/mock_data.txt")
#test_loader = DataLoader(test_ds, batch_size=64, shuffle=True, num_workers=4, collate_fn=pad_batch)
#for batch in test_loader:
#    print(batch['categories'][0])

