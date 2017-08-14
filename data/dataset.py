from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import glob
import os
import cPickle as pickle

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

    def __init__(self, vocab_file, text_file,
                 files_dir='/ais/gobi5/roeder/datasets/yelp_reviews/pickles',
                 transform=pickle.load):
        self.transform = transform
        self._files = self._get_filenames(files_dir)
        self.vocab = self._load_vocab(vocab_file)
        self.n_batches = len(self.files)
        self.batch_number = 0
        self.batch_gen = self._batch_gen(self._files)

        with open(text_file, 'r') as f:
            self.lines = [line.strip() for line in f]

    def _load_vocab(self, vocab_file):
        vocab = {}
        ind = 0
        with open(vocab_file, 'r') as f:
            for line in f:
                line = line.strip()
                self.vocab[line] = ind
                ind += 1
        return vocab

    def _get_filenames(self, files_dir, file_pattern='reviews.pickle.*'):
        return glob.glob(os.path.join(files_dir, file_pattern))

    def _batch_gen(self, files):
        """ lazily returns the next batch as list of dict"""
        for file in files:
            with open(file, "rb") as f:
                yield self.transform(f)

    def next(self):
        """ continue generating batches until none left"""
        try:
            self.current_batch = next(self.batch_gen)
            self._process_batch(self)
            self.batch_number += 1
            return True
        except StopIteration:
            # todo: better way to indicate endpoint?
            return False

    def get_batch_text(self):
        return self._current_text

    def get_batch_json(self):
        return self._current_json

    def _process_batch(self, batch):
        self._current_text = [json['text'] for json in self.current_batch]
        self._current_json = self.current_batch

    def __len__(self):
        """ returns length of current batch"""
        return len(self.batch)

    def __getitem__(self, idx):
        # returns text, json from current batch
        return self._current_text[idx], self._current_json[idx]


def pad_batch(examples):
    lens = np.array([len(s) for s in examples], dtype=np.int64)
    max_len = max(lens)
    batch_size = len(examples)
    # make batch
    batch = np.zeros((batch_size, max_len), dtype=np.int64)
    final_inds = np.zeros((batch_size, max_len), dtype=np.int64)
    mask = np.zeros((batch_size, max_len), dtype=np.int64)
    for j, ex in enumerate(examples):
        l = len(ex)
        batch[j, :l] = ex
        final_inds[j, l-1] = 1
        mask[j, :l] = 1
    return torch.from_numpy(batch), torch.from_numpy(final_inds), torch.from_numpy(mask)

#
# test_ds = TextDataset("yelp_data/vocab.txt", "yelp_data/part_0")
# test_loader = DataLoader(test_ds, batch_size=64, shuffle=True, num_workers=4, collate_fn=pad_batch)
# for batch, lens in test_loader:
#     print(batch[:4], lens[:4])
#     print(batch.size(), lens.size())
#     break





