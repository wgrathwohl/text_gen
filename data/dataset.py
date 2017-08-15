from __future__ import print_function
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
                 files_dir='/ais/gobi5/roeder/datasets/yelp_reviews/pickles',
                 transform=None):
        self.transform = transform
        self.fields = fields
        self._file_names = self._get_filenames(files_dir)
        self._file_counts = dict()
        self._n_examples = self._count_file_lines() # side effect: populates self._file_counts
        self._n_files = len(self._file_names)
        self._file_idx = 0
        self._n_seen_prev_files = 0
        self._example_idx = 0
        self._curr_file_name = self._file_names[self._file_idx]
        self._n_lines_curr_file = self._file_counts[self._curr_file_name]
        self._curr_data = self._load(self._curr_file_name)
        # self.vocab = self._load_vocab(vocab_file)

    def _load_vocab(self, vocab_file):
        vocab = {}
        with open(vocab_file, 'rb') as f:
            for idx, line in enumerate(f):
                line = line.strip()
                self.vocab[line] = idx
        return vocab

    def _count_lines(self, file):
        count = 0
        with open(file, 'rb') as f:
            f = pickle.load(f)
            for _ in f:
                count += 1
        return count

    def _count_file_lines(self):
        for file_name in self._file_names:
            self._file_counts[file_name] = self._count_lines(file_name)
        return sum(self._file_counts.values())



    def _get_filenames(self, files_dir, file_pattern='reviews.pickle.*'):
        return glob.glob(os.path.join(files_dir, file_pattern))


    def _last_idx(self, file):
        return len(file) - 1

    def _load(self, file_name):
        with open(file_name, 'r') as f:
            return pickle.load(f)

    def _load_next(self):

        # update total seen with lines in previous file
        self._n_seen_prev_files += self._n_lines_curr_file

        # update references and counts to next file
        self._file_idx += 1
        self._curr_file_name = self._file_names[self._file_idx]
        self._n_lines_curr_file = self._file_counts[self._curr_file_name]

        return self._load(self._curr_file_name)

    def __len__(self):
        return self._n_examples

    def __getitem__(self, idx):
        """ returns line at idx from current data file if any remain,
            or opens next file and returns first """
        # convert total data index to current file index
        curr_file_idx = idx - self._n_seen_prev_files

        #
        if curr_file_idx >= self._n_lines_curr_file:
            # self._curr_data has no more examples
            print("Switching from file {} to {}".format(self._file_idx, self._file_idx+1))

            # open and decode new file
            self._curr_data = self._load_next()

            # correct the index and sanity check
            curr_file_idx = idx - self._n_seen_prev_files
            assert(curr_file_idx == 0)
        print("Batch reads {} using idx {} from file {}".format(idx, curr_file_idx, self._file_idx))
        print("Current file has {} lines".format(self._n_lines_curr_file))

        line = self._curr_data[curr_file_idx]
        example = dict()
        for field in self.fields:
            example[field] = line[field]
        return example


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

test_ds = YelpDataset(vocab_file="yelp_data/vocab.txt", fields=['categories','city','cool'], files_dir="./yelp_data/test_pickles")
# NOTE: preshuffled data. Set shuffle=False or hacked forward read will fail.
n_batches = 64
batch_size = len(test_ds) / n_batches
test_loader = DataLoader(test_ds, batch_size=n_batches, shuffle=False, num_workers=1)
for batch in test_loader:
    print(len(batch['categories'][0]))





