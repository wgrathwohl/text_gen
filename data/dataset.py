from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np

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


def pad_batch(examples):
    lens = np.array([len(s) for s in examples], dtype=np.int64)
    max_len = max(lens)
    batch_size = len(examples)
    # make batch
    batch = np.zeros((batch_size, max_len), dtype=np.int64)
    final_inds = np.zeros((batch_size, max_len), dtype=np.int64)
    for j, ex in enumerate(examples):
        l = len(ex)
        batch[j, :l] = ex
        final_inds[j, l-1] = 1
    return torch.from_numpy(batch), torch.from_numpy(final_inds)

#
# test_ds = TextDataset("yelp_data/vocab.txt", "yelp_data/part_0")
# test_loader = DataLoader(test_ds, batch_size=64, shuffle=True, num_workers=4, collate_fn=pad_batch)
# for batch, lens in test_loader:
#     print(batch[:4], lens[:4])
#     print(batch.size(), lens.size())
#     break





