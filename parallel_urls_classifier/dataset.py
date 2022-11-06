
import multiprocessing

import parallel_urls_classifier.utils.utils as utils

import torch
from torch.utils.data import (
    Sampler,
    Dataset,
    DataLoader,
)
import numpy as np

class SmartBatchingURLsDataset(Dataset):
    def __init__(self, parallel_urls, non_parallel_urls, tokenizer, max_length, regression=False):
        super(SmartBatchingURLsDataset, self).__init__()

        self.max_legnth = max_length
        self.pad_token_id = tokenizer.pad_token_id

        #self.data = torch.stack(non_parallel_urls + parallel_urls).squeeze(1) # TODO problem here when creating a new tmp array -> big arrays will lead to run out of memory...
        #self.data = non_parallel_urls + parallel_urls

        # Tokenize data (we need to tokenize one by one because the length of all the provided URLs will not be the same)
        self.tokens = [utils.encode(tokenizer, url, max_length=max_length)["input_ids"][0].tolist()
                       for url in non_parallel_urls + parallel_urls]
        self._total_tokens = sum([len(t) for t in self.tokens])
        #self.attention_mask = data["attention_mask"]
        #self.tokens = [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(url)) for url in non_parallel_urls + parallel_urls]
        self.labels = np.zeros(len(self.tokens))

        #self.size_gb = self.data.element_size() * self.data.nelement() / 1000 / 1000 / 1000

        #self.labels[:len(non_parallel_urls)] = 0
        # Set to 1 the parallel URLs
        self.labels[len(non_parallel_urls):] = 1

        # Postprocess labels
        self.labels = torch.from_numpy(self.labels)
        self.labels = self.labels.type(torch.FloatTensor) if regression else self.labels.type(torch.LongTensor)

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        #return {"url_str": self.data[idx], "label": self.labels[idx]}
        return {
            "url_tokens": self.tokens[idx],
            #"url_attention_mask": self.attention_mask[idx],
            "label": self.labels[idx],
        }

    def get_dataloader(self, batch_size, device, sampler=None):
        if sampler:
            self.sampler = sampler
        else:
            self.sampler = SmartBatchingSampler(
                data_source=self.tokens,
                batch_size=batch_size,
            )

        collate_fn = SmartBatchingCollate(
            targets=self.labels,
            max_length=self.max_length,
            pad_token_id=self.pad_token_id,
        )
        dataloader = DataLoader(
            dataset=self,
            batch_size=batch_size,
            sampler=self.sampler,
            collate_fn=collate_fn,
            num_workers=multiprocessing.cpu_count() - 1,
            pin_memory=True,
            pin_memory_device=device.type,
        )

        return dataloader

    @property
    def total_tokens(self):
        return self._total_tokens

class SmartBatchingSampler(Sampler):
    def __init__(self, data_source, batch_size):
        super(SmartBatchingSampler, self).__init__(data_source)

        self.len = len(data_source)
        sample_lengths = [len(seq) for seq in data_source]
        argsort_inds = np.argsort(sample_lengths) # Get indexes of tokens sorted by length
        self.batches = list(more_itertools.chunked(argsort_inds, n=batch_size)) # Batches of indexes sorted by tokens length
        self._backsort_inds = None

    def __iter__(self):
        if self.batches:
            last_batch = self.batches.pop(-1) # Remove last element before randomizing since its length might be less than the batch size

            np.random.shuffle(self.batches) # Randomize batches
            self.batches.append(last_batch) # Add the previously removed last element

        self._inds = list(more_itertools.flatten(self.batches))

        yield from self._inds # Return index of the randomized batches flattened but sorted by tokens length

    def __len__(self):
        return self.len

    @property
    def backsort_inds(self):
        if self._backsort_inds is None:
            self._backsort_inds = np.argsort(self._inds)

        return self._backsort_inds

class SmartBatchingCollate:
    def __init__(self, targets, max_length, pad_token_id):
        self._targets = targets
        self._max_length = max_length
        self._pad_token_id = pad_token_id

    def __call__(self, batch):
        targets = None

        if self._targets is not None:
            sequences = batch["url_tokens"]
            targets = batch["label"]
        else:
            sequences = batch["url_tokens"]

        input_ids, attention_mask = self.pad_sequence(sequences, self._max_length, self._pad_token_id)

        #if self._targets is not None:
        #    output = input_ids, attention_mask, torch.tensor(targets)
        #else:
        #    output = input_ids, attention_mask

        output = {
            "url_tokens": input_ids,
            "url_attention_mask": attention_mask,
        }

        if self._targets is not None:
            output["label"] = torch.tensor(targets)

        return output

    def pad_sequence(self, sequence_batch, max_sequence_length, pad_token_id):
        max_batch_len = max(len(sequence) for sequence in sequence_batch)
        max_len = min(max_batch_len, max_sequence_length)
        padded_sequences, attention_masks = [], []
        attend, no_attend = 1, 0

        for sequence in sequence_batch:
            # As discussed above, truncate if exceeds max_len
            new_sequence = list(sequence[:max_len])

            attention_mask = [attend] * len(new_sequence)
            pad_length = max_len - len(new_sequence)

            new_sequence.extend([pad_token_id] * pad_length)
            attention_mask.extend([no_attend] * pad_length)

            padded_sequences.append(new_sequence)
            attention_masks.append(attention_mask)

        padded_sequences = torch.tensor(padded_sequences)
        attention_masks = torch.tensor(attention_masks)

        return padded_sequences, attention_masks
