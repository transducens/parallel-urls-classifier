
import logging
import multiprocessing

import parallel_urls_classifier.utils.utils as utils

import torch
from torch.utils.data import (
    Sampler,
    Dataset,
    DataLoader,
)
import numpy as np
import more_itertools

logger = logging.getLogger("parallel_urls_classifier")

# Original code from https://www.kaggle.com/code/rhtsingh/speeding-up-transformer-w-optimization-strategies?scriptVersionId=67176227&cellId=2

class SmartBatchingURLsDataset(Dataset):
    def __init__(self, parallel_urls, non_parallel_urls, tokenizer, max_length, regression=False):
        super(SmartBatchingURLsDataset, self).__init__()

        self.max_length = max_length
        self.pad_token_id = tokenizer.pad_token_id
        self.regression = regression

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
        self.labels = self.labels.type(torch.float) if regression else self.labels.type(torch.long)

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        #return {"url_str": self.data[idx], "label": self.labels[idx]}
        #return {
        #    "url_tokens": self.tokens[idx],
        #    #"url_attention_mask": self.attention_mask[idx],
        #    "label": self.labels[idx],
        #}
        return self.tokens[idx], self.labels[idx]

    def get_dataloader(self, batch_size, device, force_cpu, num_workers, sampler=None, over_sampling=False, classes_weights=None):
        is_device_gpu = device.type.startswith("cuda")

        if sampler:
            self.sampler = sampler
        else:
            self.sampler = SmartBatchingSampler(
                data_source=self.tokens,
                batch_size=batch_size,
                regression=self.regression,
                data_labels=self.labels,
                over_sampling=over_sampling,
                classes_weights=classes_weights,
            )

        collate_fn = SmartBatchingCollate(
            targets=self.labels,
            max_length=self.max_length,
            pad_token_id=self.pad_token_id,
        )
        num_workers = 0 if is_device_gpu else multiprocessing.cpu_count() - 1 \
                        if num_workers < 0 else num_workers # 0 is the adviced value in https://pytorch.org/docs/stable/data.html
                                                            #  when GPU is being used
        dataloader_kwargs = {
            "pin_memory": True,
            "pin_memory_device": device.type,
            "num_workers": num_workers,
        }

        # Check if we can use recent pytorch features
        pytorch_major, pytorch_minor, pytorch_patch = utils.get_pytorch_version()

        if pytorch_major > 1 or (pytorch_major == 1 and pytorch_minor >= 12):
            # Ok
            pass
        else:
            logger.warning("Unexpected pytorch version: making some changes in DataLoader")

            del dataloader_kwargs["pin_memory_device"]

            if force_cpu:
                # pin_memory uses GPU if available

                dataloader_kwargs["pin_memory"] = False

        dataloader = DataLoader(
            dataset=self,
            batch_size=batch_size,
            sampler=self.sampler,
            collate_fn=collate_fn,
            **dataloader_kwargs,
        )

        return dataloader

    @property
    def total_tokens(self):
        return self._total_tokens

class SmartBatchingSampler(Sampler):
    def __init__(self, data_source, batch_size, regression=False, over_sampling=False, data_labels=None, classes_weights=None):
        super(SmartBatchingSampler, self).__init__(data_source)

        self.len = len(data_source)
        self.regression = regression
        sample_lengths = [len(seq) for seq in data_source]
        argsort_inds = np.argsort(sample_lengths) # Get indexes of tokens sorted by length
        self.batches = list(more_itertools.chunked(argsort_inds, n=batch_size)) # Batches of indexes sorted by tokens length
        self._backsort_inds = None
        self.over_sampling = over_sampling
        self.classes_weights = classes_weights
        self.data_labels = data_labels.cpu().detach()
        self.data_labels = torch.round(self.data_labels).type(torch.long) if self.regression else self.data_labels

        if over_sampling:
            if data_labels is None or classes_weights is None:
                raise Exception("In order to apply over-sampling, data_labels and classes_weights have to be provided")

            if self.len != len(data_labels):
                raise Exception(f"Data length is different from the data labels length: {self.len} vs {len(data_labels)}")

    def __iter__(self):
        _batches = self.batches

        if _batches:
            last_batch = _batches.pop(-1) # Remove last element before randomizing since its length might be less than the batch size

            np.random.shuffle(_batches) # Randomize batches
            _batches.append(last_batch) # Add the previously removed last element

            if self.over_sampling:
                for chunk_idx, chunk in enumerate(_batches):
                    labels = np.array(self.data_labels)[chunk]

                    if len(np.unique(labels)) <= 1:
                        # Do not modify the batch if is not necessary
                        continue

                    _labels = torch.tensor([self.classes_weights[l] for l in labels])
                    idxs = torch.multinomial(_labels, len(chunk), True)
                    _batches[chunk_idx] = np.array(_batches[chunk_idx])[idxs].tolist()

        self._inds = list(more_itertools.flatten(_batches))

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
            sequences, targets = list(zip(*batch))
        else:
            sequences = list(batch)

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
