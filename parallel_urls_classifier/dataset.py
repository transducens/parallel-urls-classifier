
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
import transformers

logger = logging.getLogger("parallel_urls_classifier")

# Original code from https://www.kaggle.com/code/rhtsingh/speeding-up-transformer-w-optimization-strategies?scriptVersionId=67176227&cellId=2

class SmartBatchingURLsDataset(Dataset):
    def __init__(self, parallel_urls, non_parallel_urls, tokenizer, max_length, regression=False, sampler_better_randomness=True,
                 remove_instead_of_truncate=False):
        super(SmartBatchingURLsDataset, self).__init__()

        self.max_length = max_length
        self.pad_token_id = tokenizer.pad_token_id
        self.regression = regression
        self.sampler_better_randomness = sampler_better_randomness
        self.dataloader = None

        #self.data = torch.stack(non_parallel_urls + parallel_urls).squeeze(1) # Problem here when creating a new tmp array -> big arrays will lead to run out of memory...
        #self.data = non_parallel_urls + parallel_urls

        # Tokenize data (we need to tokenize one by one because the length of all the provided URLs will not be the same)
        self.tokens = utils.encode(tokenizer, non_parallel_urls + parallel_urls, max_length=max_length, return_tensors=None, truncation=False)["input_ids"]

        # Truncate or remove
        if remove_instead_of_truncate:
            initial_pairs = len(self.token)

            self.tokens = list(filter(lambda pair: len(pair) <= max_length, self.tokens))

            after_remove_pairs = len(self.token)

            logger.debug("%d pairs of URLs have been removed: from %d to %d pairs", initial_pairs - after_remove_pairs, initial_pairs, after_remove_pairs)
        else:
            needs_truncation = sum([1 if len(pair) > max_length else 0 for pair in self.tokens])

            logger.debug("%d pairs of URLs need truncation of %d pairs", needs_truncation, len(self.tokens))

            self.tokens = [pair[:max_length] for pair in self.tokens]

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

    def get_dataloader(self, batch_size, device, force_cpu, num_workers, sampler=None, max_tokens=None, set_dataloader=True):
        is_device_gpu = device.type.startswith("cuda")

        if sampler:
            self.sampler = sampler
        elif self.sampler_better_randomness:
            # LengthGroupedSampler handles worse the padding problem (suboptimal) but better the randomness than SmartBatchingSampler
            lengths = [len(seq) for seq in self.tokens]
            self.sampler = transformers.trainer_pt_utils.LengthGroupedSampler(batch_size, lengths=lengths)
        else:
            self.sampler = SmartBatchingSampler(
                data_source=self.tokens,
                batch_size=batch_size,
            )

        if max_tokens:
            logger.info("Batch size will be data-dependant: batches of, approximately, %d tokens will be returned",
                        max_tokens)

            collate_fn = MaxTokensCollate(
                pad_token_id=self.pad_token_id,
                max_tokens=max_tokens,
                total_number_of_batches=len(self.tokens),
            )
        else:
            collate_fn = SmartBatchingCollate(
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
            batch_size=None if max_tokens else batch_size, # https://pytorch.org/docs/stable/data.html#disable-automatic-batching
            sampler=self.sampler,
            collate_fn=collate_fn,
            **dataloader_kwargs,
        )

        if set_dataloader:
            if self.dataloader:
                logger.warning("Be aware that the dataloader has been updated")

            self.dataloader = dataloader

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
        _batches = self.batches

        if _batches:
            last_batch = _batches.pop(-1) # Remove last element before randomizing since its length might be less than the batch size

            np.random.shuffle(_batches) # Randomize batches
            _batches.append(last_batch) # Add the previously removed last element

        self._inds = list(more_itertools.flatten(_batches))

        yield from self._inds # Return index of the randomized batches flattened but sorted by tokens length

    def __len__(self):
        return self.len

    @property
    def backsort_inds(self):
        if self._backsort_inds is None:
            self._backsort_inds = np.argsort(self._inds)

        return self._backsort_inds

def pad_sequence(sequence_batch, pad_token_id, max_length=0):
    max_batch_len = max(len(sequence) for sequence in sequence_batch)
    max_len = min(max_batch_len, max_length) if max_length > 0 else max_batch_len
    padded_sequences, attention_masks = [], []
    attend, no_attend = 1, 0

    for sequence in sequence_batch:
        # Truncate if exceeds max_len
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

class SmartBatchingCollate:
    def __init__(self, pad_token_id):
        self._pad_token_id = pad_token_id

    def __call__(self, batch):
        sequences, targets = list(zip(*batch))
        input_ids, attention_mask = pad_sequence(sequences, self._pad_token_id)

        output = {
            "url_tokens": input_ids,
            "url_attention_mask": attention_mask,
        }
        output["label"] = torch.tensor(targets)

        return output

class MaxTokensCollate:
    # Issues related:
    #  https://github.com/microsoft/DeepSpeed/issues/1051
    #  Mentioning --max_tokens from fairseq: https://github.com/huggingface/transformers/issues/10512

    def __init__(self, pad_token_id, max_tokens, total_number_of_batches):
        self._pad_token_id = pad_token_id
        self._max_tokens = max_tokens
        self._total_number_of_batches = total_number_of_batches

        self.reset_max_tokens_variables(last_or_first_batch=True)

    def reset_max_tokens_variables(self, last_or_first_batch=False):
        # Max tokens variables
        self._current_tokens = 0
        self._current_batch = []
        self._current_max_length = 0

        if last_or_first_batch:
            self._current_number_batch = 0
            self._aux_batch = [] # Auxiliar storage (we want to avoid to exceed max_tokens)

    def __call__(self, batch):
        sequence, target = batch

        if len(self._aux_batch) > 0:
            self._current_batch.extend(self._aux_batch)
            self._aux_batch = []
            self._current_max_length = max(self._current_max_length, max([len(s) for s, _ in self._current_batch]))

        self._current_max_length = max(self._current_max_length, len(sequence)) # Necessary for padding
        self._current_tokens = self._current_max_length * (len(self._current_batch) + 1) # Simulate padding with the current longest sentence
        self._current_number_batch += 1
        equal_max_tokens_processed = self._current_tokens == self._max_tokens
        more_max_tokens_processed = self._current_tokens > self._max_tokens
        max_tokens_processed = equal_max_tokens_processed or more_max_tokens_processed
        last_batch = self._current_number_batch >= self._total_number_of_batches
        force_return = False

        if more_max_tokens_processed and not last_batch:
            self._aux_batch.append([sequence, target])

            force_return = True
        else:
            self._current_batch.append([sequence, target])

        if force_return or max_tokens_processed or last_batch:
            # Return dynamic batch when max_tokens criteria is met or last batch is being processed
            sequences, targets = list(zip(*self._current_batch))
            input_ids, attention_mask = pad_sequence(sequences, self._pad_token_id)

            output = {
                "url_tokens": input_ids,
                "url_attention_mask": attention_mask,
            }
            output["label"] = torch.tensor(targets)

            # Reset variables
            self.reset_max_tokens_variables(last_or_first_batch=last_batch)

            # Return batch
            return output
        else:
            # Keep accumulating partial batches
            return None
