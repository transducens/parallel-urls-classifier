
# Original file from https://github.com/bitextor/bicleaner-ai/blob/master/bicleaner_ai/word_freqs_zipf.py

import math

import parallel_urls_classifier.utils.utils as utils

# Class to store word freqences. Word frequences are read from a tab-sepparated file containing two fields: freqences
# first and words second. Words must be lowercased. Such files can be easyly produced from
# monolingual text running a command like this:
# cat monolingual.txt | tokenizer.sh | tr ' ' '\n' | tr '[:upper:]' '[:lower:]' | sort | uniq -c > wordfreq.txt
class WordFreqDist(object):

    # Constructor
    def __init__(self, file_with_freq):
        self.word_freqs = dict()
        self.word_occs = dict()
        fname = file_with_freq if not hasattr(file_with_freq, 'name') else file_with_freq.name

        with utils.open_xz_or_gzip_or_plain(fname) as reader:
            for line in reader:
                line = line.strip()
                parts = line.split()
                word = parts[-1]
                occs = int(parts[0])
                self.word_occs[word] = occs

        self.total_words = sum(self.word_occs.values())

        for word, occs in self.word_occs.items():
            self.word_freqs[word] = int(math.log(float(occs)/float(self.total_words))*100)

        self.min_freq = int(math.log(1.0/float(self.total_words))*100)
        max_val = max(self.word_freqs.values())
        min_max_diff = abs(max_val)-abs(self.min_freq)
        self.q1limit = self.min_freq-min_max_diff
        self.q2limit = self.min_freq-(2*min_max_diff)
        self.q3limit = self.min_freq-(3*min_max_diff)

    def split_sentence_by_freq(self, sentence):
        word_splits = dict()

        for i in range(0, 4):
            word_splits[i] = set()

        for w in sentence:
            word_splits[self.get_word_quartile(w)-1].add(w)

        return word_splits

    def get_word_quartile(self, word):
        if word in self.word_freqs:
            val_word = self.word_freqs[word]

            if val_word <= self.q1limit:
                return 1
            elif val_word <= self.q2limit:
                return 2
            elif val_word <= self.q3limit:
                return 3
            else:
                return 4
        else:
            return 4

    def word_is_in_q1(self, word):
        if word in self.word_freqs:
            val_word = self.word_freqs[word]

            return val_word <= self.q1limit
        else:
            return False

    def word_is_in_q2(self, word):
        if word in self.word_freqs:
            val_word = self.word_freqs[word]

            return val_word <= self.q2limit
        else:
            return False

    def word_is_in_q3(self, word):
        if word in self.word_freqs:
            val_word = self.word_freqs[word]

            return val_word <= self.q3limit
        else:
            return False

    def word_is_in_q4(self, word):
        if word in self.word_freqs:
            val_word = self.word_freqs[word]

            return val_word > self.q3limit
        else:
            return True

    def get_word_freq(self, word):
        word = word.lower()

        if word in self.word_freqs:
            return self.word_freqs[word]
        else:
            return self.min_freq

    def get_word_occs(self, word):
        word = word.lower()

        if word in self.word_occs:
            return self.word_occs[word]
        else:
            return 0
