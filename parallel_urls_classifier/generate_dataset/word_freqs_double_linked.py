
# Original file from https://github.com/bitextor/bicleaner-ai/blob/master/bicleaner_ai/word_freqs_zipf_double_linked.py

import os
import sys
import math

cdir = os.path.dirname(os.path.realpath(__file__))

sys.path.insert(0, f"{cdir}/../..")

from parallel_urls_classifier.generate_dataset.word_freqs import WordFreqDist

# Class to store word frequences. Word frequences are read from a tab-sepparated file containing two fields: freqences
# first and words second. Words must be lowercased. Such files can be easyly produced from
# monolingual text running a command like this:
# cat monolingual.txt | tokenizer.sh | tr ' ' '\n' | tr '[:upper:]' '[:lower:]' | sort | uniq -c > wordfreq.txt
class WordFreqDistDoubleLinked(WordFreqDist):

    # Constructor
    def __init__(self, file_with_freq):
        WordFreqDist.__init__(self, file_with_freq)

        self.freq_words = dict()
        self.occs_words = dict()
        self.max_occs = -math.inf
        self.min_occs = math.inf

        for w, f in self.word_freqs.items():
            if f not in self.freq_words:
                self.freq_words[f] = set()

            self.freq_words[f].add(w)

        for w, o in self.word_occs.items():
            if f not in self.occs_words:
                self.occs_words[f] = set()

            self.occs_words[f].add(w)

            if o > self.max_occs:
                self.max_occs = o
            if o < self.min_occs:
                self.min_occs = o

        if self.max_occs == -math.inf:
            raise Exception("Provided file empty?")
        if self.min_occs == math.inf:
            raise Exception("Something weird happened...")

    def get_words_for_freq(self, freq):
        if freq in self.freq_words:
            return self.freq_words[freq]
        else:
            return None

    def get_words_for_occs(self, occs, exact=True):
        if occs in self.occs_words:
            return self.occs_words[occs]
        elif not exact and occs > 0:
            limit = int(math.log(occs, 2) + occs / 100) # small limit with small values (1 <= x <= 100) and
                                                        #  more flexible with high values (x > 100)
            count = 1

            while count <= limit:
                non_exact_occs = occs + count

                # Taking care of boundaries
                if count > 0:
                    non_exact_occs = min(non_exact_occs, self.max_occs)

                    # Update count: change to negative
                    count *= -1
                else:
                    non_exact_occs = max(non_exact_occs, self.min_occs)

                    # Update count: change to positive and update
                    count *= -1
                    count += 1

                # Hit?
                if non_exact_occs in self.occs_words:
                    return self.occs_words[non_exact_occs]

        return None
