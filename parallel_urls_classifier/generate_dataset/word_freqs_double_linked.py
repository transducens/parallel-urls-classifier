
# Original file from https://github.com/bitextor/bicleaner-ai/blob/master/bicleaner_ai/word_freqs_zipf_double_linked.py

import os
import sys

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
        WordZipfFreqDist.__init__(self,file_with_freq)

        self.freq_words = dict()

        for w, f in self.word_freqs.items():
            if f not in self.freq_words:
                self.freq_words[f] = set()

            self.freq_words[f].add(w)

    def get_words_for_freq(self, freq):
        if freq in self.freq_words:
            return self.freq_words[freq]
        else:
            return None
