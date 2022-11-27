
import os
import sys
import logging

from nltk.tokenize import RegexpTokenizer

_tokenizer_regex = r'[^\W_0-9]+|[^\w\s]+|_+|\s+|[0-9]+' # Similar to wordpunct_tokenize
_tokenize = RegexpTokenizer(_tokenizer_regex).tokenize
_tokenize_gaps = RegexpTokenizer(_tokenizer_regex, gaps=True).tokenize

def tokenize(s, check_gaps=True, gaps_whitelist=[' ']):
    tokenized_str = _tokenize(s)

    if check_gaps:
        tokenized_str_gaps = _tokenize_gaps(s)

        if len(tokenized_str_gaps) != 0:
            for gap in tokenized_str_gaps:
                if gap not in gaps_whitelist:
                    logging.error("Found gaps tokenizing, but the tokenizer should be complete (bug?): %s", s)
                    logging.error("Gaps: %d: %s", len(tokenized_str_gaps), str(tokenized_str_gaps))

                    break

    return tokenized_str

if __name__ == "__main__":
    for s in sys.stdin.buffer:
        s = s.decode(errors="backslashreplace")

        print(' '.join(tokenize(s.rstrip('\n'), check_gaps=False if ("PUC_CHECK_GAPS" in os.environ and os.environ["PUC_CHECK_GAPS"] in ("False", "false")) else True)))
