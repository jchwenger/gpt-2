"""Byte pair encoding utilities"""

import os
import json
import regex as re
from functools import lru_cache
    # Functools: module for higher order functions (functions calling other functions)
    # lru_cache: (Python docs) Decorator to wrap a function with a memorizing callable
    # that saves up to the maxsize most recent calls. It can save time when an expensive
    # or I/O bound function is periodically called with the same arguments.

@lru_cache()
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe (Byte Pair Encoding) codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    (N.B.: <UNK> is used in many datasets as a placeholder for 'unknown' (e.g. words).)
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a signficant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """

    # ord: returns integer corresponding to Unicode character
    bs = list(range(ord("!"), ord("~")+1))+list(range(ord("¡"), ord("¬")+1))+list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:] # create a deep copy

    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8+n)
            n += 1

    # chr: returns string corresponding to Unicode integer code
    # (of such and such character)
    # replace integer codes by their characters
    cs = [chr(n) for n in cs]

    # return the dict { 33: '!', 34: '"', ... }
    return dict(zip(bs, cs))

def get_pairs(word):
    """
    Return set of symbol pairs in a word.
    Word argument is given as a tuple of symbols (symbols being variable-length strings).

    Thus, the word 'word' is represented as;
    {('o', 'r'), ('r', 'd'), ('w', 'o')}
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs

class Encoder:
                                            # errors='replace'
                                            # an option for the bytearray() conversion function used below.
                                            # cf Python doc: Replace with a suitable replacement marker;
                                            # Python will use the official U+FFFD REPLACEMENT CHARACTER
                                            # for the built-in codecs on decoding, and ‘?’ on encoding.
    def __init__(self, encoder, bpe_merges, errors='replace'):
        self.encoder = encoder
        self.decoder = {v:k for k,v in self.encoder.items()}            # simply reversing from {k:v} to {v:k}
        self.errors = errors # how to handle errors in decoding
        self.byte_encoder = bytes_to_unicode()                          # our look-up table function
        self.byte_decoder = {v:k for k, v in self.byte_encoder.items()} # reversing again, for bytes
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))  # { x0: 0, x1: 1, ...}
        self.cache = {}

        # Should haved added re.IGNORECASE so BPE merges can happen for capitalized versions of contractions
                            # Standard matches:
                            # - contractions ('s, 't, 're, 've, 'm, 'll, 'd)
                            # The next three preceded by optional space:
                            # - words: one or more of any letter (\p{L})
                            # - numbers: one or more of any (\p{N})
                            # - punctuation: neither space, letter or number, one or more times
                            #   [^\s\p{L}\p{N}]
                            # Blank spaces:
                            # - no single space: one or more spaces not followed non-whitespace (letter, etc.)
                            #   (note the lovely negative lookahead: (?!\S))
                            # - one or more spaces ok
        self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""", flags=re.IGNORECASE)
                                                                                                                # adding ignorecase, as mentioned above

    # Replace the most common byte pairs by a single one to compress the
    # message, can be done recursively, cf. here https://en.wikipedia.org/wiki/Byte_pair_encoding
    def bpe(self, token):

        # don't do the work twice, save words on the go
        if token in self.cache:
            return self.cache[token]

        word = tuple(token)     # turn word to char tuple
        pairs = get_pairs(word) # get all char pairs: ('w','o','r','d') > { ('w', 'o'), ('o', 'r'), ('r', 'd') }

        # if word was only one symbol?
        if not pairs:
            return token

        while True:

            bigram = min(pairs, key = lambda pair: self.bpe_ranks.get(pair, float('inf')))
                                                                            # float('inf'): It acts as an unbounded upper value for
            if bigram not in self.bpe_ranks:                                # comparison. This is useful for finding lowest
                break                                                       # values for something.
                                                                            # https://stackoverflow.com/a/34264749
            first, second = bigram
            new_word = []

            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)   # returns index of searched element (first), starting at i
                    new_word.extend(word[i:j]) # append items from iterable to the end of the array
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word)-1 and word[i+1] == second:
                    new_word.append(first+second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = ' '.join(word)
        self.cache[token] = word
        return word

    def encode(self, text):

        bpe_tokens = []

        # for each token found by our regex (words, numbers, more than one space, punctuation)
        for token in re.findall(self.pat, text):

            # encode to utf-8 (char > int), then encode to byte, then join in a string
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))

            bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' '))

        return bpe_tokens

    def decode(self, tokens):
        text = ''.join([self.decoder[token] for token in tokens])
        text = bytearray([self.byte_decoder[c] for c in text]).decode('utf-8', errors=self.errors)
        return text

def get_encoder(model_name):

    # get the vocabulary as a json file
    with open(os.path.join('models', model_name, 'encoder.json'), 'r') as f:
        encoder = json.load(f)

    # get the complete vocabulary as txt file
    with open(os.path.join('models', model_name, 'vocab.bpe'), 'r', encoding="utf-8") as f:
        bpe_data = f.read()

    # translates a string format with x y on each line to [(x,y),...]
    bpe_merges = [tuple(merge_str.split()) for merge_str in bpe_data.split('\n')[1:-1]]
                                                                                # skip the first line
    return Encoder(
        encoder=encoder,
        bpe_merges=bpe_merges,
    )
