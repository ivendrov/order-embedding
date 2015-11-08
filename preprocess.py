"""
Preprocesses captions
Input: a file with lines of tokens separated by whitespace
Output: the same file, with non-alphanumeric tokens filtered out, all instances of 'a' and 'the' removed,
    converted to lower case, and lemmatized.
"""
import fileinput
import nltk
from nltk.corpus import wordnet
import re

lemma_cache = {}

def lemmatize(token):
    if token not in lemma_cache:
        # pick minimum length lemma among all possible POS tags of the word
        options = wordnet._morphy(token, 'n') + wordnet._morphy(token, 'v') + wordnet._morphy(token, 'a') + [token]
        lemma_cache[token] = min(options, key=len)

    #print(token, lemma_cache[token])
    return lemma_cache[token]




for line in fileinput.input():
    tokens = re.split('\s|[-]', line.strip())
    # filter out non-alphanumeric tokens, and convert to lowercase
    tokens = [t.lower() for t in tokens if t.isalnum() and t not in ['a', 'the']]
    # lemmatize
    lemmas = map(lemmatize, tokens)
    print(' '.join(lemmas))



