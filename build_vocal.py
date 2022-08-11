from cv2 import threshold
import nltk
import pickle
import argparse
from collections import Counter
import json


class Vocabulary(object):
    """Simple vocabulary wrapper."""
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)

def build_vocab(desc_path, threshold = 4):
    """Build a simple vocabulary wrapper."""
    descriptions = json.load(open(desc_path))
    counter = Counter()
    ids = descriptions.keys()  # image names
    for i, id in enumerate(ids):
        caption = descriptions[id]
        tokens = nltk.tokenize.word_tokenize(caption.lower())
        counter.update(tokens)

#        if (i+1) % 1000 == 0:
#            print("[{}/{}] Tokenized the captions.".format(i+1, len(ids)))

    # If the word frequency is less than 'threshold', then the word is discarded.
    words = [word for word, cnt in counter.items() if cnt >= threshold]

    # Create a vocab wrapper and add some special tokens.
    vocab = Vocabulary()
    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')

    # Add the words to the vocabulary.
    for i, word in enumerate(words):
        vocab.add_word(word)
    return vocab

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--desc_path", help = "path to the train figure descriptions")
   # parser.add_argument('--vocab_path', type=str, help='path for saving vocabulary wrapper')
    parser.add_argument('--threshold', type=int, default=4, help='minimum word count threshold')
    args = parser.parse_args()
    return args


#if __name__ == '__main__':
#    args = get_args()
#    desc_path = args.desc_path
#   # vocab_path = args.vocab_path
#    threshold = args.threshold
#    vocab = build_vocab(desc_path, threshold)
#    # with open(vocab_path, 'wb') as fp:
#    #     pickle.dump(vocab, fp)
#    print("Total vocabulary size: {}".format(len(vocab)))
#    #print("Saved the vocabulary wrapper to '{}'".format(vocab_path))
