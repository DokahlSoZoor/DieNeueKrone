# process data
def build_vocab(text):
    # filter non-ascii characters
    text = text.encode("ascii", errors="ignore").decode()
    # build vocabulary
    count = collections.Counter(text).most_common()
    vocab = dict()
    for character, _ in count:
        vocab[character] = len(vocab)
    reverse_vocab = dict(zip(vocab.values(), vocab.keys()))
    return vocab, reverse_vocab

# text = "Hello, my name is Evelyn. Nice to meet you!"
with open('data/all.txt', 'r') as f:
    text = f.read()

vocab, reverse_vocab = build_vocab(text)
print(vocab)
print(reverse_vocab)
