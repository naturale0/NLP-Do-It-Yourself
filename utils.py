from nltk.corpus import gutenberg
from collections import Counter
from sklearn.manifold import TSNE
import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics.pairwise import cosine_distances


def get_gutenberg_tokens():
    """
    prepare gutenberg text from nltk, download if necessary, 
    and return it as a list of tokens(words)
    """
    try:
        words = list(map(lambda w: w.lower(), gutenberg.words()))
    except:
        import nltk; nltk.download("gutenberg")
        words = list(map(lambda w: w.lower(), gutenberg.words()))
    
    return (words, *build_vocab(words))


def build_vocab(words):
    """
    input: list of words
    output: (vocabulary, vocab_size, word_to_ix)
    """
    
    vocab, freq = list(zip(*Counter(words).items()))
    vocab = ["<unk>", "<pad>"] + list(vocab)
    word_to_ix = {word: ix+2 for ix, word in enumerate(vocab)}
    word_to_ix[0] = "<unk>"
    word_to_ix[1] = "<pad>"
    return vocab, freq, word_to_ix


def plot_train(losses, accs):
    fig = plt.figure(figsize=(7,3))
    plt.subplot(121)
    plt.plot(losses)
    plt.subplot(122)
    plt.plot(accs)
    plt.tight_layout()
    return fig


def plot_embedding(words, embedding, word_to_ix, top=150):
    """
    words: raw tokens
    embedding: torch.nn.Embedding() object
    """
    counter = Counter(words)
    
    test_words = counter.most_common(top)
    test_words_raw = [w for w, _ in test_words]
    test_words = [word_to_ix[w] for w in test_words_raw]
    
    with torch.no_grad():
        embed_xy = embedding(torch.tensor(test_words)).detach().numpy()
        embed_xy = TSNE(n_components=2).fit_transform(embed_xy)
        embed_x, embed_y = list(zip(*embed_xy))
    
    fig = plt.figure(figsize=(10, 10))
    for xy, word in zip(embed_xy, test_words_raw):
        plt.annotate(word, xy, clip_on=True, fontsize=14)

    plt.title("Word Embedding")
    plt.scatter(embed_x, embed_y, alpha=.3)
    plt.axhline([0], ls=":", c="grey")
    plt.axvline([0], ls=":", c="grey")
    
    return fig
    
    
def find_similar(word, words, embedding, word_to_ix, n=5, from_total=5000):
    distance = []
    with torch.no_grad():
        y = embedding(word_to_ix[word]).numpy().reshape(1, -1)
        total = Counter(words).most_common(from_total)
        for w, _ in total:
            x = embedding(word_to_ix[w]).numpy().reshape(1, -1)
            distance.append(cosine_distances(x, y)[0][0])
    
    distance = np.array(distance)
    top_n = distance.argsort()[1:n+1]
    
    return [total[ix][0] for ix in top_n]