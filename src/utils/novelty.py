import numpy as np
import pandas as pd
import seaborn as sns
from time import time
import matplotlib.pyplot as plt


def comparison_novelties(ref_corpus, corpuses, models_names, motifs=(2, 4, 8, 16, 32)):
    start = time()
    sets = compute_sets(ref_corpus, motifs)
    novelties = pd.DataFrame()
    print("Build hash set in {}s".format(time()-start))

    for i, corpus in enumerate(corpuses):
        start = time()
        array_novelties = compare_to_sets(corpus, motifs, sets)
        print("Comparison {} in {}s".format(models_names[i], time() - start))
        df = pd.DataFrame({'novelty': array_novelties[:, 0], 'motif-size': array_novelties[:, 1],
                           'model': models_names[i]})
        novelties = pd.concat([novelties, df])

    plt.subplots(figsize= (10,7))
    sns.boxplot(data=novelties, x="motif-size", y="novelty", hue="model")


def novelty(corpus1, corpus2, motifs=(2, 4, 8, 16, 32)):
    sets = compute_sets(corpus1, motifs)
    return compare_to_sets(corpus2, motifs, sets)


def compute_sets(corpus, motifs):
    sets = {mot: set() for mot in motifs}
    whole_corpus = isinstance(corpus, dict)
    num_songs = len(corpus["dTseqs"]) if whole_corpus else len(corpus)
    for s in range(num_songs):
        n = len(corpus["dTseqs"][s]) if whole_corpus else len(corpus[s])
        for i in range(n):
            for mot in motifs:
                if i+mot <= n:
                    if whole_corpus:
                        nextMotif = str([(corpus["dTseqs"][s][k], corpus["tseqs"][s][k], corpus["pitchseqs"][s][k])
                                         for k in range(i, i+mot)])
                    else:
                        nextMotif = str([(corpus[s][k], corpus[s][k], corpus[s][k]) for k in range(i, i + mot)])
                    sets[mot].add(nextMotif)

    return sets


def compare_to_sets(corpus, motifs, sets):
    whole_corpus = isinstance(corpus, dict)
    num_songs = len(corpus["dTseqs"]) if whole_corpus else len(corpus)

    values = []
    for s in range(num_songs):
        counters = {mot: 0 for mot in motifs}
        n = len(corpus["dTseqs"][s]) if whole_corpus else len(corpus[s])
        for i in range(n):
            for mot in motifs:
                if i+mot <= n:
                    if whole_corpus:
                        nextMotif = str([(corpus["dTseqs"][s][k], corpus["tseqs"][s][k], corpus["pitchseqs"][s][k])
                                         for k in range(i, i+mot)])
                    else:
                        nextMotif = str([(corpus[s][k], corpus[s][k], corpus[s][k]) for k in range(i, i + mot)])
                    if nextMotif in sets[mot]:
                        counters[mot] += 1

        # divide number of occurrences by number of sequences
        for mot in motifs:
            values.append([1. - counters[mot] / float(n - mot + 1), mot])

    return np.array(values)
