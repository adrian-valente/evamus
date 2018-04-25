import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tools import getSong
from prefixtree import PrefixTree
from midiparser import writeMIDI
from preprocessing import toMIDI


def novelty_analysis(corpus1, corpus2, motifs=(2,4,8,16,32), auto=None, motif_size_lcn=32, motif_size_best=8, path="gen/", corpus2Name="no_name", dictionaries=None):
    """
    Make a quick analysis of the novelty profile of corpus2 with respect to corpus1
    :attr corpus1: a dataset-formatted corpus
    :attr corpus2: idem
    :attr motifs: (a sequence of ints) motif sizes to consider
    :attr auto: pre-computed auto novelty of corpus1
    :attr path: path where to write longest common subsequences
    :attr dictionaries: dictionaries mapping back to MIDI (for writing LCSs)
    """
    N1 = len(corpus1["dTseqs"])

    novelties = novelty(corpus1, corpus2, motifs)

    if auto is None:
        auto = autonovelty(corpus1)

    # Make a plot
    df = pd.DataFrame({'value': auto.ravel(), 'motif-size': motifs * auto.shape[0], 'model': "auto-novelty"})
    df2 = pd.DataFrame({'value': novelties.ravel(),
                        'motif-size': motifs * novelties.shape[0],
                        'model': corpus2Name})
    df = pd.concat([df, df2])

    plt.subplots(figsize=(10, 7))
    sns.violinplot(data=df, x="motif-size", y="value", hue="model")

    # Find 5 less original songs
    assert motif_size_lcn in motifs
    worse = np.argsort(novelties[:, motifs.index(motif_size_lcn)])[:5]
    print("Less original songs (at motif size {}):".format(motifs[-1]))
    for i in worse:
        print("Generated song {}, novelty {}".format(i, novelties[i,-1]))

        # Find corpus1's most similar song
        song = getSong(corpus2, i)
        lenSong = len(song["dTseqs"])
        sets = compute_sets_single_song(song, [motif_size_lcn])
        curMax = 0.
        curArgmax = 0
        # for each song of the corpus1, compute similarity with currently studied song
        for s in range(N1):
            nov = compare_to_sets_single_song(getSong(corpus1, s), [motif_size_lcn], sets)
            coincidences = (1. - nov) * (len(corpus1["dTseqs"][s]) - motif_size_lcn + 1)  #renormalization
            real_sim = coincidences / (lenSong - motif_size_lcn + 1)
            if real_sim > curMax:
                curMax = real_sim
                curArgmax = s
        print("This song is very similar to song {} of the corpus: similarity {}".format(curArgmax, curMax))

        # find longest common subsequence among the 2 similar songs
        if dictionaries is not None:
            pt = PrefixTree(getSong(corpus1, curArgmax))
            lcs = pt.longest_common_subsequence(song)
            dtseq, tseq, pseq = toMIDI([note[0] for note in lcs], [note[1] for note in lcs], [note[2] for note in lcs],
                                       dictionaries)
            writeMIDI(dtseq, tseq, pseq, path=path, label="lcs"+corpus2Name, tag=str(curArgmax)+'-'+str(i))
            print("longest common subsequence of length {} written to disk".format(len(lcs)))

    # Get 5 most original songs (here wrt motif size #2)
    assert motif_size_best in motifs
    best = np.argsort(novelties[:,motifs.index(motif_size_best)])[-5:]
    print("Most original songs at motif size {}:".format(motif_size_best))
    for i in best[::-1]:
        print("Generated song {} : novelty {}".format(i, novelties[i, 2]))


def comparison_novelties(ref_dataset, datasets, names, motifs=(2, 4, 8, 16, 32), auto=None):
    """
    Computes novelties for each of the datasets against ref_dataset, and plots a violinplot
    :param auto: optional precomputed autonovelty (with for example test vs train measure if desired)
    """
    if auto is None:
        auto = autonovelty(ref_dataset, motifs)
    dfs = dict()
    for i,dataset in enumerate(datasets):
        nov = novelty(ref_dataset, dataset, motifs)
        dfs[names[i]] = pd.DataFrame({'value': nov.ravel(), 'motif-size': motifs * nov.shape[0],
                                      'model': names[i]})
    df = pd.concat(dfs)

    plt.subplots(figsize=(10,7))
    sns.violinplot(data=df, x='motif-size', y='value', hue='model')



def autonovelty(corpus, motifs=(2, 4, 8, 16, 32)):
    """
    Compute auto-novelty profile of a corpus for given motifs sizes
    :param corpus: a dataset
    :param motifs: a sequence of motif sizes
    :return: an ndarray of shape (number_of_songs, number_of_motif_sizes) giving novelty of each song
    w.r.t. the rest of the corpus for each motif size
    """
    N = len(corpus["dTseqs"])

    # Compute for each song in corpus a hashset of motifs
    sets = []
    for s in range(N):
        sets.append(compute_sets_single_song({"dTseqs": corpus["dTseqs"][s],
                                              "tseqs": corpus["tseqs"][s],
                                              "pitchseqs": corpus["pitchseqs"][s]}, motifs))

    # for each song in corpus, compute novelty w.r.t. the rest of the corpus
    values = np.zeros((N, len(motifs)))
    for s in range(N):
        song = {"dTseqs": corpus["dTseqs"][s],
                "tseqs": corpus["tseqs"][s],
                "pitchseqs": corpus["pitchseqs"][s]}
        n = len(song["dTseqs"])

        # Count number of motifs that occur in other songs of the corpus
        for i in range(n):
            for m, mot in enumerate(motifs):
                if i + mot <= n:
                    nextMotif = str([(song["dTseqs"][k], song["tseqs"][k], song["pitchseqs"][k])
                                     for k in range(i, i + mot)])

                    # Compare to other songs of the corpus
                    k = 0
                    found = False
                    while k < N:
                        if k != s and nextMotif in sets[k][mot]:
                            found = True
                            break
                        k += 1
                    if found:
                        values[s][m] += 1

        # Normalize by the number of occurrences of motif length in sequence
        for m, mot in enumerate(motifs):
            values[s][m] = 1. - values[s][m] / float(n - mot + 1)

    return values


def novelty(corpus1, corpus2, motifs=(2, 4, 8, 16, 32)):
    """
    Compute novelty profile of corpus2 against corpus1 for a given list of motif sizes
    :param corpus1: the reference corpus (a full dataset, or only a sequence type dataset)
    :param corpus2: the compared corpus (id.)
    :param motifs: a sequence of motifs
    :return: an ndarray of shape (number_of_songs_in_corpus2, number_of_motif_sizes) giving for
    each song in corpus2 and each motif size the novelty value
    """
    sets = compute_sets(corpus1, motifs)
    return compare_to_sets(corpus2, motifs, sets)


def compute_sets(corpus, motifs):
    """
    compute hashset of motifs for corpus. Returns a dict(motif_size -> set)
    """
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
    """
    Compare each song in corpus to the precomputed hashsets sets
    Returns an ndarray of shape (number_of_songs, number_of_motif_sizes)
    """
    whole_corpus = isinstance(corpus, dict)
    num_songs = len(corpus["dTseqs"]) if whole_corpus else len(corpus)

    values = np.zeros((num_songs, len(motifs)))
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
        for m, mot in enumerate(motifs):
            values[s][m] = 1. - counters[mot] / float(n - mot + 1)

    return np.array(values)


def compute_sets_single_song(song, motifs):
    """
    Same as compute_sets for a single song instead of a corpus
    """
    sets = {mot: set() for mot in motifs}
    n = len(song["dTseqs"])
    for i in range(n):
        for mot in motifs:
            if i+mot <= n:
                nextMotif = str([(song["dTseqs"][k], song["tseqs"][k], song["pitchseqs"][k])
                                     for k in range(i, i+mot)])
                sets[mot].add(nextMotif)
    return sets


def compare_to_sets_single_song(song, motifs, sets):
    """
    Same as compare_to_sets for a single song
    """
    counters = {mot: 0 for mot in motifs}
    n = len(song["dTseqs"])
    for i in range(n):
        for mot in motifs:
            if i+mot <= n:
                nextMotif = str([(song["dTseqs"][k], song["tseqs"][k], song["pitchseqs"][k])
                                     for k in range(i, i+mot)])
                if nextMotif in sets[mot]:
                    counters[mot] += 1

    # divide number of occurrences by number of sequences
    return np.array([1. - counters[mot] / float(n - mot + 1) for mot in motifs])