from collections import defaultdict
from tools import getSong, getNote
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from heapq import heappush, heappop, heapreplace
from preprocessing import preprocess


def patterns(data, sizes=(2,4,8,16), top=10, dictionaries=None):
    P = len(data["dTseqs"])  # number of songs

    # Rythmic part
    counts_r = dict()
    # Pitch part
    counts_p = dict()
    totals = dict()
    for size in sizes:
        counts_r[size] = defaultdict(float)
        counts_p[size] = defaultdict(float)
        totals[size] = 0

    # count all occurrences of all patterns of each size
    for s in range(P):
        song = getSong(data, s)
        N = len(song["dTseqs"])
        for i in range(N):
            for size in sizes:
                if size + i <= N:
                    if dictionaries is None:
                        pattern_r = [song["tseqs"][i]] + \
                                  [(song["dTseqs"][k], song["tseqs"][k]) for k in range(i+1, i+size)]
                        pattern_p = str([song["pitchseqs"][k] for k in range(i, i+size)])
                    else:
                        pattern_r = [dictionaries["duration_text"][song["tseqs"][i]]] + \
                                    [(dictionaries["duration_text"][song["dTseqs"][k]],
                                      dictionaries["duration_text"][song["tseqs"][k]]) for k in range(i+1, i + size)]
                        pattern_p = str([dictionaries["pitch_text"][song["pitchseqs"][k]] for k in range(i, i + size)])
                    pattern_r = str(pattern_r)  # to get an immutable variable
                    counts_r[size][pattern_r] += 1
                    counts_p[size][pattern_p] += 1
        for size in sizes:
            totals[size] += max(0, N - size + 1)

    for size in sizes:
        # sort the dictionary and keep top values
        mostFrequent = sorted(counts_r[size].items(), key = lambda x: x[1], reverse=True)[:top]
        print "Most Frequent rythmic patterns for size {}: ".format(size)
        for x in mostFrequent:
            print "{} : {}  ({:.2%})".format(x[0], x[1], x[1]/totals[size])
        fig, ax = plt.subplots()
        ax.set_yscale('log')
        sns.distplot(np.array(counts_r[size].values(), dtype=float)/totals[size], kde=False, ax=ax)
        plt.savefig('hist_{}.png'.format(size))

        mostFrequent = sorted(counts_p[size].items(), key=lambda x: x[1], reverse=True)[:top]
        print "Most Frequent pitch patterns for size {}: ".format(size)
        for x in mostFrequent:
            print "{} : {}  ({:.2%})".format(x[0], x[1], x[1] / totals[size])
        fig, ax = plt.subplots()
        ax.set_yscale('log')
        sns.distplot(np.array(counts_r[size].values(), dtype=float) / totals[size], kde=False, ax=ax)
        plt.savefig('hist_{}.png'.format(size))


def autocorrelation(song, dictionaries, size=8, correlation='notes', plot=False):
    N = len(song["dTseqs"])

    corrFunc = defaultdict(float)   # maps (difference in num quarter notes) -> sum of measured correlations
    divisors = defaultdict(float)   # maps (difference in num quarter notes) -> num observations for this difference

    for i in range(N):
        if i+size < N:
            pattern = [(song["tseqs"][i], song["pitchseqs"][i])] + \
                      [getNote(song, k) for k in range(i+1, i+size)]

            tDiff = 0.
            for j in range(i, N - size + 1):
                if j != i:
                    tDiff += dictionaries["dTseqs"][song["dTseqs"][j]]
                divisors[tDiff] += 1.

                if correlation == 'perfect':
                    pattern2 = [(song["tseqs"][j], song["pitchseqs"][j])] + \
                               [getNote(song, k) for k in range(j+1, j+size)]
                    if pattern == pattern2:
                        corrFunc[tDiff] += 1.

                if correlation == 'notes':
                    val = 0
                    curTi = 0
                    curTj = 0
                    ki = 0
                    kj = j
                    # for all notes in the pattern, look for appearance on the shifted score
                    while ki < size and kj < N:

                        # for given note #ki in the pattern, for all coinciding notes on the shifted score,
                        # see if they are equal
                        if curTj == curTi:
                            kj2 = kj
                            curTj2 = curTj
                            # loop over coinciding notes on the shifted score
                            while curTj2 == curTi and kj2 < N:
                                if ki == 0:
                                    if pattern[ki] == (song["tseqs"][kj2], song["pitchseqs"][kj2]):
                                        val += 1./size
                                        break
                                    else:
                                        kj2 += 1
                                        if kj2 < N:
                                            curTj2 += dictionaries['dTseqs'][song["dTseqs"][kj2]]
                                else:
                                    if (pattern[ki][1], pattern[ki][2]) == (song["tseqs"][kj2], song["pitchseqs"][kj2]):
                                        val += 1./size
                                        break
                                    else:
                                        kj2 += 1
                                        if kj2 < N:
                                            curTj2 += dictionaries['dTseqs'][song["dTseqs"][kj2]]
                            ki += 1
                            if ki < size:
                                curTi += dictionaries['dTseqs'][pattern[ki][0]]

                        if curTj < curTi:
                            kj += 1
                            if kj < N:
                                curTj += dictionaries['dTseqs'][song["dTseqs"][kj]]

                        if curTj > curTi:
                            ki += 1
                            if ki < size:
                                curTi += dictionaries['dTseqs'][pattern[ki][0]]
                    if val > 0.:
                        corrFunc[tDiff] += val

    corrs = []
    ts = []
    divs = []
    for k in corrFunc:
        corrFunc[k] /= divisors[k]
        ts.append(k)
        divs.append(divisors[k])
        corrs.append(corrFunc[k])

    if plot:
        plt.scatter(ts, corrs, s=divs)
        plt.axvline(16, alpha=0.2, c='k')
        plt.axvline(32, alpha=0.2, c='k')
        plt.axvline(64, alpha=0.2, c='k')
        plt.axvline(128, alpha=0.2, c='k')
        plt.savefig("correlation.png")
        plt.clf()

    return corrFunc, divisors


def autocorrelationDataset(data, dictionaries, size=8, correlation='notes', alpha_mapping=False,
                           plot_fn='correlation.png', threshold=None, maxNumPoints=None,
                           maxT=None):
    """
    Computes the autocorrelation function for all songs of a dataset and makes a plot with means and standard deviations
    :param data:
    :param dictionaries:
    :param size: the pattern size used for measures of autocorrelation
    :param correlation: type of correlation measure in 'perfect' or 'notes'
    :param alpha_mapping: whether to map alpha values in the plot as function of the number of observation for each
                        correlation value
    :param plot_fn: filename of plot
    :param threshold:  min value of correlation to use, None if not relevant
    :return:
    """
    nSongs = len(data["dTseqs"])
    correlationDistr = defaultdict(list)
    observationsCounts = defaultdict(int)

    for i in tqdm(range(nSongs)):
        song = getSong(data, i)
        corrFunc, divisors = autocorrelation(song, dictionaries, size, correlation)
        for k in corrFunc:
            correlationDistr[k].append(corrFunc[k])
            observationsCounts[k] += divisors[k]

    # Prepare for plotting
    ts = []
    corrMeans = []
    corrStds = []
    maxObs = 0
    minObs = observationsCounts[0]
    # Get values with most observations if max number of points specified
    if maxNumPoints is not None:

        print "maxSize: {}".format(maxNumPoints)
        heap = []
        for k in observationsCounts:  # use a heap to get top x% values
            obs = observationsCounts[k]
            tup = (obs, k)
            if len(heap) < maxNumPoints:
                heappush(heap, tup)  # add elements to the heap until it is full
            if len(heap) >= maxNumPoints and tup > heap[0]:
                heapreplace(heap, tup)  # once the heap is full, replace with bigger elements
        for tup in heap:
            k = tup[1]
            mean = np.mean(correlationDistr[k])
            if (threshold is None or mean > threshold) and (maxT is None or k <= maxT):
                ts.append(k)
                corrMeans.append(mean)
                corrStds.append(np.std(correlationDistr[k]))
                if k > 0:
                    maxObs = max(maxObs, observationsCounts[k])
                    minObs = min(minObs, observationsCounts[k])
    else:
        for k in correlationDistr:
            mean = np.mean(correlationDistr[k])
            if (threshold is None or mean > threshold) and (maxT is None or k <= maxT):
                ts.append(k)
                corrMeans.append(mean)
                corrStds.append(np.std(correlationDistr[k]))
                if k > 0:
                    maxObs = max(maxObs, observationsCounts[k])
        minObs = np.min(observationsCounts.values())

    print "minObs: {} ".format(minObs)

    # Plotting
    if alpha_mapping:
        for k in tqdm(range(len(ts))):
            plt.errorbar(ts[k], corrMeans[k], corrStds[k], alpha=min(1., observationsCounts[ts[k]]/maxObs),
                         c='C0', linestyle='None', marker='o')
    else:
        plt.errorbar(ts, corrMeans, corrStds, linestyle='None', marker='o')
    for i in range(0, int(np.max(ts)+1), 8):
        plt.axvline(i, c='grey', alpha=0.3)
    plt.savefig(plot_fn)
    plt.clf()

    # Print top 10 mean values
    top10 = np.argsort(corrMeans)[-1:-11:-1]
    for t in top10:
        print " * t={}, mean correlation {:.2%}".format(ts[t], corrMeans[t])

    return ts, corrMeans, corrStds, observationsCounts



def __test__():
    data = {"dTseqs": [[0,1,0,1,0,1]],
            "tseqs": [[0,1,0,1,0,1]],
            "pitchseqs": [[0,1,0,1,0,1]]}
    patterns(data, sizes=(2,), top=2)


def plot_all():
    data, sizes, dictionaries, labels = preprocess('../../corpora/Original')
    autocorrelationDataset(data, dictionaries, alpha_mapping=True, plot_fn='correlation-original.png',
                           maxNumPoints=100, maxT=64, special='orig')
    data, sizes, dictionaries, labels = preprocess('../../corpora/BachProp')
    autocorrelationDataset(data, dictionaries, alpha_mapping=True, plot_fn='correlation-BP.png',
                           maxNumPoints=100, maxT=64, special='bp')
    data, sizes, dictionaries, labels = preprocess('../../corpora/DeepBach')
    autocorrelationDataset(data, dictionaries, alpha_mapping=True, plot_fn='correlation-DeepBach.png',
                           maxNumPoints=100, maxT=64)
    data, sizes, dictionaries, labels = preprocess('../../corpora/IndepBP')
    autocorrelationDataset(data, dictionaries, alpha_mapping=True, plot_fn='correlation-IBP.png',
                           maxNumPoints=100, maxT=64)
    data, sizes, dictionaries, labels = preprocess('../../corpora/MidiBP')
    autocorrelationDataset(data, dictionaries, alpha_mapping=True, plot_fn='correlation-MBP.png',
                           maxNumPoints=100, maxT=64)
    data, sizes, dictionaries, labels = preprocess('../../corpora/MLP')
    autocorrelationDataset(data, dictionaries, alpha_mapping=True, plot_fn='correlation-MLP.png',
                           maxNumPoints=100, maxT=64)
    data, sizes, dictionaries, labels = preprocess('../../corpora/PolyDAC')
    autocorrelationDataset(data, dictionaries, alpha_mapping=True, plot_fn='correlation-PolyDAC.png',
                           maxNumPoints=100, maxT=64)
    data, sizes, dictionaries, labels = preprocess('../../corpora/PolyRNN')
    autocorrelationDataset(data, dictionaries, alpha_mapping=True, plot_fn='correlation-PolyRNN.png',
                           maxNumPoints=100, maxT=64)
