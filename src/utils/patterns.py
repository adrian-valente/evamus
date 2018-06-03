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


def autocorrelation(song, dictionaries, plot=False, maxT=None):
    N = len(song["dTseqs"])

    corrFunc = defaultdict(float)   # maps (difference in num quarter notes) -> sum of measured correlations
    obsCount = defaultdict(float)

    for i in range(N):
        note = (song["tseqs"][i], song["pitchseqs"][i])
        tDiff = 0.
        for j in range(i, N):
            if j != i:
                tDiff += dictionaries["dTseqs"][song["dTseqs"][j]]
            if tDiff > maxT:
                break
            if note == (song["tseqs"][j], song["pitchseqs"][j]):
                corrFunc[tDiff] += 1
            obsCount[tDiff] += 1

    corrs = []
    ts = []
    for k in corrFunc:
        corrFunc[k] /= len(song["dTseqs"])
        ts.append(k)
        corrs.append(corrFunc[k])

    if plot:
        plt.scatter(ts, corrs)
        plt.axvline(16, alpha=0.2, c='k')
        plt.axvline(32, alpha=0.2, c='k')
        plt.axvline(64, alpha=0.2, c='k')
        plt.axvline(128, alpha=0.2, c='k')
        plt.savefig("correlation.png")
        plt.clf()

    return corrFunc, obsCount


def autocorrelationDataset(data, dictionaries, plot_fn='correlation.png', alpha_mapping=False, maxNumPoints=None,
                           threshold=None, maxT=None):
    """
    Computes the autocorrelation function for all songs of a dataset and makes a plot with means and standard deviations
    :param data:
    :param dictionaries:
    :param alpha_mapping: whether to map alpha values in the plot as function of the number of observation for each
                        correlation value
    :param plot_fn: filename of plot
    :param threshold:  min value of correlation to use, None if not relevant
    :return:
    """
    nSongs = len(data["dTseqs"])
    correlationDistr = defaultdict(list)
    observationsCounts = defaultdict(float)

    for i in tqdm(range(nSongs)):
        song = getSong(data, i)
        corrFunc, songCounts = autocorrelation(song, dictionaries, maxT=maxT)
        for k in corrFunc:
            correlationDistr[k].append(corrFunc[k])
            observationsCounts[k] += songCounts[k]

    # Prepare for plotting
    ts = []
    corrMeans = []
    corrStds = []
    maxObs = 0.
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
    # Otherwise compute mean and standard deviation for all points
    else:
        for k in correlationDistr:
            mean = np.mean(correlationDistr[k])
            if threshold is None or mean > threshold:
                ts.append(k)
                corrMeans.append(mean)
                corrStds.append(np.std(correlationDistr[k]))
            if k != 0 and observationsCounts[k] > maxObs:
                maxObs = observationsCounts[k]

    # Plotting
    if alpha_mapping:
        for k in tqdm(range(len(ts))):
            plt.errorbar(ts[k], corrMeans[k], corrStds[k], alpha=min(1., observationsCounts[ts[k]] / maxObs),
                         c='C0', linestyle='None', marker='o')
    else:
        plt.errorbar(ts, corrMeans, corrStds, linestyle='None', marker='o')
    for i in range(0, int(np.max(ts)+1), 4):
        plt.axvline(i, c='grey', alpha=0.3)
    plt.xlabel("$\delta T$ (beats)")
    plt.ylabel("autocorrelation")
    plt.savefig(plot_fn)
    plt.clf()

    # Print top 10 mean values
    top10 = np.argsort(corrMeans)[-1:-11:-1]
    for t in top10:
        print " * t={}, mean correlation {:.2%}".format(ts[t], corrMeans[t])

    return ts, corrMeans, corrStds



def __test__():
    data = {"dTseqs": [[0,1,0,1,0,1]],
            "tseqs": [[0,1,0,1,0,1]],
            "pitchseqs": [[0,1,0,1,0,1]]}
    patterns(data, sizes=(2,), top=2)


def plot_all():
    data, sizes, dictionaries, labels = preprocess('../corpora/Original')
    autocorrelationDataset(data, dictionaries, alpha_mapping=True, plot_fn='correlation-original.png',
                           maxNumPoints=100, maxT=64)
    data, sizes, dictionaries, labels = preprocess('../corpora/BachProp')
    autocorrelationDataset(data, dictionaries, alpha_mapping=True, plot_fn='correlation-BP.png',
                           maxNumPoints=100, maxT=64)
    data, sizes, dictionaries, labels = preprocess('../corpora/DeepBach')
    autocorrelationDataset(data, dictionaries, alpha_mapping=True, plot_fn='correlation-DeepBach.png',
                           maxNumPoints=100, maxT=64)
    data, sizes, dictionaries, labels = preprocess('../corpora/IndepBP')
    autocorrelationDataset(data, dictionaries, alpha_mapping=True, plot_fn='correlation-IBP.png',
                           maxNumPoints=100, maxT=64)
    data, sizes, dictionaries, labels = preprocess('../corpora/MidiBP')
    autocorrelationDataset(data, dictionaries, alpha_mapping=True, plot_fn='correlation-MBP.png',
                           maxNumPoints=100, maxT=64)
    data, sizes, dictionaries, labels = preprocess('../corpora/MLP')
    autocorrelationDataset(data, dictionaries, alpha_mapping=True, plot_fn='correlation-MLP.png',
                           maxNumPoints=100, maxT=64)
    data, sizes, dictionaries, labels = preprocess('../corpora/PolyDAC')
    autocorrelationDataset(data, dictionaries, alpha_mapping=True, plot_fn='correlation-PolyDAC.png',
                           maxNumPoints=100, maxT=64)
    data, sizes, dictionaries, labels = preprocess('../corpora/PolyRNN')
    autocorrelationDataset(data, dictionaries, alpha_mapping=True, plot_fn='correlation-PolyRNN.png',
                           maxNumPoints=100, maxT=64)
    # data, sizes, dictionaries, labels = preprocess('../corpora/Nottingham/BachProp/')
    # autocorrelationDataset(data, dictionaries, alpha_mapping=True, plot_fn='correlation-N-BP.png',
    #                        maxNumPoints=100, maxT=64)
    # data, sizes, dictionaries, labels = preprocess('../corpora/Nottingham/Original/')
    # autocorrelationDataset(data, dictionaries, alpha_mapping=True, plot_fn='correlation-N-Original.png',
    #                        maxNumPoints=100, maxT=64)
    # data, sizes, dictionaries, labels = preprocess('../corpora/Nottingham/FUNgram/')
    # autocorrelationDataset(data, dictionaries, alpha_mapping=True, plot_fn='correlation-N-FUNgram.png',
    #                        maxNumPoints=100, maxT=64)