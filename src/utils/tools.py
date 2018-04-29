from collections import defaultdict
from random import random


def getSong(dataset, i):
    """
    return song i of the dataset
    """
    return {"dTseqs": dataset["dTseqs"][i],
            "tseqs": dataset["tseqs"][i],
            "pitchseqs": dataset["pitchseqs"][i]}


def normalize(d):
    """
    Returns a normalized version of the dictionary d
    :param d: a dictionary mapping to POSITIVE numbers
    :return: the normalized dictionary
    """
    Z = 0
    res = defaultdict(float)
    for key in d:
        Z += d[key]
    for key in d:
        res[key] = d[key]/float(Z)
    return res

def mydefaultdict():
    return defaultdict(float)

def trainsingleorder(data, order):
    """
    Returns a trained dictionary on data at given order for a single sequence
    :param data: a sequence
    :param order: an int (0 for frequency count, 1 for Markov...)
    :return: if order is 0 a dict {value -> prob of occurrence}
             for bigger orders a dict {str(history) -> {value -> prob of occurrence} }
    """
    if order == 0:
        res = defaultdict(float)
        for song in data:
            for i in song:
                res[i] += 1
        res = normalize(res)
        return res
    else:
        res = defaultdict(mydefaultdict)
        # Counting occurrences of transitions
        for song in data:
            for i in xrange(len(song) - order):
                hist = str(song[i:i + order])
                n = song[i + order]
                res[hist][n] += 1
        # Normalization
        for hist in res:
            res[hist] = normalize(res[hist])
        return res

def dic_argmax(d):
    maxi = 0
    argmax = None
    for k in d:
        if d[k] > maxi:
            maxi = d[k]
            argmax = k
    return argmax

def dic_sample(d):
    dn = normalize(d)
    u = random()
    cumulative = 0
    elts = sorted(d.keys())
    for k in elts:
        cumulative += dn[k]
        if cumulative > u:
            return k

def keys_subtract(d, x):
    res = defaultdict(float)
    for k in d:
        res[k-x] = d[k]
    return res

def tvDistance(p, q):
    """
    total variation distance
    """
    res = 0.
    for key in set(p.keys()).union(set(q.keys())):
        res += abs(p[key] - q[key])
    return 0.5 * res
