from collections import defaultdict

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
    :return: if order is 0 a dict {str(value) -> prob of occurrence}
             for bigger orders a dict {str(history) -> {str(value) -> prob of occurrence} }
    """
    if order == 0:
        res = defaultdict(float)
        for i in data:
            res[str(i)] += 1
        res = normalize(res)
        return res
    else:
        res = defaultdict(mydefaultdict)
        # Counting occurrences of transitions
        for i in xrange(len(data) - order):
            hist = str(data[i:i + order])
            n = str(data[i + order])
            res[hist][n] += 1
        # Normalization
        for hist in res:
            res[hist] = normalize(res[hist])
        return res
