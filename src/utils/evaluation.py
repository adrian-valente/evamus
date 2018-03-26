import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from utils.midiparser import getDictionaries, parseFolder, cleanDic, writeMIDI
from utils.preprocessing import toZ, toMIDI


def preprocess(datapath, writeMIDI=False):
    """
    Parses a datapath and returns a preprocessed dataset, either splitted or not
    :param datapath:
    :param writeMIDI:
    :return:
    """
    dictionaries = getDictionaries()
    dataset = parseFolder(datapath, dictionaries)
    dictionaries = cleanDic(dataset, dictionaries)
    xdTs, xTs, xPs, dTvocsize, Tvocsize, pitchvocsize = toZ(dataset, dictionaries)
    dataset = {'dTseqs': xdTs, 'tseqs': xTs, 'pitchseqs': xPs}

    if writeMIDI:
        dtseq, Tseq, pitchseq = toMIDI(xdTs[0], xTs[0], xPs[0], dictionaries)
        for dt, t, p in zip(dtseq, Tseq, pitchseq):
            print(dictionaries['duration_text'][dictionaries['dtseqs'].index(dt)],
                  dictionaries['duration_text'][dictionaries['Tseqs'].index(t)],
                  dictionaries['pitch_text'][dictionaries['pitchseqs'].index(p)])
        writeMIDI(dtseq, Tseq, pitchseq, path='../data/', label='example')

    return dataset, (dTvocsize, Tvocsize, pitchvocsize), dictionaries


def split(dataset, k):
    """
    :param dataset: a dictionary of list of sequences containing the lists 'dtseqs', 'Tseqs', 'pitchseq'
    :param k: the number of folds
    :return: a list of dictionaries {'train':, 'test':} where train and test have the same format as dataset
    (note: train and test are views, and not copies of the original dataset)
    """
    n = len(dataset['tseqs'])
    ret = []
    for i in xrange(k):
        ret.append({'train':{}, 'test':{}})
        for key in dataset:
            ret[-1]['train'][key] =  dataset[key][:i*(n//k)] + dataset[key][min((i+1)*(n//k), n):]
            ret[-1]['test'][key] = dataset[key][i*(n//k):min((i+1)*(n//k), n)]
    return ret


def evaluate(generated_data, reference_data):
    print "evaluated not yet defined"
    return None


def plot_metric(metrics, metric, title="", uniform=None):
    """
    Plots a bar chart comparing the value of metric on different models with error bars
    :param values: a dictionary mapping model name -> a sequence of values of the metric
    """
    values = {}
    for model in metrics:
        if metric in metrics[model]:
            values[model] = metrics[model][metric]

    x_i = np.arange(len(values))
    y = [np.mean(values[model]) for model in values]
    yerr = [np.std(values[model]) for model in values]
    print yerr

    plt.xticks(x_i, list(values.keys()))
    plt.title(title)
    plt.bar(x_i, y, yerr=yerr)
    if uniform is not None:
        plt.axhline(uniform, color='red', alpha=0.5)

    plt.show()


def analyze_chords(real_data, gen_data):
    real_dis = defaultdict(float)
    gen_dis = defaultdict(float)

    for s, song in enumerate(real_data['dTseqs']):
        for i, dT in enumerate(song):
            if dT == 0 and i > 0:
                diff = abs(real_data['pitchseqs'][s][i] - real_data['pitchseqs'][s][i-1])%12
                real_dis[diff] += 1

    for s, song in enumerate(gen_data['dTseqs']):
        for i, dT in enumerate(song):
            if dT == 0 and i > 0:
                diff = abs(gen_data['pitchseqs'][s][i] - gen_data['pitchseqs'][s][i-1])%12
                gen_dis[diff] += 1

    print real_dis

    keys = sorted(set(real_dis.keys()).union(set(gen_dis.keys())))
    idx = np.arange(len(keys))
    plt.xticks(idx, keys)

    ax = plt.subplot(211)
    ax.bar(idx, [real_dis[k] for k in keys])

    ax = plt.subplot(212)
    ax.bar(idx, [gen_dis[k] for k in keys])
    plt.show()


def analyze_intervals(real_data, gen_data):
    real_dis = defaultdict(float)
    gen_dis = defaultdict(float)

    for s, song in enumerate(real_data['dTseqs']):
        for i, dT in enumerate(song):
            if dT != 0 and i > 0:
                diff = abs(real_data['pitchseqs'][s][i] - real_data['pitchseqs'][s][i - 1])%12
                real_dis[diff] += 1

    for s, song in enumerate(gen_data['dTseqs']):
        for i, dT in enumerate(song):
            if dT != 0 and i > 0:
                diff = abs(gen_data['pitchseqs'][s][i] - gen_data['pitchseqs'][s][i - 1])%12
                gen_dis[diff] += 1

    print real_dis

    keys = sorted(set(real_dis.keys()).union(set(gen_dis.keys())))
    idx = np.arange(len(keys))
    plt.xticks(idx, keys)

    ax = plt.subplot(211)
    ax.bar(idx, [real_dis[k] for k in keys])

    ax = plt.subplot(212)
    ax.bar(idx, [gen_dis[k] for k in keys])
    plt.show()

