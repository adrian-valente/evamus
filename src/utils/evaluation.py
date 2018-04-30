import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import seaborn as sns
from collections import defaultdict
from math import log
from tools import dic_argmax, normalize, tvDistance

import matplotlib.style
matplotlib.style.use('ggplot')


def compare_models(splitted, modelsDic):
    """
    Trains a set of models doing cross-validation on a dataset, and computes metrics.
    :param splitted: a standard splitted dataset: sequence of dicts {'train': train_data, 'test': test_data} where
    train_data and test_data have standard dataset format
    :param modelsDic: a dictionary {model_name (string) -> (model_class (a class), keyword_arguments (a dictionary)) }
    :return: a dict models {model_name -> list of trained models} and a dict metrics {model_name -> model_metrics}
    where model_metrics can be either a dict {metric -> list of values computed on each fold} or a dict
    {'train': train_metrics, 'test': test_metrics} where train_metrics and test_metrics have
    """
    models = {}
    metrics = {}
    for m in modelsDic:
        model_class = modelsDic[m][0]
        model_args = modelsDic[m][1]
        models[m], metrics[m] = evaluate_model(splitted, model_class, **model_args)
    return models, metrics


def evaluate_model(splitted, model_class, **kargs):
    models = []
    metrics = {'train': {}, 'test': {}}
    for i in range(len(splitted)):
        model = model_class(**kargs)
        train_data = splitted[i]['train']
        test_data = splitted[i]['test']
        train_preds = model.train(train_data)

        train_metrics = compute_metrics(train_preds, train_data)
        for key in train_metrics:
            if key not in metrics['train']:
                metrics['train'][key] = []
            metrics['train'][key].append(train_metrics[key])
        print train_metrics['NLL_global']

        test_preds = model.predict(test_data)
        test_metrics = compute_metrics(test_preds, test_data)
        for key in test_metrics:
            if key not in metrics["test"]:
                metrics['test'][key] = []
            metrics['test'][key].append(test_metrics[key])
        print test_metrics['NLL_global']

        models.append(model)
    return models, metrics


def plot_metric(metrics, metric, title="", uniform=None):
    """
    Plots a bar chart comparing the value of metric on different models with error bars
    :param values: a dictionary mapping model name -> a sequence of values of the metric
    """
    keys = []
    y = []
    yerr = []
    c = []
    handles = []
    i = 0

    for model in metrics:
        if 'train' in metrics[model] and metric in metrics[model]['train']:
            keys.append('train')
            y.append(np.mean(metrics[model]['train'][metric]))
            yerr.append(np.std(metrics[model]['train'][metric]))
            c.append('C'+str(i)[-1])
            handles.append(mpatches.Patch(color=c[-1], label=model))
            keys.append('test')
            y.append(np.mean(metrics[model]['test'][metric]))
            yerr.append(np.std(metrics[model]['test'][metric]))
            c.append('C' + str(i)[-1])
            i += 1
        elif metric in metrics[model]:
            keys.append(model + '')
            y.append(np.mean(metrics[model][metric]))
            yerr.append(np.std(metrics[model][metric]))
            c.append('C' + str(i)[-1])
            handles.append(mpatches.Patch(color=c[-1], label=model))
            i += 1

    x_i = np.arange(len(y))

    plt.figure(figsize=(10,7))

    plt.xticks(x_i, keys, fontsize=20)
    plt.title(title, size='xx-large')
    plt.bar(x_i, y, yerr=yerr, color=c)
    plt.legend(handles=handles, fontsize=20)
    if uniform is not None:
        plt.axhline(uniform, color='red', alpha=0.5)

    plt.show()


def nLogLikelihoods(predictions, data, smoothing=True, smoothing_lambda=0.05, n_dT=25, n_t=25, n_pitch=54):
    """
    Computes log-likelihoods for the separate sequences dT, t, p as well as for the notes
    :param predictions: a standard predictions dataset
    :return:
    """
    llglobal = 0.
    lldT = 0.
    llt = 0.
    llpitch = 0.
    total_length = 0

    for s in range(len(predictions["dTseqs"])):
        l = len(predictions["dTseqs"][s])
        total_length += l

        for i in range(l):
            dT, t, p = data["dTseqs"][s][i], data["tseqs"][s][i], data["pitchseqs"][s][i]
            pdT = predictions["dTseqs"][s][i][dT]
            pt = predictions["tseqs"][s][i][t]
            ppitch = predictions["pitchseqs"][s][i][p]
            pglobal = pdT * pt * ppitch

            if smoothing:
                pdT = smoothing_lambda * (1./n_dT) + (1 - smoothing_lambda) * pdT
                pt = smoothing_lambda * (1./n_t) + (1 - smoothing_lambda) * pt
                ppitch = smoothing_lambda * (1./n_pitch) + (1 - smoothing_lambda) * ppitch
                pglobal = smoothing_lambda * (1./(n_dT*n_t*n_pitch)) + (1 - smoothing_lambda) * pglobal

            lldT += log(pdT)
            llt += log(pt)
            llpitch += log(ppitch)
            llglobal += log(pglobal)

    return -lldT/total_length, -llt/total_length, -llpitch/total_length, -llglobal/total_length


def accuracies(predictions, data):
    """
    Computes accuracy for all sequences dT, t, pitch and for the notes
    :param predictions:
    :param data:
    :return:
    """
    aglobal = 0.
    adT = 0.
    at = 0.
    apitch = 0.
    total_length = 0.

    for s in range(len(data["dTseqs"])):
        l = len(data["dTseqs"][s])
        total_length += l-1  #note: we ignore first note

        for i in range(1,l):
            dT, t, p = data["dTseqs"][s][i], data["tseqs"][s][i], data["pitchseqs"][s][i]
            if isinstance(predictions["dTseqs"][s][i], dict):
                good_dT = 1 if (dic_argmax(predictions["dTseqs"][s][i]) == dT) else 0
                good_t = 1 if (dic_argmax(predictions["tseqs"][s][i]) == t) else 0
                good_p = 1 if (dic_argmax(predictions["pitchseqs"][s][i]) == p) else 0
            elif isinstance(predictions["dTseqs"][s][i], list):
                good_dT = 1 if (np.argmax(predictions["dTseqs"][s][i]) == dT) else 0
                good_t = 1 if (np.argmax(predictions["tseqs"][s][i]) == t) else 0
                good_p = 1 if (np.argmax(predictions["pitchseqs"][s][i]) == p) else 0

            adT += good_dT
            at += good_t
            apitch += good_p
            aglobal += good_dT * good_t * good_p

    return adT/total_length, at/total_length, apitch/total_length, aglobal/total_length


def compute_metrics(predictions, data):
    d = {
            'NLL_dT': [], 'NLL_t': [], 'NLL_pitch': [], 'NLL_global': [],
            'accuracy_dT': [], 'accuracy_t': [], 'accuracy_pitch': [], 'accuracy_global': []
    }

    lldT, llt, llp, llg = nLogLikelihoods(predictions, data)
    d['NLL_dT'] = lldT
    d['NLL_t'] = llt
    d['NLL_pitch'] = llp
    d['NLL_global'] = llg

    adT, at, ap, ag = accuracies(predictions, data)
    d['accuracy_dT'] = adT
    d['accuracy_t'] = at
    d['accuracy_pitch'] = ap
    d['accuracy_global'] = ag

    return d


def preanalysis_chords(data, make_plot=False, plot_fp=None):
    """
    Computes the distribution of intervals in chords for data
    :param make_plot: boolean, whether to realize a plot
    :param plot_fp: filename where to save the plot
    :returns: the disribution as a defaultdict (semitones -> frequencies)
    """
    distr = defaultdict(float)
    for s, song in enumerate(data['dTseqs']):
        cur_chord = set()
        for i, dT in enumerate(song):
            if dT == 0:
                p = data['pitchseqs'][s][i]
                for x in cur_chord:
                    diff = abs(p-x) % 12
                    distr[diff] += 1
                cur_chord.add(p)
            else:
                cur_chord = {data['pitchseqs'][s][i]}
    normalize(distr)

    if make_plot:
        keys = sorted(set(distr.keys()))
        idx = np.arange(len(keys))
        fig, ax = plt.subplots() 
        fig.suptitle("Chords decomposition - real distribution")
        ax.bar(idx, [distr[k] for k in keys])
        ax.set_xticks(idx, keys)
        plt.savefig(plot_fp)
    return distr


def analyze_chords(real_data, gen_data, title="Chord decomposition", real_dis=None, 
                   show_plot=False, plot_fp=None):
    """
    Analysis of intervals between notes in a same chord
    :param real_data: a dataset of the reference corpus (can be None if real_dis is already given)
    :param gen_data: dataset of studied corpus
    :param title: title for the plot
    :param real_dis: if given, the distribution on the reference corpus will not be recomputed
    :param show_plot: True to directly show plot
    :param plot_fp: if not None, where to save the plot
    :returns: total variation distance between the real and the generated distribution
    """
    gen_dis = defaultdict(float)

    if real_dis is None:
        real_dis = defaultdict(float)
        for s, song in enumerate(real_data['dTseqs']):
            cur_chord = set()
            for i, dT in enumerate(song):
                if dT == 0:
                    p = real_data['pitchseqs'][s][i]
                    for x in cur_chord:
                        diff = abs(p-x) % 12
                        real_dis[diff] += 1
                    cur_chord.add(p)
                else:
                    cur_chord = {real_data['pitchseqs'][s][i]}

    for s, song in enumerate(gen_data['dTseqs']):
        cur_chord = set()
        for i, dT in enumerate(song):
            if dT == 0:
                p = gen_data['pitchseqs'][s][i]
                for x in cur_chord:
                    diff = abs(p - x) % 12
                    gen_dis[diff] += 1
                cur_chord.add(p)
            else:
                cur_chord = {gen_data['pitchseqs'][s][i]}
    
    # normalization
    real_dis = normalize(real_dis)
    gen_dis = normalize(gen_dis)

    # Make plot
    keys = sorted(set(real_dis.keys()).union(set(gen_dis.keys())))
    fig, ax = plt.subplots()
    df = pd.DataFrame({'intervals': list(range(12)),
                       'frequency': [real_dis[i] for i in range(12)],
                       'distribution': 'real'})
    df2 = pd.DataFrame({'intervals': list(range(12)),
                       'frequency': [gen_dis[i] for i in range(12)],
                       'distribution': 'generated'})
    df = pd.concat([df, df2])
    sns.barplot(x='intervals', y='frequency', hue='distribution', data=df, ax=ax)
    fig.suptitle(title)

    if show_plot:
        fig.show()
    elif plot_fp is not None:
        plt.savefig(plot_fp)

    # Compute statistical distance
    return tvDistance(real_dis, gen_dis)


def preanalysis_intervals(data, make_plot=False, plot_fp=None):
    """
    Computes the distribution of intervals for all successive notes
    :param make_plot: boolean, whether to realize a plot
    :param plot_fp: filename where to save the plot
    :returns: the disribution as a defaultdict (semitones -> frequencies)
    """
    distr = defaultdict(float)

    for s, song in enumerate(data['dTseqs']):
        for i, dT in enumerate(song):
            if i > 0:
                diff = abs(data['pitchseqs'][s][i]-p) % 12
                distr[diff] += 1
            p = data['pitchseqs'][s][i]             

    if make_plot:
        keys = sorted(set(distr.keys()))
        idx = np.arange(len(keys))
        fig, ax = plt.subplots() 
        fig.suptitle("Intervals decomposition - Real distribution")
        ax.bar(idx, [distr[k] for k in keys])
        ax.set_xticks(idx, keys)
        plt.savefig(plot_fp)
    return distr


def analyze_intervals(real_data, gen_data, title="Interval decomposition", real_dis=None, 
                      show_plot=False, plot_fp=None):
    """
    Analysis of intervals between successive notes
    :param real_data: a dataset of the reference corpus (can be None if real_dis is already given)
    :param gen_data: dataset of studied corpus
    :param title: title for the plot
    :param real_dis: if given, the distribution on the reference corpus will not be recomputed
    :param show_plot: True to directly show plot
    :param plot_fp: if not None, where to save the plot
    :returns: total variation distance between the real and the generated distribution
    """
    gen_dis = defaultdict(float)

    if real_dis is None:
        real_dis = defaultdict(float)
        for s, song in enumerate(real_data['dTseqs']):
            for i, dT in enumerate(song):
                if i > 0:
                    diff = abs(real_data['pitchseqs'][s][i]-p) % 12
                    real_dis[diff] += 1
                p = real_data['pitchseqs'][s][i]

    for s, song in enumerate(gen_data['dTseqs']):
        for i, dT in enumerate(song):
            if i > 0:
                diff = abs(gen_data['pitchseqs'][s][i]-p) % 12
                gen_dis[diff] += 1
            p = gen_data['pitchseqs'][s][i]

    # Normalize
    real_dis = normalize(real_dis)
    gen_dis = normalize(gen_dis)

    # Make plot
    fig, ax = plt.subplots()
    df = pd.DataFrame({'intervals': list(range(12)),
                       'frequency': [real_dis[i] for i in range(12)],
                       'distribution': 'real'})
    df2 = pd.DataFrame({'intervals': list(range(12)),
                       'frequency': [gen_dis[i] for i in range(12)],
                       'distribution': 'generated'})
    df = pd.concat([df, df2])
    sns.barplot(x='intervals', y='frequency', hue='distribution', data=df, ax=ax)
    fig.suptitle(title)

    if show_plot:
        fig.show()
    elif plot_fp is not None:
        plt.savefig(plot_fp)
    
    # Compute statistical distance
    return tvDistance(real_dis, gen_dis)


def analyze_transitions_singletype(real_seqs, gen_seqs, size, title, seqname):
    transitions_real = np.zeros((size, size))
    for song in real_seqs:
        for i in range(len(song) - 1):
            transitions_real[song[i], song[i + 1]] += 1

    transitions_gen = np.zeros((size, size))
    for song in gen_seqs:
        for i in range(len(song) - 1):
            transitions_gen[song[i], song[i + 1]] += 1

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    fig.suptitle(title, size='xx-large', y=1.04)
    ax1.matshow(transitions_real)
    ax1.set_title("Real distribution", fontsize=25)
    ax1.set_xlabel("$"+seqname+"_i$", fontsize=25)
    ax1.set_ylabel("$"+seqname+"_{i+1}$", fontsize=25)
    ax2.matshow(transitions_gen)
    ax2.set_title("Generated distribution", size="xx-large")
    ax2.set_xlabel("$"+seqname+"_i$", fontsize=25)
    ax2.set_ylabel("$"+seqname+"_{i+1}$", fontsize=25)
    fig.show()


def analyze_transitions(real_data, gen_data, sizes, title_prefix=""):
    analyze_transitions_singletype(real_data["pitchseqs"], gen_data["pitchseqs"], sizes[2],
                                   title_prefix+"Transition probabilites for pitch", seqname="p")
    analyze_transitions_singletype(real_data["dTseqs"], gen_data["dTseqs"], sizes[0],
                                   title_prefix+"Transition probabilities for dT", seqname="dT")
    analyze_transitions_singletype(real_data["tseqs"], gen_data["tseqs"], sizes[1],
                                   title_prefix+"Transition probabilities for T", seqname="T")
