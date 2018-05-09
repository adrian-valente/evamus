import music21
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from music21Interface import seqs2stream
from tools import getSong


def key_analysis(dataset, dictionaries, report=None, per_measures=True, labels=None, slice_size=3):
    print("Starting key analysis...")
    keys = defaultdict(float)
    correlations = []
    changes = []
    switches = []
    for s in tqdm(range(len(dataset["dTseqs"]))):
        song = getSong(dataset, s)
        stream = seqs2stream(song, dictionaries)
        key_fit = stream.analyze('key')
        keys[key_fit.name] += 1
        correlations.append(key_fit.correlationCoefficient)
        if labels is not None:
            report.write("Song %s\n" % labels[s])
        if per_measures:
            ch, sw = perMeasureKeyAnalysis(stream, slice_size, report)
            changes.append(ch)
            switches.append(sw)

    if report is not None:
        report.write("Key analyses\n--------------\n\n")
        report.write("* Mean key correlation: {}\n".format(np.mean(correlations)))
        report.write("* Most frequent keys: \n")
        l = [(keys[k], k) for k in keys]
        l.sort()
        l.reverse()
        for i in range(min(5, len(l))):
            report.write(" - {} : {} songs\n".format(l[i][1], l[i][0]))
        report.write("* Mean percentage of changes: {:.2%}\n".format(np.mean(changes)))
        report.write("* Mean percentage of switches: {:.2%}\n".format(np.mean(switches)))
        report.write("\n\n\n")


def perMeasureKeyAnalysis(song, slice_size, report=None):
    measures = song.makeMeasures()
    key_fit_song = song.analyze('key')
    if report:
        report.write("Key fit for song: " + key_fit_song.name + '\n')
    changes = 0
    switches = 0
    if len(measures) > slice_size:
        for i in range(len(measures) - slice_size):
            slicem = measures[i:i+slice_size]
            try:
                key_fit = slicem.analyze('key')
            except music21.analysis.discrete.DiscreteAnalysisException:
                if report:
                    report.write("no key fit for slice %d\n" % i)
            else:
                if report:
                    report.write(key_fit.name+'\n')
                if key_fit.name != key_fit_song.name:
                    changes += 1
                if i > 0 and key_fit.name != previous_key_fit.name:
                    switches += 1
                previous_key_fit = key_fit

        changes /= float(len(measures) - slice_size)
        switches /= float(len(measures) - slice_size - 1)
        if report:
            report.write("fraction of changes:  {:.2%}\n".format(changes))
            report.write("fraction of switches: {:.2%}\n".format(switches))
    return changes, switches
