from utils.tools import trainsingleorder, dic_argmax, dic_sample
from math import log
from utils.preprocessing import toMIDI
from utils.midiparser import writeMIDI

class FUNgram:
    """
    Stores a Feature Unrolled Ngram model

    :attr tstart: first index of t values sector
    :attr pstart: first index of pitch values sector
    :attr nValues: total number of values in the 3 sectors (thus a value is in [0..nValues)
    :attr order: order of the model (0 for a simple frequency counter, 1 for normal Markov)
    :attr probs: sequence of fallback Ngram models (for 0 a dict {str(value) -> prob of occurrence}
                 for bigger orders a dict {str(history) -> {str(value) -> prob of occurrence} })
    """

    def __init__(self, sizes, order):
        self.tstart = sizes[0]
        self.pstart = self.tstart + sizes[1]
        self.nValues = self.pstart + sizes[2]
        self.order = order
        self.probs = []

    def preprocess(self, dataset):
        """
        Flattens the dataset in feature unrolled format
        Assumes the dataset is in Z format (integer values starting at zero for each sequence)
        :return: a list of sequences of unrolled notes (dT0, t0, p0, dT1, ....)
        """
        unrolled = []
        for dTs, ts, ps in zip(dataset['dTseqs'], dataset['tseqs'], dataset['pitchseqs']):
            unrolled.append([])
            for i in xrange(len(dTs)):
                unrolled[-1].append(dTs[i])
                unrolled[-1].append(self.tstart + ts[i])
                unrolled[-1].append(self.pstart + ps[i])
        return unrolled

    def train(self, data):
        """
        :param data: Unrolled dataset (apply preprocess first)
        """
        for i in range(self.order+1):
            self.probs.append(trainsingleorder(data, i))
        return self.test(data, False)

    def predict(self, hist, return_fallback=False):
        """
        Gives probability distribution of next value given hist
        :param hist: an unrolled history of previous values
        :return: a defaultdict giving the probability distribution
        """
        if len(hist) > self.order:
            hist = hist[-self.order:]
        # Fallback part
        l = len(hist)
        while l > 0 and str(hist) not in self.probs[l]:
            l -= 1
            hist = hist[1:]

        # Retrieve predictions
        if l == 0:
            if return_fallback:
                return self.probs[0], l
            else:
                return self.probs[0]
        else:
            if return_fallback:
                return self.probs[l][str(hist)], l
            else:
                return self.probs[l][str(hist)]

    def probNote(self, hist, x):
        """
        :param hist: unrolled history
        :param x: a note x=(dT, t, pitch)
        :return: the probability of note x given hist
        """
        dT, t, p = x
        pdT = self.predict(hist)[str(dT)]  # Pr(dT | hist)
        hist.append(dT)
        pt = self.predict(hist)[str(t)]   # Pr(t | dT, hist)
        hist.append(t)
        ppitch = self.predict(hist)[str(p)]  # Pr(p | t, dT, hist)
        return pdT * pt * ppitch

    def testNote(self, hist, x, metrics, smoothing=True, smoothing_lambda=0.05, n_dT=25, n_t=35, n_pitch=54):
        """
        CAUTION: FOR INTERNAL USE OF THE CLASS ONLY (uses side effects)
        This function modifies the metrics dictionary by adding the log-likelihoods and accuracy for note x
        for note x given history hist
        :param hist: a sequence of notes
        :param x: a tuple (dT, t, p) representing the current note
        :param metrics: a dictionary of metrics to modify
        :return: nothing (modifies metrics directly)
        """
        dT, t, p = x
        distr_dT, fall_dT = self.predict(hist, True)
        hist.append(dT)
        distr_t, fall_t = self.predict(hist, True)
        hist.append(t)
        distr_pitch, fall_pitch = self.predict(hist, True)

        metrics['fall_dT'] += fall_dT
        metrics['fall_t'] += fall_t
        metrics['fall_pitch'] += fall_pitch

        pdT = distr_dT[str(x[0])]
        pt = distr_t[str(x[1])]
        ppitch = distr_pitch[str(x[2])]
        pglobal = pdT * pt * ppitch

        if smoothing:
            pdT = smoothing_lambda * (1./n_dT) + (1 - smoothing_lambda) * pdT
            pt = smoothing_lambda * (1. / n_t) + (1 - smoothing_lambda) * pt
            ppitch = smoothing_lambda * (1. / n_pitch) + (1 - smoothing_lambda) * ppitch
            pglobal = smoothing_lambda * (1./(n_dT * n_t * n_pitch)) + (1-smoothing_lambda) * pglobal

        metrics['LL_dT'] += log(pdT) if pdT > 0 else -10000
        metrics['LL_t'] += log(pt) if pt > 0 else -10000
        metrics['LL_pitch'] += log(ppitch) if ppitch > 0 else -10000
        metrics['LL_global'] += log(pglobal) if pglobal > 0 else -10000

        accurate_dT = 1 if dic_argmax(distr_dT) == str(x[0]) else 0
        accurate_t = 1 if dic_argmax(distr_t) == str(x[1]) else 0
        accurate_pitch = 1 if dic_argmax(distr_pitch) == str(x[2]) else 0

        metrics['accuracy_dT'] += accurate_dT
        metrics['accuracy_t'] += accurate_t
        metrics['accuracy_pitch'] += accurate_pitch
        metrics['accuracy_global'] += accurate_t * accurate_pitch * accurate_dT

    def test(self, data, smoothing=True, smoothing_lambda=0.05):
        """
        Computes metrics on test set data
        :param data: unrolled dataset
        :return: a dictionary of metrics
        """
        metrics = {
            'LL_dT': 0., 'LL_t': 0., 'LL_pitch': 0., 'LL_global': 0.,
            'accuracy_dT': 0., 'accuracy_t': 0., 'accuracy_pitch': 0., 'accuracy_global': 0.,
            'fall_dT': 0., 'fall_t': 0., 'fall_pitch': 0.
        }
        total_length = 0
        for song in data:
            total_length += len(song)//3
            for i in xrange(0, len(song), 3):
                hist = song[max(0,i-self.order):i]
                x = (song[i], song[i+1], song[i+2])
                self.testNote(hist, x, metrics, smoothing, smoothing_lambda)

        for key in metrics:
            metrics[key] /= total_length

        return metrics

    def generate_data(self, seed=[0], n_songs=20, N=599, write_MIDI=False, dictionaries=None):
        """
        :param seed:
        :return:
        """
        dataset = {'dTseqs': [], 'tseqs': [], 'pitchseqs': []}
        for i in range(n_songs):
            song = self.postprocess(self.generate_song(seed, N))
            dataset['dTseqs'].append(song['dTseqs'])
            dataset['tseqs'].append(song['tseqs'])
            dataset['pitchseqs'].append(song['pitchseqs'])
            if write_MIDI:
                dtseq, tseq, pseq = toMIDI(song['dTseqs'], song['tseqs'], song['pitchseqs'], dictionaries)
                writeMIDI(dtseq, tseq, pseq, path='../data/generated',
                          label='fungram-order'+str(self.order)+'-'+str(i))

        return dataset

    def generate_song(self, seed=[0], N=599):
        """
        :param seed:
        :return:
        """
        assert (N+len(seed)) % 3 == 0
        ret = seed[:]
        for i in range(N):
            hist = ret[max(0, i-self.order):]
            distr = self.predict(hist)
            ret.append(int(dic_sample(distr)))
        return ret

    def postprocess(self, seq):
        song = {'dTseqs':[], 'tseqs':[], 'pitchseqs':[]}
        for i,elt in enumerate(seq):
            if i%3 == 0:
                song['dTseqs'].append(elt)
            elif i%3 == 1:
                song['tseqs'].append(elt - self.tstart)
            elif i%3 == 2:
                song['pitchseqs'].append(elt - self.pstart)
        return song


