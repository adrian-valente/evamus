from utils.tools import trainsingleorder, dic_argmax, dic_sample
from math import log

class SingleNgram:
    """
    Simple fallback Ngram model on single sequence type
    """

    def __init__(self, order):
        self.order = order
        self.probs = []

    def train(self, data):
        for i in xrange(self.order+1):
            self.probs.append(trainsingleorder(data, i))

    def predict(self, hist):
        if len(hist) > self.order:
            hist = hist[-self.order:]

        # Fallback part
        l = len(hist)
        while l > 0 and str(hist) not in self.probs[l]:
            l -= 1
            hist = hist[1:]

        # Retrieve predictions
        if l == 0:
            return self.probs[0]
        else:
            return self.probs[l][str(hist)]



class INgram:
    """
    Ngram model assuming independence of sequences dT, t, and pitch
    :attr order: the order
    :attr models: a dictionary of underlying SingleNgram models for each sequence type
    """

    def __init__(self, order):
        self.order = order
        self.models = {'dTseqs': SingleNgram(order), 'tseqs': SingleNgram(order), 'pitchseqs': SingleNgram(order)}

    def preprocess(self, data):
        """
        :param data: dataset in Z format
        :return: flattened dataset (list of songs replaced by single sequence for each type)
        """
        return data

    def train(self, data):
        """
        :param data: preprocessed dataset
        """
        for key in self.models:
            self.models[key].train(data[key])
        return self.test(data, False)

    def probNote(self, hist, x):
        """
        :param hist: dictionary (in dataset format) giving previous notes
        :param x: a note x=(dT, t, pitch)
        :return: probability of note x given history hist
        """
        pdT = self.models['dTseqs'].predict(hist['dTseqs'])
        pt = self.models['tseqs'].predict(hist['tseqs'])
        ppitch = self.models['pitchseqs'].predict(hist['pitchseqs'])
        return pdT[str(x[0])] * pt[str(x[1])] * ppitch[str(x[2])]

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
        distr_dT = self.models['dTseqs'].predict(hist['dTseqs'])
        distr_t = self.models['tseqs'].predict(hist['tseqs'])
        distr_pitch = self.models['pitchseqs'].predict(hist['pitchseqs'])

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
            'accuracy_dT': 0., 'accuracy_t': 0., 'accuracy_pitch': 0., 'accuracy_global': 0.
        }
        n_songs = len(data['dTseqs'])
        total_length = 0
        for s in xrange(n_songs):
            l = len(data['dTseqs'][s])
            total_length += l
            for i, dT, t, p in zip(list(range(l)), data['dTseqs'][s], data['tseqs'][s], data['pitchseqs'][s]):
                self.testNote({key: data[key][s][max(i-self.order,0):i] for key in data}, (dT, t, p), metrics,
                              smoothing, smoothing_lambda)

        for key in metrics:
            metrics[key] /= total_length

        return metrics

    def generate(self, seed, N=200):
        """
        :param seed:
        :return:
        """
        ret = seed
        for i in range(N):
            for key in ret:
                hist = ret[key][max(0,i-self.order):]
                distr = self.models.predict(hist)
                ret[key].append(dic_sample(distr))
        return ret

