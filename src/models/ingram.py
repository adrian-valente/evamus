from utils.tools import trainsingleorder
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
        flattened = {}
        for key in data:
            flattened[key] = [x for s in data[key] for x in s]
        return flattened

    def train(self, data):
        """
        :param data: preprocessed dataset
        """
        for key in self.models:
            self.models[key].train(data[key])
        return self.test(data)

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

    def test(self, data, smoothing=False, smoothing_lambda=0.05, uniformP=1./(25*25*54)):
        """
        :param data: preprocessed data format
        :return: likelihood, log-likelihood
        """
        logL = 0.
        l = len(data['dTseqs'])
        for i, dT, t, p in zip(list(range(l)), data['dTseqs'], data['tseqs'], data['pitchseqs']):
            prob = self.probNote({key: data[key][max(i-self.order,0):i] for key in data}, (dT, t, p))

            if smoothing:
                prob = smoothing_lambda * uniformP + (1 - smoothing_lambda) * prob

            if prob > 0:
                logL += log(prob)
            else:
                logL -= 100000

        return logL/l

